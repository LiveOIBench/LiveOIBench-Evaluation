#!/usr/bin/env python3
"""
Evaluation driver for competitive programming tasks.
Generates JSON and/or CSV outputs at four levels:
1. Problem-Level: per-problem, per-model detailed results
2. Contest-Level: per contest-split, best solutions per problem
3. Competition-Level: per competition-year-model summary (one row per problem, plus TOTAL row)
4. Aggregated-Level: per model aggregated summary across competition-years
Additionally generates a status-statistics CSV with counts per status (including empty solutions) per model.

Empty solution files (zero-byte) are skipped during evaluation and counted in the status statistics.

Usage:
  python evaluation_driver.py --competitions IOI --years 2024 \
    --solution_types llm correct --llm_models Qwen2.5-Coder-7B-Instruct \
    --workers 5 --max_solutions 10 --output-format both
"""
import argparse
import os
import json
import time
import sys
import traceback
from datetime import datetime
from enum import Enum
from concurrent.futures import ThreadPoolExecutor
import threading
import hashlib
import csv
from itertools import groupby
from collections import defaultdict
from judges.judge import Judge
from judges.problem import Problem
from judges.result_type import ResultType

global_counter = 0
counter_lock = threading.Lock()


def determine_result_type(score_info, detailed_results):
    if "compile_output" in score_info:
        return ResultType.COMPILATION_ERROR
    if "exception" in score_info:
        return ResultType.UNKNOWN
    if score_info.get("ace", False):
        return ResultType.ACCEPTED
    if detailed_results:
        if any(r.get("exit_code", 0) == -9 for r in detailed_results):
            return ResultType.TIME_LIMIT_EXCEEDED
        if any(r.get("memory_limit_exceeded", False) or r.get("exit_code", 0) == -6 for r in detailed_results):
            return ResultType.MEMORY_LIMIT_EXCEEDED
        if any(r.get("exit_code", 0) != 0 for r in detailed_results):
            return ResultType.RUNTIME_ERROR
    return ResultType.WRONG_ANSWER


def parse_arguments():
    parser = argparse.ArgumentParser(description="Evaluate solutions and generate structured outputs.")
    parser.add_argument("--competitions", nargs="+", default=["IOI"] )
    parser.add_argument("--years", nargs="+", default=["2024"] )
    parser.add_argument("--rounds", nargs="+", default=None )
    parser.add_argument("--tasks", nargs="+", default=None )
    parser.add_argument("--task_types", nargs="+", default=None )
    parser.add_argument("--solution_types", nargs="+")
    parser.add_argument("--llm_models", nargs="+", default=None )
    parser.add_argument("--llm_solutions_dir", type=str,
                        help="Directory containing generated LLM solutions")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--verbose", action="store_true" )
    parser.add_argument("--stop_on_failure", action="store_true",
                        help="Stop evaluating test cases on the first failure in subtask order")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing benchmark problem data")
    parser.add_argument("--evaluation_dir", type=str, required=True,
                        help="Directory containing evaluation resources (judges, configs, etc.)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory where evaluation outputs will be written")
    parser.add_argument("--cache_dir", type=str, required=True,
                        help="Directory to store evaluation cache files")
    parser.add_argument("--max_solutions", type=int, default=1 )
    parser.add_argument("--use_cache", action="store_true", default=True )
    parser.add_argument("--reeval", action="store_true" )
    parser.add_argument("--output_format", choices=["json","csv","both"],
                        default="both" )
    args = parser.parse_args()
    args.solution_types = [
        "llm" if st.lower() in ("llm", "llm_solutions") else st
        for st in args.solution_types
    ]
    raw_years = args.years
    if any(y.lower() == "all" for y in raw_years):
        year_set = set()
        for comp in args.competitions:
            comp_dir = os.path.join(args.data_dir, comp)
            if not os.path.isdir(comp_dir):
                continue
            for entry in os.listdir(comp_dir):
                if entry.isdigit() and os.path.isdir(os.path.join(comp_dir, entry)):
                    year_set.add(entry)
        yrs = sorted(year_set)
    else:
        yrs = []
        for y in raw_years:
            if '-' in y:
                try:
                    s,e = map(int, y.split('-'))
                    yrs.extend(str(i) for i in range(s,e+1))
                except:
                    yrs.append(y)
            else:
                yrs.append(y)
    args.years = yrs
    if "llm" in args.solution_types and not args.llm_models:
        parser.error("--llm_models required when llm in solution_types")

    if "llm" in args.solution_types and not args.llm_solutions_dir:
        parser.error("--llm_solutions_dir required when llm in solution_types")

    if "llm" in args.solution_types and "all" in args.llm_models:
        args.llm_models = [f for f in os.listdir(args.llm_solutions_dir)
                           if os.path.isdir(os.path.join(args.llm_solutions_dir, f))]

    return args


def discover_problems(args):
    problems, count = [], 0
    for comp in args.competitions:
        for year in args.years:
            rounds = args.rounds or []
            base_year = os.path.join(args.data_dir, comp, year)
            if not args.rounds:
                try: rounds = os.listdir(base_year)
                except: continue
            for rnd in rounds:
                base = os.path.join(base_year, rnd)
                meta_f = os.path.join(base, 'meta_info.json')
                
                if not os.path.exists(meta_f): continue
                meta = json.load(open(meta_f))
                for split, tasks in meta.items():
                    for task in tasks:
                        if args.tasks and task not in args.tasks: continue
                        task_dir = os.path.join(base, task)
                        cfg_f = os.path.join(task_dir, 'problem.json')
                        ttype = 'batch'
                        if os.path.exists(cfg_f):
                            cfg = json.load(open(cfg_f))
                            ttype = cfg.get('task_type') or 'batch'
                        if args.task_types and ttype.lower() not in args.task_types: continue
                        pid = f"{comp}-{year}-{rnd}-{task}"
                        problems.append({
                            'competition': comp,'year': year,'round': rnd,
                            'split': split,'task': task,'dir': task_dir,'id': pid
                        })
                        count += 1
    print(f"Found {count} tasks.")
    return problems


def get_solution_files(problem_info, stype, args):
    try:
        prob = Problem(problem_info['dir'], problem_info['task'],
                       problem_info['year'], problem_info['competition'],
                       problem_info['round'], problem_info['split'])
        sols = []
        if stype == 'llm':
            for m in args.llm_models:
                mdir = os.path.join(args.llm_solutions_dir,m,
                                    problem_info['competition'], problem_info['year'],
                                    problem_info['round'], problem_info['task'], 'codes')
                if os.path.isdir(mdir):
                    for f in os.listdir(mdir):
                        if f.endswith('.cpp'):
                            seed = f.split('_')[-1].replace('.cpp','')
                            if int(seed) >= args.max_solutions:
                                continue
                            sols.append({'path':os.path.join(mdir,f),'model':m,'name':f})
            return sols
        else:
            for p in prob.get_code_solution('cpp', stype):
                sols.append({'path':p,'model':'original','name':os.path.basename(p)})
            return sols[:args.max_solutions]
    except Exception as e:
        print(f"Error gathering solutions {problem_info['id']}: {e}")
        return []


def get_cache_key(pid, sol_path):
    return pid + '_' + hashlib.md5(open(sol_path,'rb').read()).hexdigest()


def get_cached_result(key, args):
    f = os.path.join(args.cache_dir, key+'.json')
    if os.path.exists(f):
        try: return json.load(open(f))
        except: return None
    return None


def save_to_cache(key, res, args):
    open(os.path.join(args.cache_dir, key+'.json'),'w').write(json.dumps(res))


def evaluate_solution(judge, pinfo, sol_info, args, idx, total):
    global global_counter
    with counter_lock:
        global_counter +=1
        cnt = global_counter
    pid, mname, model = pinfo['id'], sol_info['name'], sol_info['model']
    task_name = pinfo['task']
    competition = pinfo['competition']
    year = pinfo['year']
    round_name = pinfo['round']
    
    print(f"[{cnt}/{total}] {competition} {year} {round_name} | Task: {task_name} | Solution: {mname} | Model: {model}")
    
    key = get_cache_key(pid, sol_info['path'])
    if args.use_cache and not args.reeval:
        
        cr = get_cached_result(key, args)
        if cr: 
            # Print result for cached solution
            print(f"  → Result: {cr['status']} | Score: {cr['score']} | Tests: {cr['tests_passed']*100:.2f}%")
            return cr
    try:
        prob = Problem(pinfo['dir'], pinfo['task'], pinfo['year'],
                       pinfo['competition'], pinfo['round'], pinfo['split'])
        score_info, details = judge.judge(prob, sol_info['path'], 
                                         verbose=args.verbose, 
                                         save_output=False,
                                         generate_gold_output=False,
                                         max_workers=args.workers,
                                         stop_on_failure=args.stop_on_failure)
        times = [r.get('cpu_time', 0) for r in details if r.get('cpu_time')]
        mems  = [r.get('memory',   0) for r in details if r.get('memory')]
        max_mem  = max(mems)  if mems  else 0
        max_time = max(times) if times else 0
        rt = determine_result_type(score_info, details)
        res = {
            'problem_id': pid, 'solution_file': mname, 'model': model,
            'status': rt.name, 'status_code': int(rt),
            'score': score_info.get('score', 0),
            'tests_passed': score_info.get('tests_passed', 0),
            'execution_time': max_time, 'memory_usage': max_mem,
            'subtasks': score_info.get('subtasks', {}),
            'details': details  # Save the detailed test results to the cache
        }
        if 'compile_output' in score_info:
            res['compile_output'] = score_info['compile_output']
        save_to_cache(key, res, args)
        # Print result for newly evaluated solution
        print(f"  → Result: {res['status']} | Score: {res['score']} | Tests: {res['tests_passed']*100:.2f}%")
        return res
    except Exception as e:
        print(f"Error eval {mname}: {e}")
        res = {
            'problem_id': pid, 'solution_file': mname, 'model': model,
            'status': ResultType.UNKNOWN.name, 'status_code': int(ResultType.UNKNOWN),
            'score': 0, 'tests_passed': 0,
            'execution_time': 0, 'memory_usage': 0, 'subtasks': {}
        }
        print(f"  → Result: {res['status']} | Score: 0 | Tests: 0.00%")
        return res


def print_evaluation_summary(results, ps_map):
    """Print a summary of all evaluation results"""
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    
    # Group results by model
    model_results = defaultdict(list)
    for pid, solution_results in results.items():
        for result in solution_results:
            model_results[result['model']].append(result)
    
    # Print summary for each model
    for model, model_data in sorted(model_results.items()):
        print(f"\n== Model: {model} ==")
        
        # Count solutions by status
        status_counts = defaultdict(int)
        total_score = 0
        total_tests_passed = 0
        max_score = 0
        
        for result in model_data:
            status_counts[result['status']] += 1
            total_score += result['score']
            total_tests_passed += result['tests_passed']
            max_score += 100  # Assuming each problem has a max score of 100
        
        # Print status breakdown
        print(f"Solutions evaluated: {len(model_data)}")
        for status, count in sorted(status_counts.items()):
            print(f"  {status}: {count} ({count/len(model_data)*100:.1f}%)")
        
        # Print score summary
        print(f"Total score: {total_score} / {max_score} ({total_score/max_score*100:.2f}%)")
        print(f"Average tests passed: {total_tests_passed/len(model_data)*100:.2f}%")
        accepted_rate = status_counts.get('ACCEPTED', 0) / len(model_data) * 100
        print(f"Solution acceptance rate: {accepted_rate:.2f}%")
    
    print("\n" + "="*80)


def print_problem_summaries(results, ps_map):
    """Print summaries for each problem"""
    print("\n" + "="*80)
    print("PROBLEM SUMMARIES")
    print("="*80)
    
    # Group results by problem
    for pid, problem_results in sorted(results.items()):
        problem_info = ps_map[pid]['problem_info']
        task_name = problem_info['task']
        competition = problem_info['competition']
        year = problem_info['year']
        round_name = problem_info['round']
        
        print(f"\n== Problem: {task_name} ({competition} {year} {round_name}) ==")
        
        # Group by model
        model_results = defaultdict(list)
        for result in problem_results:
            model_results[result['model']].append(result)
        
        print(f"Total solutions evaluated: {len(problem_results)}")
        
        # For each model, find the best solution
        for model, solutions in sorted(model_results.items()):
            best = sorted(
                solutions,
                key=lambda s: (-s['score'], -s['tests_passed'], s['execution_time'])
            )[0]
            
            print(f"  Model: {model}")
            print(f"    Best solution: {best['solution_file']}")
            print(f"    Status: {best['status']}")
            print(f"    Score: {best['score']}")
            print(f"    Tests passed: {best['tests_passed']*100:.2f}%")
            print(f"    Time: {best['execution_time']:.3f}s | Memory: {best['memory_usage']} KB")
            
            # If there are multiple solutions, show statistics
            if len(solutions) > 1:
                status_counts = defaultdict(int)
                for s in solutions:
                    status_counts[s['status']] += 1
                
                print(f"    Solution statuses: ", end="")
                status_strings = [f"{status}: {count}" for status, count in sorted(status_counts.items())]
                print(", ".join(status_strings))
    
    print("\n" + "="*80)


def get_problem_max_score(pid, ps_map):
    """Get the maximum possible score for a problem"""
    try:
        pinfo = ps_map[pid]['problem_info']
        prob = Problem(pinfo['dir'], pinfo['task'], pinfo['year'],
                       pinfo['competition'], pinfo['round'], pinfo['split'])
        return prob.get_total_points()
    except Exception:
        # Fallback to 100 if we can't determine the max score
        raise Exception(f"Failed to get max score for problem {pid}")


def generate_json(results, ps_map, args):
    out = os.path.join(args.output_dir, 'all_results.json')
    json.dump(results, open(out,'w'), indent=2)


def generate_csv(results, ps_map, args):
    import os, csv
    from collections import defaultdict

    base_dir = args.output_dir
    os.makedirs(base_dir, exist_ok=True)

    # Prepare best_results for later levels
    best_results = {}
    for pid, recs in results.items():
        for model in set(r['model'] for r in recs):
            mrecs = [r for r in recs if r['model'] == model]
            best = sorted(
                mrecs,
                key=lambda s: (
                    -s['score'], -s['tests_passed'],
                    s['execution_time'], s['memory_usage'], s['status']
                )
            )[0]
            best_results[(pid, model)] = best

    # Group by model, competition, year for level 3 & 4 & status stats
    comp_model = defaultdict(list)
    for (pid, model), best in best_results.items():
        info = ps_map[pid]['problem_info']
        comp_model[(model, info['competition'], info['year'])].append((pid, best))

    # === 1) Problem-Level: per-problem, per-model detailed results ===
    for pid, data in ps_map.items():
        pinfo = data['problem_info']
        prob = Problem(pinfo['dir'], pinfo['task'], pinfo['year'],
                       pinfo['competition'], pinfo['round'], pinfo['split'])
        subtasks = prob.get_subtasks()
        # Get the maximum possible score for this problem
        max_score = prob.get_total_points()
        for model in set(r['model'] for r in results.get(pid, [])):
            # Get results for this model and sort them by solution_file
            recs = sorted(
                [r for r in results.get(pid, []) if r['model'] == model],
                key=lambda r: r['solution_file']  # Sort by solution filename
            )
            
            out_dir = os.path.join(base_dir, model,
                                   pinfo['competition'], pinfo['year'], pinfo['round'])
            os.makedirs(out_dir, exist_ok=True)
            subkeys = sorted({k for r in recs for k in r['subtasks'].keys()})
            fields = ['solution', 'status', 'score', 'relative_score', '%test passed', 'time', 'memory']
            fields += [f'subtask_{k}' for k in subkeys]
            fname = f"{pinfo['task']}_{pinfo['split']}_{model}_problem.csv"
            with open(os.path.join(out_dir, fname), 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                for r in recs:  # This is now sorted by solution_file
                    # Calculate relative score as percentage of max score
                    relative_score = (r['score'] / max_score * 100) if max_score > 0 else 0
                    row = {
                        'solution': r['solution_file'],
                        'status': r['status'],
                        'score': r['score'],
                        'relative_score': f"{relative_score:.2f}%",
                        '%test passed': f"{r['tests_passed']*100:.2f}%",
                        'time': f"{r['execution_time']:.3f}",
                        'memory': r['memory_usage']
                    }
                    for k in subkeys:
                        val = r['subtasks'].get(k, 0)
                        if isinstance(val, dict):
                            val = val.get('score', 0)
                        row[f'subtask_{k}'] = val
                    writer.writerow(row)

    # === 2) Contest-Level: per contest-split, best solutions per problem ===
    contest_groups = defaultdict(list)
    for (pid, model), best in best_results.items():
        info = ps_map[pid]['problem_info']
        key = (model, info['competition'], info['year'], info['round'], info['split'])
        contest_groups[key].append(pid)
    for (model, comp, year, rnd, split), pids in contest_groups.items():
        out_dir = os.path.join(base_dir, model, comp, year, rnd)
        os.makedirs(out_dir, exist_ok=True)
        fname = f"{split}_contest_best.csv"
        fields = ['problem', 'model', 'status', 'score', 'relative_score', '%test passed', 'time', 'memory']
        with open(os.path.join(out_dir, fname), 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for pid in pids:
                best = best_results[(pid, model)]
                info = ps_map[pid]['problem_info']
                max_score = get_problem_max_score(pid, ps_map)
                relative_score = (best['score'] / max_score * 100) if max_score > 0 else 0
                writer.writerow({
                    'problem': info['task'],
                    'model': best['model'],
                    'status': best['status'],
                    'score': best['score'],
                    'relative_score': f"{relative_score:.2f}%",
                    '%test passed': f"{best['tests_passed']*100:.2f}%",
                    'time': f"{best['execution_time']:.3f}",
                    'memory': best['memory_usage']
                })

    # === 3) Competition-Level: per competition-year summary & status_stats (all solutions) ===
    for (model, comp, year), items in comp_model.items():
        out_dir = os.path.join(base_dir, model, comp, year)
        os.makedirs(out_dir, exist_ok=True)
        # competition summary
        fname = f"{comp}_{year}_{model}_competition_summary.csv"
        subkeys = sorted({k for _, r in items for k in r['subtasks'].keys()})
        fields = ['problem', 'status', 'score', 'relative_score', '%test passed', 'time', 'memory', 'pass_rate']
        fields += [f'subtask_{k}' for k in subkeys]
        total_score = total_pct = total_acc = 0
        with open(os.path.join(out_dir, fname), 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for pid, r in items:
                problem = ps_map[pid]['problem_info']['task']
                pct = r['tests_passed'] * 100
                total_score += r['score']; total_pct += pct
                if r['status'] == 'ACCEPTED': total_acc += 1
                max_score = get_problem_max_score(pid, ps_map)
                relative_score = (r['score'] / max_score * 100) if max_score > 0 else 0
                row = {
                    'problem': problem,
                    'status': r['status'],
                    'score': r['score'],
                    'relative_score': f"{relative_score:.2f}%",
                    '%test passed': f"{pct:.2f}%",
                    'time': f"{r['execution_time']:.3f}",
                    'memory': r['memory_usage'],
                    'pass_rate': ''
                }
                for k in subkeys:
                    val = r['subtasks'].get(k, 0)
                    if isinstance(val, dict): val = val.get('score', 0)
                    row[f'subtask_{k}'] = val
                writer.writerow(row)
            # total row
            n = len(items)
            avg_pct = total_pct/n if n else 0
            pr = total_acc/n*100 if n else 0
            summary = {k: '' for k in fields}
            summary.update({'problem': 'TOTAL', 'score': total_score,
                             '%test passed': f"{avg_pct:.2f}%", 'pass_rate': f"{pr:.2f}%"})
            writer.writerow(summary)

        # competition subtask acceptance (1 accepted, 0 failed, -1 not exist)
        # Collect union of subtask keys across problems in this competition-year for this model
        subtask_keys = set()
        pid_to_problem = {}
        for pid, r in items:
            pinfo = ps_map[pid]['problem_info']
            prob = Problem(pinfo['dir'], pinfo['task'], pinfo['year'], pinfo['competition'], pinfo['round'], pinfo['split'])
            pid_to_problem[pid] = prob
            try:
                subtask_keys.update(prob.get_subtasks().keys())
            except Exception:
                # Fallback to keys present in results if problem subtasks unavailable
                subtask_keys.update(r.get('subtasks', {}).keys())
        subtask_keys = sorted(subtask_keys)

        # Build and write acceptance table
        accept_fname = f"{comp}_{year}_{model}_competition_subtask_accept.csv"
        accept_fields = ['problem'] + [f'subtask_{k}' for k in subtask_keys]
        with open(os.path.join(out_dir, accept_fname), 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=accept_fields)
            writer.writeheader()
            for pid, r in items:
                pinfo = ps_map[pid]['problem_info']
                prob = pid_to_problem[pid]
                prob_subtasks = {}
                try:
                    prob_subtasks = prob.get_subtasks()
                except Exception:
                    prob_subtasks = {}

                row = {'problem': pinfo['task']}
                for k in subtask_keys:
                    if k not in prob_subtasks:
                        row[f'subtask_{k}'] = -1
                        continue
                    achieved = r['subtasks'].get(k, 0)
                    if isinstance(achieved, dict):
                        achieved = achieved.get('score', 0)
                    try:
                        total = prob_subtasks[k].get('score', None) if isinstance(prob_subtasks[k], dict) else None
                    except Exception:
                        total = None
                    if total is None:
                        # If total unknown though subtask exists, treat non-accept as 0 unless fully matched by chance
                        row[f'subtask_{k}'] = 1 if achieved else 0
                    else:
                        row[f'subtask_{k}'] = 1 if achieved == total else 0
                writer.writerow(row)
        # status_stats: count all solutions including empties
        status_file = os.path.join(out_dir, 'status_stats.csv')
        statuses = [st.name for st in ResultType] + ['empty solutions']
        counts = dict.fromkeys(statuses, 0)
        for pid, data in ps_map.items():
            info = data['problem_info']
            if info['competition'] != comp or info['year'] != year:
                continue
            for sol in data['solutions']:
                if sol['model'] != model:
                    continue
                path = sol['path']
                try:
                    if os.path.getsize(path) == 0:
                        counts['empty solutions'] += 1
                        continue
                except OSError:
                    counts['empty solutions'] += 1
                    continue
                match = next((r for r in results.get(pid, [])
                              if r['model']==model and r['solution_file']==sol['name']), None)
                if match:
                    counts[match['status']] += 1
                else:
                    counts['empty solutions'] += 1
        with open(status_file, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['competition', 'year', 'model'] + statuses)
            w.writerow([comp, year, model] + [counts[s] for s in statuses])

    # === 4) Aggregated-Level: per-model aggregated_summary.csv with update ===
    for model in sorted({m for m,_,_ in comp_model.keys()}):
        model_dir = os.path.join(base_dir, model)
        os.makedirs(model_dir, exist_ok=True)
        agg_path = os.path.join(model_dir, 'aggregated_summary.csv')
        existing = {}
        if os.path.exists(agg_path):
            with open(agg_path, 'r', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    existing[row['competition_year']] = row
        for (m, comp, year), items in comp_model.items():
            if m != model:
                continue
            total_score = sum(r['score'] for _,r in items)
            pct = sum(r['tests_passed']*100 for _,r in items)
            n = len(items)
            avg_pct = pct/n if n else 0
            avg_time = sum(r['execution_time'] for _,r in items)/n if n else 0
            avg_mem = sum(r['memory_usage'] for _,r in items)/n if n else 0
            acc = sum(1 for _,r in items if r['status']=='ACCEPTED')
            pr = acc/n*100 if n else 0
            key = f"{comp}-{year}"
            existing[key] = {
                'competition_year': key,
                'model': model,
                'total_score': str(total_score),
                'avg %passed': f"{avg_pct:.2f}%",
                'avg_time': f"{avg_time:.2f}",
                'avg_mem': f"{avg_mem:.2f}",
                'pass_rate': f"{pr:.2f}%"
            }
        rows = [v for k,v in existing.items() if k != 'ALL_COMPETITIONS']
        ts_all = sum(float(v['total_score']) for v in rows)
        pct_all = sum(float(v['avg %passed'].strip('%')) for v in rows)
        time_all = sum(float(v['avg_time']) for v in rows)
        mem_all = sum(float(v['avg_mem']) for v in rows)
        cnt = len(rows)
        existing['ALL_COMPETITIONS'] = {
            'competition_year': 'ALL_COMPETITIONS',
            'model': model,
            'total_score': str(int(ts_all)),
            'avg %passed': f"{(pct_all/cnt if cnt else 0):.2f}%",
            'avg_time': f"{(time_all/cnt if cnt else 0):.2f}",
            'avg_mem': f"{(mem_all/cnt if cnt else 0):.2f}",
            'pass_rate': f"{(sum(float(v['pass_rate'].strip('%')) for v in rows)/cnt if cnt else 0):.2f}%"
        }
        with open(agg_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'competition_year', 'model', 'total_score',
                'avg %passed', 'avg_time', 'avg_mem', 'pass_rate'
            ])
            writer.writeheader()
            for key in sorted(existing.keys(), key=lambda k: (k != 'ALL_COMPETITIONS', k)):
                writer.writerow(existing[key])

    # === 5) Cross-Year Summary: aggregate all years in the same competition ===
    for model in sorted({m for m,_,_ in comp_model.keys()}):
        for comp in sorted({c for (m, c, _) in comp_model.keys() if m == model}):
            model_comp_dir = os.path.join(base_dir, model, comp)
            os.makedirs(model_comp_dir, exist_ok=True)
            summary_path = os.path.join(model_comp_dir, f"{comp}_years_summary.csv")
            
            # Load existing data if available
            existing_data = {}
            if os.path.exists(summary_path):
                try:
                    with open(summary_path, 'r', newline='') as f:
                        reader = csv.reader(f)
                        headers = next(reader)  # Read headers
                        for row in reader:
                            if row and len(row) >= 4 and row[0] != 'ALL_YEARS':
                                existing_data[row[0]] = row
                except Exception as e:
                    print(f"Warning: Could not read existing year summary: {e}")
            
            # Update with current data
            current_years = sorted([y for (m, c, y) in comp_model.keys() if m == model and c == comp])
            agg_scores = agg_pcts = agg_acc = agg_cnt = 0
            
            # Process current years
            for year in current_years:
                items = comp_model[(model, comp, year)]
                score_sum = sum(r['score'] for _, r in items)
                pct_sum = sum(r['tests_passed'] * 100 for _, r in items)
                n = len(items)
                avg_pct = pct_sum / n if n else 0
                acc = sum(1 for _, r in items if r['status'] == 'ACCEPTED')
                pr = acc / n * 100 if n else 0
                existing_data[year] = [year, str(score_sum), f"{avg_pct:.2f}%", f"{pr:.2f}%"]
            
            # Calculate aggregates for ALL_YEARS including both existing and new data
            for year, row in existing_data.items():
                try:
                    # Parse values from the row
                    score = float(row[1])
                    pct = float(row[2].strip('%'))
                    pr = float(row[3].strip('%'))
                    
                    # Add to aggregates
                    agg_scores += score
                    agg_pcts += pct
                    if pr > 0:  # If pass rate exists, add to the count
                        agg_acc += pr
                        agg_cnt += 1
                except (ValueError, IndexError):
                    print(f"Warning: Invalid row data for {year}: {row}")
            
            # Calculate overall averages
            overall_pct = agg_pcts / len(existing_data) if existing_data else 0
            overall_pr = agg_acc / agg_cnt if agg_cnt else 0
            
            # Write updated data
            with open(summary_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['year', 'total_score', 'avg %passed', 'pass_rate'])
                
                # Write all years sorted
                for year in sorted(existing_data.keys()):
                    writer.writerow(existing_data[year])
                    
                # Write ALL_YEARS summary
                writer.writerow(['ALL_YEARS', str(int(agg_scores)), f"{overall_pct:.2f}%", f"{overall_pr:.2f}%"])

    # === New 6) Model-Level Status Statistics (update) ===
    for model in sorted({m for (m,_,_) in comp_model.keys()}):
        model_dir = os.path.join(base_dir, model)
        os.makedirs(model_dir, exist_ok=True)
        status_path = os.path.join(model_dir, 'status_stats.csv')
        statuses = [st.name for st in ResultType] + ['empty solutions']
        # load existing
        existing_stats = {}
        if os.path.exists(status_path):
            with open(status_path, 'r', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    cnts = {s: int(row[s]) for s in statuses}
                    existing_stats[row['competition_year']] = cnts
        # update per competition-year
        for (m, comp, year), items in comp_model.items():
            if m != model:
                continue
            counts = dict.fromkeys(statuses, 0)
            # iterate all generated solutions for this comp-year
            for pid, data in ps_map.items():
                info = data['problem_info']
                if info['competition'] != comp or info['year'] != year:
                    continue
                for sol in data['solutions']:
                    if sol['model'] != model:
                        continue
                    path = sol['path']
                    try:
                        if os.path.getsize(path) == 0:
                            counts['empty solutions'] += 1
                            continue
                    except OSError:
                        counts['empty solutions'] += 1
                        continue
                    match = next((r for r in results.get(pid, [])
                                  if r['model']==model and r['solution_file']==sol['name']), None)
                    if match:
                        counts[match['status']] += 1
                    else:
                        counts['empty solutions'] += 1
            existing_stats[f"{comp}-{year}"] = counts
        # recompute ALL_COMPETITIONS
        all_counts = dict.fromkeys(statuses, 0)
        for cnts in existing_stats.values():
            for s in statuses:
                all_counts[s] += cnts[s]
        existing_stats['ALL_COMPETITIONS'] = all_counts
        # write back
        with open(status_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['competition_year'] + statuses)
            for key in sorted(existing_stats.keys(), key=lambda k: (k!='ALL_COMPETITIONS', k)):
                row = [key] + [existing_stats[key][s] for s in statuses]
                writer.writerow(row)

def generate_comprehensive_results_csv(results, ps_map, args):
    """
    Generate comprehensive CSV files with all evaluation results - one per model.
    Ensures solution files are properly sorted and updates existing results.
    """
    # Group results by model
    model_results = {}
    for pid, solution_results in results.items():
        for result in solution_results:
            model = result['model']
            model_results.setdefault(model, []).append((pid, result))
    
    # Create a comprehensive CSV for each model
    for model, model_data in model_results.items():
        # Create model directory
        model_dir = os.path.join(args.output_dir, model)
        os.makedirs(model_dir, exist_ok=True)
        
        output_path = os.path.join(model_dir, 'all_evaluation_results.csv')
        
        # Collect all fields we want to include
        fields = [
            'problem_id', 'model',
            'solution_file', 'status', 'score', 'relative_score', 'tests_passed',
            'execution_time', 'memory_usage'
        ]
        
        # Check if file exists and load existing results
        existing_results = {}
        if os.path.exists(output_path):
            try:
                with open(output_path, 'r', newline='') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        key = (row['problem_id'], row['solution_file'])
                        existing_results[key] = row
            except Exception as e:
                print(f"Warning: Could not read existing results: {e}")
        
        # Update existing results with new data
        for pid, result in model_data:
            key = (pid, result['solution_file'])
            max_score = get_problem_max_score(pid, ps_map)
            relative_score = (result['score'] / max_score * 100) if max_score > 0 else 0
            row = {
                'problem_id': pid,
                'model': result['model'],
                'solution_file': result['solution_file'],
                'status': result['status'],
                'score': result['score'],
                'relative_score': f"{relative_score:.2f}%",
                'tests_passed': f"{result['tests_passed']*100:.2f}%",
                'execution_time': f"{result['execution_time']:.3f}",
                'memory_usage': result['memory_usage']
            }
            existing_results[key] = row
        # Write combined results
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            
            # Sort data by problem_id and then by solution_file for consistent ordering
            sorted_results = sorted(existing_results.values(), 
                                    key=lambda x: (x['problem_id'], x['solution_file']))
            
            for row in sorted_results:
                writer.writerow(row)
        
        print(f"Comprehensive results for {model} updated in {output_path}")

def generate_comparison_tables(results, ps_map, args):
    """
    Generate comparison tables at three levels:
    1. Contest level: All models in each contest round (with individual task scores)
    2. Competition level: All models averaged across all years in each competition
    3. Global level: All models averaged across all competitions
    
    Only updates data for models that are currently being evaluated.
    """
    # Identify current models being evaluated
    current_models = set()
    for pid, records in results.items():
        for record in records:
            current_models.add(record['model'])
    
    print(f"Updating comparison tables for models: {', '.join(sorted(current_models))}")
    
    # Get best result for each (problem, model) combination for current models
    best_results = {}
    for pid, recs in results.items():
        for model in set(r['model'] for r in recs):
            mrecs = [r for r in recs if r['model'] == model]
            best = sorted(
                mrecs,
                key=lambda s: (
                    -s['score'], -s['tests_passed'],
                    s['execution_time'], s['memory_usage'], s['status']
                )
            )[0]
            best_results[(pid, model)] = best
    
    # Group results by contest and model
    contest_model_results = {}  # (comp, year, round) -> {model -> stats}
    for (pid, model), res in best_results.items():
        info = ps_map[pid]['problem_info']
        contest_key = (info['competition'], info['year'], info['round'])
        if contest_key not in contest_model_results:
            contest_model_results[contest_key] = {}
        if model not in contest_model_results[contest_key]:
            contest_model_results[contest_key][model] = []
        contest_model_results[contest_key][model].append((pid, res))
    
    # 1. Generate contest-level comparison tables with individual task scores
    for (comp, year, round_name), model_data in contest_model_results.items():
        out_dir = os.path.join(args.output_dir, "all", comp, year, round_name)
        os.makedirs(out_dir, exist_ok=True)
        
        # Collect all tasks in this contest
        tasks = set()
        task_info = {}  # Map problem_id to task_name for easier lookup
        
        for model in current_models:
            if model in model_data:
                for pid, _ in model_data[model]:
                    task_name = ps_map[pid]['problem_info']['task']
                    tasks.add(task_name)
                    task_info[pid] = task_name
        
        tasks = sorted(tasks)
        
        # Load existing data if available
        output_path = os.path.join(out_dir, f"{comp}_{year}_{round_name}_model_comparison.csv")
        existing_data = {}
        if os.path.exists(output_path):
            try:
                with open(output_path, 'r', newline='') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        existing_data[row['Model']] = row
            except Exception as e:
                print(f"Warning: Could not read existing comparison data: {e}")
        
        # Calculate statistics for current models in this contest
        for model in current_models:
            if model in model_data:
                items = model_data[model]
                
                # Calculate aggregate statistics
                score_sum = sum(r['score'] for _, r in items)
                pct_sum = sum(r['tests_passed'] * 100 for _, r in items)
                time_sum = sum(r['execution_time'] for _, r in items)
                mem_sum = sum(r['memory_usage'] for _, r in items)
                acc_count = sum(1 for _, r in items if r['status'] == 'ACCEPTED')
                n = len(items)
                
                # Calculate relative score sum
                rel_score_sum = 0
                for pid, r in items:
                    max_score = get_problem_max_score(pid, ps_map)
                    rel_score_sum += (r['score'] / max_score * 100) if max_score > 0 else 0
                
                # Initialize the row with aggregate statistics
                row = {
                    'Model': model,
                    'Total Score': f"{score_sum:.2f}",
                    'Avg Relative Score': f"{rel_score_sum / n if n else 0:.2f}%",
                    'Avg Tests Passed': f"{pct_sum / n if n else 0:.2f}%",
                    'Pass Rate': f"{acc_count / n * 100 if n else 0:.2f}%",
                    'Avg Time': f"{time_sum / n if n else 0:.3f}",
                    'Avg Memory': f"{mem_sum / n if n else 0:.2f}",
                    'Problem Count': str(n)
                }
                
                # Add individual task scores and relative scores
                for task in tasks:
                    # Find the result for this task if it exists
                    task_result = None
                    task_pid = None
                    for pid, res in items:
                        if task_info[pid] == task:
                            task_result = res
                            task_pid = pid
                            break
                    
                    # Add the score and relative score to the row
                    if task_result:
                        row[f"{task}"] = f"{task_result['score']:.2f}"
                        max_score = get_problem_max_score(task_pid, ps_map)
                        rel_score = (task_result['score'] / max_score * 100) if max_score > 0 else 0
                        row[f"{task}_rel"] = f"{rel_score:.2f}%"
                    else:
                        row[f"{task}"] = "N/A"
                        row[f"{task}_rel"] = "N/A"
                
                existing_data[model] = row
        
        # Write contest-level comparison CSV with task scores
        fields = ['Model', 'Total Score', 'Avg Relative Score', 'Avg Tests Passed', 'Pass Rate', 'Avg Time', 'Avg Memory', 'Problem Count']
        # Add individual task score columns and relative score columns
        for task in tasks:
            fields.append(f"{task}")
            fields.append(f"{task}_rel")
            
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            # Sort all models by performance
            sorted_models = sorted(existing_data.keys(), 
                                  key=lambda m: (
                                      -float(existing_data[m]['Total Score']),
                                      -float(existing_data[m]['Pass Rate'].rstrip('%')), 
                                      float(existing_data[m]['Avg Time'])
                                  ))
            for model in sorted_models:
                writer.writerow(existing_data[model])
    
    # 2. Generate competition-level summary tables aggregated across all years
    comp_model_stats = {}  # comp -> {model -> stats}
    
    for (comp, year, round_name), model_data in contest_model_results.items():
        comp_key = comp  # Just use the competition name as the key
        if comp_key not in comp_model_stats:
            comp_model_stats[comp_key] = {}
        
        for model in current_models:
            if model in model_data:
                items = model_data[model]
                if model not in comp_model_stats[comp_key]:
                    comp_model_stats[comp_key][model] = {
                        'score_sum': 0, 'rel_score_sum': 0, 'pct_sum': 0, 'time_sum': 0, 
                        'mem_sum': 0, 'acc_count': 0, 'problem_count': 0
                    }
                
                # Aggregate statistics for this model across rounds and years
                stats = comp_model_stats[comp_key][model]
                stats['score_sum'] += sum(r['score'] for _, r in items)
                stats['pct_sum'] += sum(r['tests_passed'] * 100 for _, r in items)
                stats['time_sum'] += sum(r['execution_time'] for _, r in items)
                stats['mem_sum'] += sum(r['memory_usage'] for _, r in items)
                stats['acc_count'] += sum(1 for _, r in items if r['status'] == 'ACCEPTED')
                stats['problem_count'] += len(items)
                # Calculate relative score sum
                for pid, r in items:
                    max_score = get_problem_max_score(pid, ps_map)
                    stats['rel_score_sum'] += (r['score'] / max_score * 100) if max_score > 0 else 0
    
    # Write competition-level summary CSVs aggregated across all years
    for comp, model_stats in comp_model_stats.items():
        out_dir = os.path.join(args.output_dir, "all", comp)
        os.makedirs(out_dir, exist_ok=True)
        
        # Load existing data if available
        output_path = os.path.join(out_dir, f"{comp}_all_years_summary.csv")
        existing_data = {}
        if os.path.exists(output_path):
            try:
                with open(output_path, 'r', newline='') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        existing_data[row['Model']] = row
            except Exception as e:
                print(f"Warning: Could not read existing competition summary: {e}")
        
        # Update competition-level statistics for current models
        for model in current_models:
            if model in model_stats:
                stats = model_stats[model]
                n = stats['problem_count']
                existing_data[model] = {
                    'Model': model,
                    'Total Score': f"{stats['score_sum']:.2f}",
                    'Avg Relative Score': f"{stats['rel_score_sum'] / n if n else 0:.2f}%",
                    'Avg Tests Passed': f"{stats['pct_sum'] / n if n else 0:.2f}%",
                    'Pass Rate': f"{stats['acc_count'] / n * 100 if n else 0:.2f}%",
                    'Avg Time': f"{stats['time_sum'] / n if n else 0:.3f}",
                    'Avg Memory': f"{stats['mem_sum'] / n if n else 0:.2f}",
                    'Problem Count': str(n)
                }
        
        # Write competition summary
        fields = ['Model', 'Total Score', 'Avg Relative Score', 'Avg Tests Passed', 'Pass Rate', 'Avg Time', 'Avg Memory', 'Problem Count']
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            # Sort all models by performance
            sorted_models = sorted(existing_data.keys(), 
                                  key=lambda m: (
                                      -float(existing_data[m]['Total Score']),
                                      -float(existing_data[m]['Pass Rate'].rstrip('%')), 
                                      float(existing_data[m]['Avg Time'])
                                  ))
            for model in sorted_models:
                writer.writerow(existing_data[model])
    
    # 3. Generate global summary table aggregating across all competitions
    # Load existing data if available
    out_dir = os.path.join(args.output_dir, "all")
    os.makedirs(out_dir, exist_ok=True)
    global_path = os.path.join(out_dir, "all_competitions_summary.csv")
    existing_global = {}
    
    if os.path.exists(global_path):
        try:
            with open(global_path, 'r', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    existing_global[row['Model']] = row
        except Exception as e:
            print(f"Warning: Could not read existing global summary: {e}")
    
    # Calculate global statistics for current models
    all_model_stats = {model: {'score_sum': 0, 'rel_score_sum': 0, 'pct_sum': 0, 'time_sum': 0, 
                              'mem_sum': 0, 'acc_count': 0, 'problem_count': 0} 
                      for model in current_models}
    for comp, model_stats in comp_model_stats.items():
        for model in current_models:
            if model in model_stats:
                stats = model_stats[model]
                global_stats = all_model_stats[model]
                
                # Aggregate statistics across all competitions
                global_stats['score_sum'] += stats['score_sum']
                global_stats['rel_score_sum'] += stats['rel_score_sum']
                global_stats['pct_sum'] += stats['pct_sum']
                global_stats['time_sum'] += stats['time_sum']
                global_stats['mem_sum'] += stats['mem_sum']
                global_stats['acc_count'] += stats['acc_count']
                global_stats['problem_count'] += stats['problem_count']
    
    # Update global summary for current models
    for model, stats in all_model_stats.items():
        n = stats['problem_count']
        if n > 0:  # Only update if we have data
            existing_global[model] = {
                'Model': model,
                'Total Score': f"{stats['score_sum']:.2f}",  # Changed from Avg Score to Total Score
                'Avg Relative Score': f"{stats['rel_score_sum'] / n if n else 0:.2f}%",
                'Avg Tests Passed': f"{stats['pct_sum'] / n if n else 0:.2f}%",
                'Pass Rate': f"{stats['acc_count'] / n * 100 if n else 0:.2f}%",
                'Avg Time': f"{stats['time_sum'] / n if n else 0:.3f}",
                'Avg Memory': f"{stats['mem_sum'] / n if n else 0:.2f}",
                'Problem Count': str(n)
            }
    
    # Write global summary
    fields = ['Model', 'Total Score', 'Avg Relative Score', 'Avg Tests Passed', 'Pass Rate', 'Avg Time', 'Avg Memory', 'Problem Count']
    with open(global_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        # Sort all models by performance
        sorted_models = sorted(existing_global.keys(), 
                              key=lambda m: (
                                  -float(existing_global[m]['Total Score']),  # Sort by Total Score instead of Avg Score
                                  -float(existing_global[m]['Pass Rate'].rstrip('%')), 
                                  float(existing_global[m]['Avg Time'])
                              ))
        for model in sorted_models:
            writer.writerow(existing_global[model])

    print(f"Updated comparison tables for {len(current_models)} model(s) at all levels")

def main():
    args = parse_arguments()
    print(args.competitions)
    os.makedirs(args.output_dir, exist_ok=True)
    # Create the cache directory
    os.makedirs(args.cache_dir, exist_ok=True)
    judge = Judge(args.evaluation_dir)
    start = time.time()
    print(f"=== Start at {datetime.now()} ===")
    problems = discover_problems(args)

    # Collect solutions, skipping empty files
    all_solutions = []
    ps_map = {}
    total = 0
    for p in problems:
        sols = []
        for st in args.solution_types:
            sols.extend(get_solution_files(p, st, args))
        ps_map[p['id']] = {'problem_info': p, 'solutions': sols}
        if len(sols) == 0:
            print(f"Warning: No solutions found for {p['id']}")
            continue
        for sol in sols:
            try:
                if os.path.getsize(sol['path']) == 0:
                    # skip empty solution file
                    continue
            except OSError:
                continue
            all_solutions.append((p, sol))
            total += 1
    print(f"Total non-empty solutions to evaluate: {total}")
    results = {}
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [
            (pinfo['id'], ex.submit(evaluate_solution, judge, pinfo, sol, args, i+1, total))
            for i, (pinfo, sol) in enumerate(all_solutions)
        ]
        for pid, fut in futures:
            res = fut.result()
            results.setdefault(pid, []).append(res)

    # Print problem summaries
    print_problem_summaries(results, ps_map)
    
    # Print overall evaluation summary
    print_evaluation_summary(results, ps_map)

    if args.output_format in ('json','both'):
        generate_json(results, ps_map, args)
    if args.output_format in ('csv','both'):
        generate_csv(results, ps_map, args)
        # Add this line to generate the comprehensive CSV
        generate_comprehensive_results_csv(results, ps_map, args)
        # Add this line to generate the comparison tables
        if args.llm_models:
            generate_comparison_tables(results, ps_map, args)
        
    print(f"=== Done in {(time.time()-start):.2f}s ===")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(1)
    except Exception:
        traceback.print_exc()
        sys.exit(1)
