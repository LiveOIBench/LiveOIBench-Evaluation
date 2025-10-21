import argparse
import os
from judges.judge import Judge
from judges.problem import Problem

def main():
    parser = argparse.ArgumentParser(
        description="Run judge evaluation on a single solution for a specified problem."
    )
    parser.add_argument("--competition", type=str, default="IOI",
                        help="Competition name (default: IOI)")
    parser.add_argument("--year", type=str, default="2024",
                        help="Year (default: 2024)")
    parser.add_argument("--round", type=str, default="contest",
                        help="Round name (default: contest)")
    parser.add_argument("--task", type=str, required=True,
                        help="Task name (default: nile)")
    parser.add_argument("--solution_file", type=str, required=True,
                        help="Full path to the solution file to evaluate")
    parser.add_argument("--problem_folder", type=str, required=True,
                        help="Root directory for benchmark problems")
    parser.add_argument("--evaluation_folder", type=str, required=True,
                        help="Directory with evaluation resources")
    parser.add_argument("--stop_on_failure", dest="stop_on_failure", action="store_true",)
    
    # Add boolean flags for verbose
    parser.add_argument("--verbose", dest="verbose", action="store_true",help="Enable verbose mode")

    
    # Add boolean flags for save_output
    parser.add_argument("--save_output", dest="save_output", action="store_true",
                        help="Save the output of the evaluation")
    
    # Add boolean flags for generate_gold_output
    parser.add_argument("--generate_gold_output", dest="generate_gold_output", action="store_true",
                        help="Generate gold output")
    
    #Set max workers
    parser.add_argument("--max_workers", type=int, default=4,
                        help="Maximum number of workers for evaluation (default: 4)")
    
    args = parser.parse_args()
    
    # Determine problem directory based on arguments.
    problem_dir = os.path.join(
        args.problem_folder,
        args.competition,
        args.year,
        args.round,
        args.task,
    )
    
    # Create a Problem instance using the problem_dir.
    problem = Problem(problem_dir, args.task, args.year, args.competition, args.round, "contest")
    
    # Create a Judge with the given evaluation folder.
    judge = Judge(args.evaluation_folder)
    
    # Evaluate the provided solution file using the specified flags.
    score_info, _ = judge.judge(
        problem, 
        args.solution_file, 
        verbose=args.verbose, 
        save_output=args.save_output, 
        generate_gold_output=args.generate_gold_output,
        max_workers=args.max_workers,
        stop_on_failure=args.stop_on_failure
    )
          
    print(score_info)

if __name__ == "__main__":
    main()
# Example usage:
# python src/evaluation/run_judge.py --competition IOI --year 2024 --task nile \
#   --solution_file /path/to/solution.cpp \
#   --problem_folder ${IOI_BENCH_DATA_DIR} \
#   --evaluation_folder ${IOI_EVAL_RESOURCE_DIR}
