# LiveOIBench-Evaluation 

![PDF](https://img.shields.io/badge/PDF-Preprint-red?logo=arxiv)](https://arxiv.org/abs/2510.09595)
[![Dataset](https://img.shields.io/badge/Dataset-HuggingFace-informational?logo=huggingface)](https://huggingface.co/datasets/LiveOIBench/LiveOIBench)
[![Leaderboard](https://img.shields.io/badge/Dataset-HuggingFace-informational.svg?logo=data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0idXRmLTgiPz48IS0tIFVwbG9hZGVkIHRvOiBTVkcgUmVwbywgd3d3LnN2Z3JlcG8uY29tLCBHZW5lcmF0b3I6IFNWRyBSZXBvIE1peGVyIFRvb2xzIC0tPg0KPHN2ZyBmaWxsPSIjMDAwMDAwIiB3aWR0aD0iODAwcHgiIGhlaWdodD0iODAwcHgiIHZpZXdCb3g9IjAgMCAyNCAyNCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cGF0aCBkPSJNMjIsN0gxNi4zMzNWNGExLDEsMCwwLDAtMS0xSDguNjY3YTEsMSwwLDAsMC0xLDF2N0gyYTEsMSwwLDAsMC0xLDF2OGExLDEsMCwwLDAsMSwxSDIyYTEsMSwwLDAsMCwxLTFWOEExLDEsMCwwLDAsMjIsN1pNNy42NjcsMTlIM1YxM0g3LjY2N1ptNi42NjYsMEg5LjY2N1Y1aDQuNjY2Wk0yMSwxOUgxNi4zMzNWOUgyMVoiLz48L3N2Zz4=)](https://liveoibench.github.io/leaderboard.html)

LiveOIBnehch consists of 403 coding problems collected directly from the official websites of 72 competitions across 14 renowned Informatics Olympiads, focusing on contests held from 2023 onward. We collect all the offical test cases, human contestant ranking results, and contestant Codeforces profiles.

This Github Repo contains the evaluation toolkit for testing LLMs' solutions against the test cases and compare their performance against human contesants. 


## Installation

### Prerequisites

- Python 3.9+
- `g++` compiler (for compiling C++ solutions and checkers)
- (Optional) `vllm` for serving local models

### Setup

1. Clone the repository:
```bash
git clone https://github.com/your-org/LiveOIBench-Evaluation.git
cd LiveOIBench-Evaluation
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure environment variables:
```bash
source scripts/set_liveoibench_env.sh
```

## Data Setup

### Download from HuggingFace

The benchmark data is available on HuggingFace:

- **Problems & Metadata**: [LiveOIBench/LiveOIBench](https://huggingface.co/datasets/LiveOIBench/LiveOIBench)
- **Test Cases**: [LiveOIBench/LiveOIBench_tests](https://huggingface.co/datasets/LiveOIBench/LiveOIBench_tests)
- **Human Contestant Data**: [LiveOIBench/LiveOIBench_contestants](https://huggingface.co/datasets/LiveOIBench/LiveOIBench_contestants)

### Reconstruct Dataset

Use the provided script to download and reconstruct the dataset:

```bash
python src/process_dataset.py \
  --output-dir ./data \
  --stage all
```

## Quick Start

### Judge a Single Solution

```bash
python src/run_judge.py \
  --competition IOI \
  --year 2024 \
  --round contest \
  --task nile \
  --solution_file ./solutions/my_solution.cpp \
  --problem_folder ./data \
  --evaluation_folder ./evaluation \
  --verbose
```

### Generate Solutions with LLM

```bash
python src/run_ioi.py \
  --model gpt-4o \
  --competitions IOI \
  --years 2024 \
  --problems_dir ./data \
  --prediction_dir ./predictions \
  --seeds 8
```

### Generate Rankings

```bash
python src/generate_rankings.py \
  --submission-results-dir ./evaluation/submission_results \
  --problem-results-dir ./evaluation/problem_results \
  --contest-results-dir ./evaluation/contest_results \
  --final-results-file ./evaluation/final_results.csv \
  --stage all
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `LIVEOIBENCH_DATA_DIR` | Problem data directory | `./data` |
| `LIVEOIBENCH_EVAL_RESOURCE_DIR` | Evaluation resource root | `./` |
| `LIVEOIBENCH_PREDICTIONS_DIR` | LLM predictions storage | `./predictions` |
| `LIVEOIBENCH_SUBMISSION_RESULTS_DIR` | Submission results | `./evaluation/submission_results` |
| `LIVEOIBENCH_PROBLEM_RESULTS_DIR` | Problem-level results | `./evaluation/problem_results` |
| `LIVEOIBENCH_CONTEST_RESULTS_DIR` | Contest-level results | `./evaluation/contest_results` |
| `LIVEOIBENCH_FINAL_RESULTS` | Final rankings CSV | `./evaluation/final_results.csv` |
| `LIVEOIBENCH_CONTESTANT_PARQUET` | Human contestant data | `./data/contest_results.parquet` |
| `LIVEOIBENCH_PROBLEMS_PARQUET` | Problems metadata | `./data/liveoibench_v1.parquet` |
| `TESTLIB_PATH` | Path to testlib.h | `./evaluation/testlib.h` |

You can set these manually or use the provided script:

```bash
# Use default paths (relative to project root)
source scripts/set_liveoibench_env.sh

# Or override specific paths
export LIVEOIBENCH_DATA_DIR="/path/to/your/data"
source scripts/set_liveoibench_env.sh
```
## Submission

To submit your result, please share your submission result file with Kai[zkjzou@umich.edu].

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Citation

If you use LiveOIBench in your research, please cite:

```bibtex
@article{zou2025liveoibench,
  title={LiveOIBench: Can Large Language Models Outperform Human Contestants in Informatics Olympiads?},
  author={Zou, Kaijian and Xiong, Aaron and Zhang, Yunxiang and Zhang, Frederick and Ren, Yueqi and Yang, Jirong and Lee, Ayoung and Bhushan, Shitanshu and Wang, Lu},
  journal={arXiv preprint arXiv:2510.09595},
  year={2025},
  url={https://arxiv.org/abs/2510.09595},
  doi={10.48550/arXiv.2510.09595}
}
```

## Acknowledgments

- [testlib.h](https://github.com/MikeMirzayanov/testlib) for checker support
- The competitive programming community for problem data
