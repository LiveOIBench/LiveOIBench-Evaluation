# LiveOIBench-Evaluation 

![PDF](https://img.shields.io/badge/PDF-Preprint-red?logo=arxiv)](https://arxiv.org/abs/2510.09595)
[![Dataset](https://img.shields.io/badge/Dataset-HuggingFace-informational?logo=huggingface)](https://huggingface.co/datasets/LiveOIBench/LiveOIBench)
[![Leaderboard](https://img.shields.io/badge/Dataset-HuggingFace-informational.svg?logo=data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0idXRmLTgiPz48IS0tIFVwbG9hZGVkIHRvOiBTVkcgUmVwbywgd3d3LnN2Z3JlcG8uY29tLCBHZW5lcmF0b3I6IFNWRyBSZXBvIE1peGVyIFRvb2xzIC0tPg0KPHN2ZyBmaWxsPSIjMDAwMDAwIiB3aWR0aD0iODAwcHgiIGhlaWdodD0iODAwcHgiIHZpZXdCb3g9IjAgMCAyNCAyNCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cGF0aCBkPSJNMjIsN0gxNi4zMzNWNGExLDEsMCwwLDAtMS0xSDguNjY3YTEsMSwwLDAsMC0xLDF2N0gyYTEsMSwwLDAsMC0xLDF2OGExLDEsMCwwLDAsMSwxSDIyYTEsMSwwLDAsMCwxLTFWOEExLDEsMCwwLDAsMjIsN1pNNy42NjcsMTlIM1YxM0g3LjY2N1ptNi42NjYsMEg5LjY2N1Y1aDQuNjY2Wk0yMSwxOUgxNi4zMzNWOUgyMVoiLz48L3N2Zz4=)](https://liveoibench.github.io/leaderboard.html)

LiveOIBnehch consists of 403 coding problems collected directly from the official websites of 72 competitions across 14 renowned Informatics Olympiads, focusing on contests held from 2023 onward. We collect all the offical test cases, human contestant ranking results, and contestant Codeforces profiles.

This GitHub repo contains the evaluation toolkit for testing LLMs' solutions against the test cases and comparing their performance against human contestants. 


## Installation

### Prerequisites

- Python **3.9+**
- `g++` (required for compiling C++ solutions and checkers)
- Linux environment (**strongly recommended**)
- *(Optional)* [`vllm`](https://github.com/vllm-project/vllm) for serving local models

### Setup

```bash
git clone https://github.com/your-org/LiveOIBench-Evaluation.git
cd LiveOIBench-Evaluation
pip install -r requirements.txt
```

> ⚠️ This repo is developed and tested on Linux. macOS may work but is not officially supported.

---

## Data Setup

### Download from HuggingFace

The benchmark is hosted across three HuggingFace datasets:

- **Problems & Metadata**  
  https://huggingface.co/datasets/LiveOIBench/LiveOIBench

- **Official Test Cases**  
  https://huggingface.co/datasets/LiveOIBench/LiveOIBench_tests

- **Human Contestant Data**  
  https://huggingface.co/datasets/LiveOIBench/LiveOIBench_contestants

### Reconstruct the Dataset

```bash
export LIVEOIBENCH_ROOT=<path_to_store_data_and_results>

python src/process_dataset.py \
  --download-dir "${LIVEOIBENCH_ROOT}/parquet_files" \
  --output-dir "${LIVEOIBENCH_ROOT}/data"
```

⏳ **Note:**  
Reconstruction may take significant time and disk space.  
Total test cases exceed **30 GB**.

---

## Quick Start

### 1. Generate Solutions with an LLM

#### Start a local vLLM server

```bash
bash scripts/start_vllm.sh
```

#### Generate model solutions

```bash
bash scripts/run_model.sh
```

Outputs will be saved to:

```
${LIVEOIBENCH_ROOT}/predictions/<model>/
  ├── <model>_code.json
  └── <model>_raw.json
```

---

### 2. Judge Model Solutions

Run official judging against test cases:

```bash
bash scripts/run_model_solutions.sh
```

Results will be saved to:

```
${LIVEOIBENCH_ROOT}/evaluation/submission_results/<model>/<model>_<timestamp>.json
```

---

### 3. Compare Against Human Contestants

Compute rankings across:
- Individual contests
- Olympiad competitions
- Overall benchmark performance

```bash
bash scripts/generate_all_ranking.sh
```

This generates CSV summaries with:
- Contest-level scores
- Relative human percentile
- Aggregate benchmark results

---

## Submitting Results

To submit results to the LiveOIBench leaderboard:

📩 Email your model’s  
`<model>_code.json`  
to **Kai** at **zkjzou@umich.edu**

---

## License

This project is released under the **Apache License 2.0**.  
See the [LICENSE](LICENSE) file for details.

---

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
