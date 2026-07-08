# DigitalWordCounts

Scripts for measuring digitalization-related content in annual MD&A text files using Chinese sentence segmentation and BGE embeddings.

Matching uses an OR rule:

- sentence contains a valid term in `digital_keywords.txt` and its best topic
  embedding similarity is at least `--keyword-score-threshold` (default `0.5`), or
- sentence's best topic embedding similarity is at least `--threshold`.

## Environment

Create or sync a Python environment with `uv`:

```powershell
uv venv .venv --python 3.11 --seed
uv pip install --python .\.venv\Scripts\python.exe -r requirements.txt
```

The pipeline expects `BAAI/bge-base-zh-v1.5` and can cache Hugging Face files inside the project:

```powershell
$env:HF_HOME='D:\Stata-Projects\DigitalWordCounts\.hf-cache'
```

If you need a specific CUDA PyTorch build, install PyTorch from the official PyTorch index before or after installing `requirements.txt`, for example:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu126
```

## Annual Pipeline

Run annual-only reports for 2023-2025:

```powershell
cd D:\Stata-Projects\DigitalWordCounts

$env:PYTHONIOENCODING='utf-8'
$env:HF_HOME='D:\Stata-Projects\DigitalWordCounts\.hf-cache'

.\.venv\Scripts\python.exe scripts\run_digital_embedding_pipeline.py `
  --start-year 2023 `
  --end-year 2025 `
  --threshold 0.62 `
  --keyword-score-threshold 0.5 `
  --keyword-path .\digital_keywords.txt `
  --keyword-match-mode strict `
  --keyword-tokenizer jieba `
  --exclude-theme-ids D14 `
  --batch-size 128 `
  --gpu-report-batch-size 16 `
  --max-gpu-sentences 8192 `
  --num-workers 4 `
  --device cuda:0 `
  --resume `
  --save-matches sample `
  --sample-per-report 20 `
  --progress-every 25
```

By default, logging is intentionally compact: startup, periodic progress, exports, and errors. Add `--verbose` to enable per-report and per-GPU-batch debug logs for task discovery, preprocessing, GPU batch flushing, and batch encoding details.

Use `--disable-keyword-match` to run embedding-threshold-only matching.
Use `--exclude-theme-ids D14` to exclude the broad R&D innovation theme if it
pulls in general technology-development sentences that are not digitalization.
Use `--keyword-tokenizer substring` to skip jieba tokenization. The default
keyword mode is strict, so generic standalone terms such as generic "data" or
"information" do not create keyword matches unless a more specific digitalization
term also appears. When `--keyword-tokenizer jieba` is used, terms from
`digital_keywords.txt` are automatically registered in jieba's dictionary.

For fast keyword-only matching without loading the embedding model:

```powershell
.\.venv\Scripts\python.exe scripts\run_digital_embedding_pipeline.py `
  --start-year 2023 `
  --end-year 2025 `
  --keyword-path .\digital_keywords.txt `
  --keyword-match-mode strict `
  --keyword-tokenizer substring `
  --match-rule keyword-only `
  --keyword-score-threshold 0 `
  --output-dir output_keyword_only `
  --num-workers 4 `
  --save-matches sample `
  --sample-per-report 20 `
  --progress-every 25
```

Outputs are written under `output/`:

- `pipeline_state.sqlite`
- `digital_report_level_annual.csv`
- `digital_sentence_matches_annual.csv`

## Progress Check

Check the current SQLite state once:

```powershell
.\.venv\Scripts\python.exe scripts\check_pipeline_progress.py
```

Refresh automatically every 30 seconds:

```powershell
.\.venv\Scripts\python.exe scripts\check_pipeline_progress.py --watch 30
```

On Linux or SSH instances:

```bash
python scripts/check_pipeline_progress.py --watch 30
```

Input data folders, virtual environments, model caches, logs, and outputs are ignored by Git.
