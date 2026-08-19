# Probe Detector

Deep learning system that detects the bounding box of an ultrasonic
thickness-measurement probe in images captured by an Elios3 drone, or
reports that no probe is present. Built for the "Probe Detection with Deep
Learning" candidate assignment.

See [`report/dataset_analysis.tex`](report/dataset_analysis.tex) for the
full exploratory data analysis, model-family selection rationale, and
negative-example strategy behind the design choices below.

## Project layout

```
probe_dataset/          Raw assignment data (images + probe_labels.json)
probe_common/           Shared helpers (filename parsing, COCO I/O, bbox math)
data_prep/               Dataset-build pipeline (group split, negatives, YOLO conversion)
yolo_dataset/            GENERATED -- Ultralytics-ready dataset (gitignored, regenerable)
training/                train.py (Ultralytics wrapper), train_colab.ipynb (training only),
                         analyze_results.ipynb (evaluation/comparison only, see below)
weights/                 Trained checkpoints (best.pt tracked once training is done)
report/                  EDA + final report + generated figures/results
inference.py              Assignment-mandated inference script (repo root)
evaluate.py               Evaluation protocol (repo root)
compare_results.py        Baseline-vs-fine-tuned / variant comparison charts
analyze_results.py        Evaluates+compares already-trained checkpoints, picks the winner
```

## Setup

Two separate virtual environments are used, both **Python 3.12** (chosen
over the system's newer Python to avoid missing-wheel risk for
`ultralytics`/`onnx`-family packages):

**Main environment** -- training, inference, evaluation:
```
py -3.12 -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

**Data-prep environment** -- only needed to (re)generate `yolo_dataset/`,
since it depends on `simple-lama-inpainting` for the synthetic "no probe"
negatives, which pulls in older numpy/opencv/pillow versions that conflict
with the pins above. Kept isolated so the main environment stays clean and
reproducible:
```
py -3.12 -m venv .venv-lama
.venv-lama\Scripts\pip install -r requirements-dataprep-lama.txt -r requirements.txt
```

On Google Colab, both sets of dependencies are installed together in the
same (ephemeral) runtime -- see `training/train_colab.ipynb` -- since the
conflict only matters for a persistent local dev environment.

## Pipeline

### 1. Build the dataset

```
.venv-lama\Scripts\python -m data_prep.build_yolo_dataset --clean
```

Produces `yolo_dataset/` from `probe_dataset/`: a group-stratified
train/val/test split (by device + flight sequence, to avoid leaking
near-duplicate frames across splits -- see the report), plus a synthetic
"no probe" negative for every image via LaMa generative inpainting (the
dataset otherwise has zero negative examples). Fully deterministic given
`--seed`; nothing under `yolo_dataset/` is meant to be hand-edited.

Optional QA before trusting the negatives at scale:
```
.venv-lama\Scripts\python -m data_prep.inspect_negatives
```
writes a before/after contact sheet to `data_prep/negatives_qa/`.

**Optional utilities** (both only need the main `.venv`, not `.venv-lama` --
neither re-runs LaMa):
- `python -m data_prep.rescore_negatives`: recomputes the quality-review
  score (`score_negative_quality`, see the report) for already-generated
  negatives, e.g. after tweaking the scoring formula, without regenerating
  the actual fill images.
- `python -m data_prep.select_negatives --target-ratio 0.3 --splits train`:
  prunes an already-built dataset's negatives down to a target ratio by
  deleting a random subset of files/manifest entries, without recalling
  LaMa -- useful for quickly trying a different `--train-neg-ratio` on data
  you already generated instead of rebuilding from scratch.

### 2. Train (Google Colab) -- training only

Local hardware is CPU-only, so training runs on Colab's free GPU. Open
`training/train_colab.ipynb`: mount Drive, build the dataset, and train
every candidate checkpoint (YOLOv8 n/s/m). Each checkpoint and the built
dataset are synced to Drive as they're produced. **This notebook does not
evaluate or pick a winner.** (A horizontal-flip augmentation ablation was
considered but dropped for time once the n/s/m comparison alone proved
decisive -- see the report's future-work section.)

To run a single training job locally (e.g. a fast CPU smoke test with few
epochs) instead of the full Colab sweep:
```
.venv\Scripts\python -m training.train --model yolov8n.pt --epochs 5 --device cpu
```

### 3. Analyze results -- evaluation only

Deliberately a separate notebook/script from training, so evaluation code
can be tweaked and re-run cheaply without retraining. Needs only the
`weights/` and `yolo_dataset/` folders (both produced by step 2) -- no GPU,
no LaMa.

On Colab: open `training/analyze_results.ipynb` (a CPU-only runtime is
enough). Locally, once `weights/` and `yolo_dataset/` are downloaded from
Drive:
```
.venv\Scripts\python analyze_results.py --candidates yolov8n yolov8s yolov8m
```
Evaluates the baseline + every candidate on val, writes
`report/figures/model_comparison.png`, auto-picks the winner by AP@0.5,
copies it to `weights/best.pt`, and runs the one-time held-out test-split
check.

### 4. Run inference

```
.venv\Scripts\python inference.py <folder_of_images> [--weights weights/best.pt] [--conf 0.22] [--out outputs/] [--show]
```
Draws the detected probe bounding box on each image, or explicitly marks
"No probe detected" if no detection reaches the confidence threshold.
Results are saved to `outputs/` by default (or shown in a window with
`--show`), along with a `detections.json` sidecar.

### 5. Evaluate a single checkpoint manually

```
.venv\Scripts\python evaluate.py --weights weights/best.pt --split val
```
Reports localization quality (mean IoU, Precision/Recall/F1/AP@0.5, ROC-AUC),
no-probe behavior (false-positive rate vs. confidence threshold, with a
suggested operating threshold and the recall at it), and CPU runtime. Use
`--split val` during development and `--split test` only once, at the end,
for the final reported numbers (see the report for why `test` is small and
noisy here). Normally invoked by `analyze_results.py` (step 3) rather than
by hand.

### 6. Compare models manually

```
.venv\Scripts\python compare_results.py --result baseline=report/eval_results_baseline.json --result yolov8s=report/eval_results_yolov8s.json --out report/figures/model_comparison.png
```
Bar charts comparing AP@0.5, F1, recall at the FP-rate-calibrated threshold,
and runtime across any set of `evaluate.py` result files -- used both for the
baseline-vs-fine-tuned comparison and the n/s/m variant comparison.
