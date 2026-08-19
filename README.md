# Probe Detector

A deep learning system that detects the bounding box of an ultrasonic thickness-measurement probe in drone-captured images or confidently reports that no probe is present.

## Setup

**1. Clone the repository**

```
git clone https://github.com/leogym2/Probe_detector.git
cd Probe_detector
```

**2. Create a virtual environment (Python 3.12 required)**

```
py -3.12 -m venv .venv
```

**3. Activate it**

Windows:
```
.venv\Scripts\activate
```

macOS / Linux:
```
source .venv/bin/activate
```

**4. Install dependencies**

```
pip install -r requirements.txt
```

## Inference

```
python inference.py <folder_of_images> --weights weights/best.pt
```

You can use the folder "examples" to try it out.

Useful options:

- `--conf 0.22` — confidence threshold (default 0.22)
- `--out outputs/` — output folder (default `outputs/`)
- `--show` — display results in a window instead of saving them

For each image, the detected bounding box is drawn (or "No probe detected" if none is found), and results are saved to `outputs/` along with a `detections.json`.
