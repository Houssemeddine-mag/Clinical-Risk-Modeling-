"""Step 1.1 (Option B) - build the 920-row UCI Heart Disease CSV. No login needed.

The 4 raw files are the official combined dataset (Cleveland 303 + Hungarian 294
+ Switzerland 123 + Long Beach VA 200 = 920 rows). Target `num` is binarized in
notebook 01 (num > 0 -> 1).

Run from anywhere:  python scripts/download_data.py
"""
from pathlib import Path
import io
import urllib.request

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]  # .../Clinical risk
RAW = ROOT / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)

BASE = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease"
FILES = [
    "processed.cleveland.data",
    "processed.hungarian.data",
    "processed.switzerland.data",
    "processed.va.data",
]
COLS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
        "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"]
DEST = RAW / "heart_disease.csv"

if DEST.exists():
    print(f"Already downloaded: {DEST}")
else:
    frames = []
    for name in FILES:
        print("Downloading", name)
        txt = urllib.request.urlopen(f"{BASE}/{name}", timeout=60).read().decode()
        rows = [r for r in txt.splitlines() if r.strip()]
        frames.append(pd.read_csv(io.StringIO("\n".join(rows)), names=COLS, na_values="?"))
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(DEST, index=False)
    print(f"Saved {len(df)} rows -> {DEST}")
