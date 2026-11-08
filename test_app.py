"""Smoke check: app renders, and every saved model reproduces model/metrics.csv on test_data.csv."""
from pathlib import Path

import joblib
import pandas as pd
from streamlit.testing.v1 import AppTest

from train_models import TARGET, evaluate

ROOT = Path(__file__).parent
MODEL_DIR = ROOT / "model"

at = AppTest.from_file(str(ROOT / "app.py")).run(timeout=60)
assert not at.exception, at.exception

data = pd.read_csv(ROOT / "test_data.csv")
saved = pd.read_csv(MODEL_DIR / "metrics.csv").set_index("ML Model Name")
for name in saved.index:
    model = joblib.load(MODEL_DIR / f"{name.lower().replace(' ', '_')}.joblib")
    got = evaluate(model, data[list(model.feature_names_in_)], data[TARGET])
    for k, v in got.items():
        assert abs(round(v, 4) - saved.loc[name, k]) < 1e-9, (name, k, v)
print("OK: app renders, all 5 models match metrics.csv")
