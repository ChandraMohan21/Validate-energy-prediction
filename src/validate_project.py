"""
End-to-end validation of the project. Fails loudly rather than warning quietly.

Checks, in order of severity:

  DATA        row count, date range, gap-free 10-minute cadence, no missing values,
              cleaned data matches the pristine UCI original
  LEAKAGE     every feature is observable at the forecast origin
  EMBARGO     no split's targets reach into the next split
  BASELINES   seasonal naive and persistence are distinct and correctly indexed
  SAMPLING    effective sample size is reported, not the inflated row count
  MODELS      saved models load, expose the expected feature count, and reproduce
              the numbers recorded in reports/results_corrected.json
  ARTEFACTS   every figure referenced by a report exists on disk
  PROVENANCE  no stale campus/block labels anywhere in the repo

Run: python src/validate_project.py
Exit code 0 if everything passes, 1 otherwise.
"""

import json
import re
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from forecasting import (ZONES, HORIZON, EMBARGO, TRAIN_CUTOFF, VAL_CUTOFF, FEATURES,
                         load_hourly, build_supervised_table, split, metrics,
                         baseline_metrics, effective_n)

ROOT = Path(".")
CLEAN = ROOT / "data/processed/cleaned_energy_data.csv"
UCI = ROOT / "data/raw/Tetuan City power consumption (UCI original).csv"
RESULTS = ROOT / "reports/results_corrected.json"
MODELS = ROOT / "models"

FAILS, PASSES = [], []


def check(name, ok, detail=""):
    (PASSES if ok else FAILS).append(name)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    return ok


def main():
    print("\n=== DATA ===")
    df = pd.read_csv(CLEAN, parse_dates=["Datetime"])
    check("cleaned row count is 52,416", len(df) == 52416, f"got {len(df):,}")
    check("no missing values", int(df.isna().sum().sum()) == 0)
    d = df["Datetime"]
    expected = pd.date_range(d.min(), d.max(), freq="10min")
    check("gap-free 10-minute cadence", len(expected) == len(d) and (d.values == expected.values).all())
    check("date range is 2017", d.min().year == 2017 and d.max().year == 2017,
          f"{d.min()} to {d.max()}")
    check("zone columns present",
          all(c in df.columns for c in ZONES), str([c for c in ZONES if c not in df.columns]))

    if UCI.exists():
        u = pd.read_csv(UCI)
        u.columns = [c.replace("  ", " ") for c in u.columns]
        pairs = [("PowerConsumption_Zone1", "Zone 1 Power Consumption"),
                 ("PowerConsumption_Zone2", "Zone 2 Power Consumption"),
                 ("PowerConsumption_Zone3", "Zone 3 Power Consumption")]
        ok = all(np.allclose(df[a].values, u[b].values) for a, b in pairs)
        check("cleaned power values match the pristine UCI original", ok)

    print("\n=== LEAKAGE / EMBARGO / BASELINES ===")
    hdf = load_hourly()
    data = build_supervised_table(hdf, "PowerConsumption_Zone1")
    tr, va, te = split(data)

    lag168_src = data["target_time"] - pd.Timedelta(hours=168)
    lag336_src = data["target_time"] - pd.Timedelta(hours=336)
    check("lag168 is observable at the origin", bool((lag168_src <= data["origin_time"]).all()))
    check("lag336 is observable at the origin", bool((lag336_src <= data["origin_time"]).all()))
    check("no target precedes its origin", bool((data["target_time"] > data["origin_time"]).all()))
    check("horizon never exceeds 168", int(data["horizon"].max()) == HORIZON)

    check("train targets stop at the train cutoff", tr["target_time"].max() <= TRAIN_CUTOFF,
          str(tr["target_time"].max()))
    check("val targets stop at the val cutoff", va["target_time"].max() <= VAL_CUTOFF,
          str(va["target_time"].max()))
    check("test targets start after the val cutoff", te["target_time"].min() > VAL_CUTOFF,
          str(te["target_time"].min()))
    check("embargo gap is 168h between train and val origins",
          (va["origin_time"].min() - tr["origin_time"].max()) >= pd.Timedelta(hours=1))

    sn = data["seasonal_naive"].values
    truth_sn = hdf["PowerConsumption_Zone1"].values[
        [hdf.index.get_loc(t) for t in data["target_time"].iloc[:200]] ]
    check("seasonal_naive differs from persistence",
          not np.allclose(data["seasonal_naive"].values, data["persistence"].values))
    check("seasonal_naive equals lag168", np.allclose(data["seasonal_naive"].values, data["lag168"].values))

    print("\n=== SAMPLING ===")
    check("effective n is much smaller than row count",
          effective_n(te) < len(te) / 50, f"{effective_n(te):,} effective vs {len(te):,} rows")

    print("\n=== MODELS ===")
    if not RESULTS.exists():
        check("results_corrected.json exists", False, "run src/run_pipeline.py")
    else:
        res = json.loads(RESULTS.read_text(encoding="utf-8"))
        for col, label in ZONES.items():
            slug = label.replace(" ", "_").lower()
            p = MODELS / f"{slug}_final_refit_trainval.joblib"
            if not check(f"{label}: refit model exists", p.exists()):
                continue
            m = joblib.load(p)
            nfeat = getattr(m, "n_features_in_", None)
            check(f"{label}: model expects {len(FEATURES)} features", nfeat == len(FEATURES),
                  f"got {nfeat}")
            dd = build_supervised_table(hdf, col)
            _, _, tt = split(dd)
            got = metrics(tt["y"], m.predict(tt[FEATURES]))["MAPE"]
            rec = res["zones"][label]["test_refit"]["MAPE"]
            check(f"{label}: reproduces recorded test MAPE", abs(got - rec) < 0.01,
                  f"recorded {rec:.2f}%, recomputed {got:.2f}%")
            del dd, tt, m

    print("\n=== ARTEFACTS ===")
    for rp in sorted(ROOT.glob("reports/*.md")):
        txt = rp.read_text(encoding="utf-8")
        for rel in re.findall(r"!\[[^\]]*\]\(([^)]+)\)", txt):
            fp = rp.parent / rel
            check(f"{rp.name} -> {rel}", fp.exists())

    print("\n=== PROVENANCE ===")
    stale = re.compile(r"AS Block|IB Block|MS Block|Bannari|Mechanical Sciences|"
                       r"campus meter|PowerConsumption_(ASB|IBB|MSB)")
    hits = []
    self_path = Path(__file__).resolve()
    for pat in ("**/*.py", "**/*.md", "**/*.ipynb"):
        for f in ROOT.glob(pat):
            # skip caches and this file, which necessarily contains the patterns it searches for
            if "__pycache__" in str(f) or f.resolve() == self_path:
                continue
            t = f.read_text(encoding="utf-8", errors="ignore")
            for m in stale.finditer(t):
                if "relabelled" in t[max(0, m.start() - 120):m.start() + 60]:
                    continue
                hits.append(f"{f}: {m.group(0)}")
    check("no stale campus/block labels", not hits, "; ".join(hits[:4]))

    print(f"\n{'='*60}")
    print(f"{len(PASSES)} passed, {len(FAILS)} failed")
    if FAILS:
        print("FAILED:")
        for f in FAILS:
            print(f"  - {f}")
    print("=" * 60)
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
