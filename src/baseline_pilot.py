"""Origin-frozen daily/weekly/hybrid/blended reference forecasts.

Run from any directory: python src/baseline_pilot.py --download
Subsequent runs use cached inputs. No fitted ML models or foundation models run here.
"""

import argparse
import hashlib
import io
import json
import platform
import sys
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/baseline_pilot"
RAW = ROOT / "data/raw/baseline_pilot"
PROCESSED = ROOT / "data/processed/baseline_pilot"
PROTOCOL = OUT / "protocol.json"


def file_hash(path, algorithm="sha256"):
    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_sources(protocol):
    RAW.mkdir(parents=True, exist_ok=True)
    for name, source in protocol["sources"].items():
        path = RAW / source["filename"]
        if not path.exists():
            print(f"Downloading {name}: {source['page']}", flush=True)
            request = urllib.request.Request(source["url"], headers={"User-Agent": "ElectricityBaselinePilot/1.0"})
            part = path.with_suffix(".part")
            with urllib.request.urlopen(request, timeout=60) as response, part.open("wb") as target:
                while block := response.read(1024 * 1024):
                    target.write(block)
            if not zipfile.is_zipfile(part):
                raise ValueError(f"Downloaded file is not a ZIP: {part}")
            if source.get("md5") and file_hash(part, "md5") != source["md5"]:
                raise ValueError(f"Publisher checksum mismatch for {name}")
            part.replace(path)
        if source.get("md5") and file_hash(path, "md5") != source["md5"]:
            raise ValueError(f"Cached checksum mismatch for {name}")
        print(f"Ready {name}: {path.stat().st_size:,} bytes", flush=True)


def tsf_records(path, limit):
    """Read published TSF attributes instead of assuming a field position."""
    with zipfile.ZipFile(path) as archive:
        names = [n for n in archive.namelist() if n.endswith(".tsf")]
        if len(names) != 1:
            raise ValueError(f"Expected one TSF file: {names}")
        with archive.open(names[0]) as raw:
            stream = io.TextIOWrapper(raw, encoding="cp1252")
            attributes, in_data, frequency = [], False, None
            count = 0
            for line in stream:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("@attribute"):
                    attributes.append(line.split()[1])
                elif line.startswith("@frequency"):
                    frequency = line.split()[1]
                elif line.lower() == "@data":
                    in_data = True
                elif in_data:
                    parts = line.split(":")
                    if len(parts) != len(attributes) + 1:
                        raise ValueError("Unexpected TSF fields")
                    meta = dict(zip(attributes, parts[:-1]))
                    values = np.array([float(v) if v != "?" else np.nan for v in parts[-1].split(",")])
                    start = pd.to_datetime(meta["start_timestamp"], format="%Y-%m-%d %H-%M-%S")
                    freq = {"hourly": "h", "half_hourly": "30min"}[frequency]
                    # These Monash archives start at xx:00:01. Preserve interval
                    # order but align the one-second offset to the native grid.
                    aligned = start.floor(freq)
                    if start - aligned not in (pd.Timedelta(0), pd.Timedelta(seconds=1)):
                        raise ValueError(f"Unexpected archive timestamp offset: {start}")
                    start = aligned
                    series = pd.Series(values, index=pd.date_range(start, periods=len(values), freq=freq))
                    yield meta, series, frequency
                    count += 1
                    if count >= limit:
                        return
            if count < limit:
                raise ValueError(f"Only {count} TSF records, expected {limit}")


def complete_hourly(series, expected_count):
    bins = series.resample("h")
    return bins.mean().where(bins.count() == expected_count)


def prepare_series(protocol):
    """Deterministic sample and years fixed in protocol before scoring."""
    from forecasting import ZONES, load_hourly
    PROCESSED.mkdir(parents=True, exist_ok=True)
    entries = []
    tetouan = load_hourly(ROOT / "data/processed/cleaned_energy_data.csv")
    for col, label in ZONES.items():
        entries.append(("tetouan", label.lower().replace(" ", "_"), tetouan[col], "kW"))
    for meta, series, _ in tsf_records(RAW / protocol["sources"]["portugal"]["filename"], 3):
        name = next((str(v) for k, v in meta.items() if k != "start_timestamp"), "series")
        entries.append(("portugal", name, series, protocol["sources"]["portugal"]["unit"]))
    for meta, series, freq in tsf_records(RAW / protocol["sources"]["victoria"]["filename"], 1):
        if freq != "half_hourly":
            raise ValueError("Victoria archive frequency changed")
        entries.append(("victoria", "regional_demand", complete_hourly(series, 2), "GW"))

    household_year = protocol["years"]["household"]
    household_cache = PROCESSED / f"household_sceaux_household_{household_year}.csv"
    household_cache_meta = PROCESSED / "household_preparation.json"
    cache_key = {"raw_sha256": file_hash(RAW / protocol["sources"]["household"]["filename"]),
                 "year": household_year, "minimum_samples_per_hour": 60}
    cache_valid = (household_cache_meta.exists()
                   and json.loads(household_cache_meta.read_text()) == cache_key)
    if household_cache.exists() and cache_valid:
        household = pd.read_csv(household_cache, index_col="Datetime", parse_dates=True)["load"]
    else:
        with zipfile.ZipFile(RAW / protocol["sources"]["household"]["filename"]) as archive:
            member = next(n for n in archive.namelist() if n.endswith("household_power_consumption.txt"))
            with archive.open(member) as stream:
                frame = pd.read_csv(stream, sep=";", usecols=["Date", "Time", "Global_active_power"],
                                    na_values=["?"], dtype={"Global_active_power": "float64"})
        timestamps = pd.to_datetime(frame["Date"] + " " + frame["Time"], format="%d/%m/%Y %H:%M:%S")
        minute = pd.Series(frame["Global_active_power"].to_numpy(), index=timestamps)
        if not minute.index.is_unique:
            raise ValueError("Household duplicate timestamps")
        year = protocol["years"]["household"]
        minute = minute.loc[f"{year}-01-01":f"{year}-12-31 23:59:59"]
        household = complete_hourly(minute, 60)
    entries.append(("household", "sceaux_household", household, "kW"))

    prepared = []
    for dataset, name, series, unit in entries:
        year = protocol["years"][dataset]
        index = pd.date_range(f"{year}-01-01", f"{year}-12-31 23:00", freq="h")
        # Tetouan genuinely ends on Dec 30; do not invent a missing final day.
        if dataset == "tetouan":
            index = index[index <= series.index.max()]
        series = series.reindex(index).astype(float)
        if not series.index.is_unique or (series.dropna() < 0).any():
            raise ValueError(f"Invalid nonnegative load series: {dataset}/{name}")
        path = PROCESSED / f"{dataset}_{name}_{year}.csv"
        series.rename("load").to_csv(path, index_label="Datetime")
        prepared.append({"dataset": dataset, "series": name, "unit": unit,
                         "data": series, "path": path})
    household_cache_meta.write_text(json.dumps(cache_key, indent=2), encoding="utf-8")
    return prepared


def origin_indices(series, protocol):
    """Common origins across horizons; no gap-filling or future-based selection."""
    history, horizon = protocol["history_hours"], max(protocol["horizons_hours"])
    origins = np.arange(history, len(series) - horizon)
    missing = np.r_[0, np.cumsum(~np.isfinite(series.to_numpy()))]
    good = ((missing[origins + 1] - missing[origins - history + 1] == 0)
            & (missing[origins + horizon + 1] - missing[origins + 1] == 0))
    eligible = origins[good]
    end = series.index.max()
    val_cutoff = end - pd.Timedelta(days=protocol["test_calendar_days"])
    train_cutoff = val_cutoff - pd.Timedelta(days=protocol["validation_calendar_days"])
    embargo = pd.Timedelta(hours=protocol["embargo_hours"])
    times = series.index[eligible]
    val = eligible[(times > train_cutoff) & (times <= val_cutoff - embargo)]
    test = eligible[times > val_cutoff]
    if not len(val) or not len(test):
        raise ValueError("No complete validation or test origins after applying the prespecified missing-data rule")
    if not series.index[val[-1]] + pd.Timedelta(hours=horizon) <= val_cutoff:
        raise AssertionError("Validation targets cross the test boundary")
    return val, test, {"train_cutoff": str(train_cutoff), "validation_cutoff": str(val_cutoff),
                       "candidate_origins": int(len(origins)), "eligible_origins": int(len(eligible)),
                       "origins_removed_for_gaps": int((~good).sum()),
                       "validation_origins": int(len(val)), "test_origins": int(len(test))}


def reference_predictions(series, origins, horizon):
    """Every returned source index is an observation at or before its origin."""
    values = series.to_numpy()
    origins = np.asarray(origins)
    steps = np.arange(1, horizon + 1)
    targets = origins[:, None] + steps
    daily_sources = origins[:, None] + (steps - 1) % 24 - 23
    weekly_sources = targets - 168 * (((steps - 1) // 168) + 1)
    hybrid_sources = np.empty_like(targets)
    weekdays = series.index.dayofweek.to_numpy()[targets]
    for j, step in enumerate(steps):
        use_weekly = np.isin(weekdays[:, j], [0, 5, 6])
        daily_source = targets[:, j] - 24 if step <= 24 else hybrid_sources[:, j - 24]
        hybrid_sources[:, j] = np.where(use_weekly, weekly_sources[:, j], daily_source)
    for sources in (daily_sources, weekly_sources, hybrid_sources):
        if not np.all((sources >= 0) & (sources <= origins[:, None])):
            raise AssertionError("A baseline reads an unobserved value")
    return {"daily": values[daily_sources], "weekly": values[weekly_sources],
            "hybrid": values[hybrid_sources]}, values[targets], targets


def score(actual, predicted):
    actual = np.asarray(actual, float)
    residual = np.asarray(predicted, float) - actual
    zero = int(np.count_nonzero(actual == 0))
    mape = None if zero else float(np.mean(np.abs(residual / actual)) * 100)
    total = np.abs(actual).sum()
    return {"MAE": float(np.mean(np.abs(residual))), "RMSE": float(np.sqrt(np.mean(residual ** 2))),
            "MAPE_pct": mape, "WAPE_pct": float(np.abs(residual).sum() / total * 100) if total else None,
            "zero_target_rows": zero}


def choose_on_validation(actual, predictions, protocol):
    weights = protocol["blend_daily_weights"]
    losses = [float(np.mean(np.abs(actual - (a * predictions["daily"] + (1 - a) * predictions["weekly"]))))
              for a in weights]
    # First minimum is deterministic; endpoints duplicate daily/weekly, which win ties below.
    alpha = weights[int(np.argmin(losses))]
    candidates = dict(predictions)
    candidates["blend"] = alpha * candidates["daily"] + (1 - alpha) * candidates["weekly"]
    scores = {name: score(actual, candidates[name]) for name in protocol["selection_tie_order"]}
    best_mae = min(v["MAE"] for v in scores.values())
    selected = next(name for name in protocol["selection_tie_order"]
                    if np.isclose(scores[name]["MAE"], best_mae, rtol=1e-12, atol=1e-12))
    return float(alpha), selected, scores


def evaluate(entry, protocol):
    series = entry["data"]
    val, test, split_info = origin_indices(series, protocol)
    records = []
    for horizon in protocol["horizons_hours"]:
        vp, vy, vt = reference_predictions(series, val, horizon)
        alpha, selected, validation = choose_on_validation(vy, vp, protocol)
        tp, ty, tt = reference_predictions(series, test, horizon)
        tp["blend"] = alpha * tp["daily"] + (1 - alpha) * tp["weekly"]
        scores = {name: score(ty, pred) for name, pred in tp.items()}
        weekly_mae = scores["weekly"]["MAE"]
        result = {"dataset": entry["dataset"], "series": entry["series"], "unit": entry["unit"],
                  "horizon_hours": horizon, "blend_daily_weight": alpha, "selected_on_validation": selected,
                  "validation": validation, "test": scores, "split": split_info,
                  "validation_origin_first": str(series.index[val[0]]),
                  "validation_origin_last": str(series.index[val[-1]]),
                  "test_origin_first": str(series.index[test[0]]), "test_origin_last": str(series.index[test[-1]]),
                  "validation_target_last": str(series.index[vt.max()]),
                  "test_target_first": str(series.index[tt.min()]), "test_target_last": str(series.index[tt.max()]),
                  "test_distinct_target_hours": int(np.unique(tt).size),
                  "test_forecast_rows": int(ty.size),
                  "selected_MAE_gain_vs_weekly_pct": float(100 * (1 - scores[selected]["MAE"] / weekly_mae)) if weekly_mae else None,
                  "processed_sha256": file_hash(entry["path"])}
        destination = OUT / "predictions" / f"{entry['dataset']}_{entry['series']}_h{horizon}.npz"
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(destination, hourly_time=series.index.to_numpy(dtype="datetime64[ns]"),
                            validation_origins=val, validation_actual=vy, test_origins=test, actual=ty,
                            daily=tp["daily"], weekly=tp["weekly"], hybrid=tp["hybrid"], blend=tp["blend"],
                            blend_daily_weight=alpha, selected_on_validation=selected)
        result["prediction_file"] = str(destination.relative_to(ROOT)).replace("\\", "/")
        records.append(result)
    return records


def save_report(results, metadata):
    rows = []
    for r in results:
        for name, values in r["test"].items():
            rows.append({"dataset": r["dataset"], "series": r["series"], "horizon_hours": r["horizon_hours"],
                         "method": name, "selected_on_validation": name == r["selected_on_validation"],
                         "blend_daily_weight": r["blend_daily_weight"] if name == "blend" else None,
                         **values, "unit": r["unit"]})
    pd.DataFrame(rows).to_csv(OUT / "scores.csv", index=False)
    (OUT / "results.json").write_text(json.dumps({"metadata": metadata, "results": results}, indent=2, allow_nan=False), encoding="utf-8")
    lines = ["# Seasonal baseline pilot", "", "A baseline-only feasibility experiment. No foundation model or trained model was evaluated in this pilot.", "",
             "## What was done", "", "Forecasts were frozen at each hourly origin. Daily repeats the last observed day; weekly repeats the last observed week. The weekday hybrid uses weekly values for Saturday through Monday and daily recursion otherwise. The blend is alpha * daily + (1-alpha) * weekly, with alpha selected from a fixed grid using validation MAE only. The final rule is also selected on validation MAE, separately for each series and horizon.", "",
             "The 168-hour task is primary; 24 hours is a secondary comparison. Both use identical origin sets and a 168-hour embargo. Validation and test losses count every origin/horizon pair equally. Repeated targets are not independent observations.", "",
             "## Seven-day test results", "", "MAE stays in the series' own units; compare methods within a row, not raw MAE between datasets. Positive gain means the validation-selected baseline beats weekly; negative gain means it loses on test.", "",
             "| Dataset / series | Daily MAE | Weekly MAE | Hybrid MAE | Blend MAE | Validation choice | Daily weight | Gain vs weekly | Test origins |",
             "|---|---:|---:|---:|---:|---|---:|---:|---:|"]
    for r in results:
        if r["horizon_hours"] != 168:
            continue
        s = r["test"]
        gain = r["selected_MAE_gain_vs_weekly_pct"]
        lines.append(f"| {r['dataset']} / {r['series']} | {s['daily']['MAE']:.4f} | {s['weekly']['MAE']:.4f} | {s['hybrid']['MAE']:.4f} | {s['blend']['MAE']:.4f} | {r['selected_on_validation']} | {r['blend_daily_weight']:.2f} | {gain:+.2f}% | {r['split']['test_origins']} |")
    primary = [r for r in results if r["horizon_hours"] == 168]
    gains = [r["selected_MAE_gain_vs_weekly_pct"] for r in primary]
    better = sum(g > 1e-9 for g in gains)
    worse = sum(g < -1e-9 for g in gains)
    tied = len(gains) - better - worse
    lines += ["", f"Descriptively, validation selection beats fixed weekly on {better} series, ties on {tied}, and loses on {worse}. These are correlated series, not independent significance tests.", "",
              "## Tetouan in percentage-error terms", "", "The comparison below uses the same 840 test origins as the existing seven-day study. Selection still uses validation MAE, not test MAPE. A different selection metric may select a different rule and should be declared before a future study.", "",
              "| Zone | Daily MAPE | Weekly MAPE | Hybrid MAPE | Blend MAPE | Validation choice |",
              "|---|---:|---:|---:|---:|---|"]
    for r in primary:
        if r["dataset"] == "tetouan":
            s = r["test"]
            lines.append(f"| {r['series']} | {s['daily']['MAPE_pct']:.2f}% | {s['weekly']['MAPE_pct']:.2f}% | {s['hybrid']['MAPE_pct']:.2f}% | {s['blend']['MAPE_pct']:.2f}% | {r['selected_on_validation']} |")
    lines += ["", "## Sources and preparation", "", "The year and meter-selection rules were fixed in protocol.json before scoring. Portugal uses the first three archive records, not the best-performing meters. Both Monash TSF archives start one second after their native time grid; this constant one-second offset is removed before hourly alignment, without shifting observation order. Victoria half-hourly power is averaged to hourly power; household minute power is averaged only when all 60 samples exist. Missing hours are retained as missing, with affected origins excluded. Tetouan uses the existing cleaned UCI series. Calendar timestamps otherwise follow the source, without claiming UTC alignment. Portugal retains the source's DST conventions and archive scaling.", ""]
    for key, source in metadata["protocol"]["sources"].items():
        lines.append(f"- {key}: [{source['title']}]({source['page']}). {source['license']}. Unit: {source['unit']}.")
    lines += ["", "## What this can and cannot tell us", "", "This pilot measures baseline sensitivity, not foundation-model advantage, statistical significance, or novelty. Only one prespecified year per dataset and a few series were used. Related zones/meters are not independent datasets; do not interpret the eight series as eight independent replications. Validation selection can fail under changing patterns, and all such failures remain reported. These familiar public datasets may overlap foundation-model pretraining; no contamination-free claim is made.", "",
              "The weekday hybrid is an origin-frozen adaptation of established day-ahead price practice, not a newly invented method. The daily/weekly blend is an elementary baseline inspired by reference-combination literature, not Yang's exact climatology/persistence method.", "",
              "MAPE is undefined when a target is zero, so those series/horizons have null MAPE rather than an arbitrary epsilon. scores.csv also includes MAE, RMSE, WAPE and zero counts. results.json contains the full selection and split metadata. predictions/*.npz retain actuals, origins and all four test forecasts for paired follow-up analysis.", "",
              "## What happens next", "", "Use these frozen series and origins to evaluate Chronos in Colab, saving every prediction. Then evaluate clearly labelled trained models on the same data, select their settings before test, and compare paired errors with dependence-aware uncertainty. Recheck the novelty of a specific empirical finding before expanding the paper. None of those model runs was performed by this script.", "",
              "## Reproduce", "", "Run `python src/baseline_pilot.py --download` from the repository. Cached inputs permit `python src/baseline_pilot.py`. Only pandas, numpy and the Python standard library are required. Run `python -m unittest discover -s tests -p test_baseline_pilot.py` for protocol tests.", ""]
    (OUT / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="Download public sources if not cached")
    args = parser.parse_args()
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if args.download:
        download_sources(protocol)
    missing = [s["filename"] for s in protocol["sources"].values() if not (RAW / s["filename"]).exists()]
    if missing:
        raise FileNotFoundError(f"Missing {missing}; run with --download")
    entries = prepare_series(protocol)
    results = []
    for entry in entries:
        print(f"Scoring {entry['dataset']}/{entry['series']}", flush=True)
        results.extend(evaluate(entry, protocol))
    metadata = {"run_time_utc": datetime.now(timezone.utc).isoformat(),
                "protocol": protocol, "protocol_sha256": file_hash(PROTOCOL),
                "python": sys.version, "numpy": np.__version__, "pandas": pd.__version__,
                "platform": platform.platform(),
                "source_files": {key: {"path": str((RAW / source["filename"]).relative_to(ROOT)),
                                       "sha256": file_hash(RAW / source["filename"]),
                                       "bytes": (RAW / source["filename"]).stat().st_size,
                                       "url": source["url"]} for key, source in protocol["sources"].items()},
                "local_source_files": {"tetouan_cleaned_sha256": file_hash(ROOT / "data/processed/cleaned_energy_data.csv"),
                                       "pilot_script_sha256": file_hash(Path(__file__)),
                                       "forecasting_script_sha256": file_hash(ROOT / "src/forecasting.py")}}
    save_report(results, metadata)
    print(f"Saved {len(results)} series/horizon results to {OUT}", flush=True)


if __name__ == "__main__":
    main()
