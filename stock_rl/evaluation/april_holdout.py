"""Replay frozen no-institutional and flow-only policies on April 2024 sessions."""
import argparse
import hashlib
import json
import tempfile
from pathlib import Path
from urllib.request import urlopen

import pandas as pd
import torch

from .market_drl_pilot import (DATA_DIR, ROOT, FeatureNormalizer, HistoricalSnapshotBuilder,
                               STOCK_METADATA, StockRLFullModel, evaluate, load_real_market_feeds)
from .paired_performance import compare


def prepare_files(cache, dest):
    manifest = json.loads((ROOT / "docs/evidence/april_market_sources.json").read_text())
    macro = {}
    for name, source in manifest.items():
        file = cache / name
        if not file.exists():
            with urlopen(source["url"], timeout=40) as response:
                raw = response.read()
            if hashlib.sha256(raw).hexdigest() != source["sha256"]:
                raise ValueError(f"revised source: {name}")
            file.write_bytes(raw)
        if hashlib.sha256(file.read_bytes()).hexdigest() != source["sha256"]:
            raise ValueError(f"cache mismatch: {name}")
        if file.suffix == ".csv":
            series = pd.read_csv(file, index_col=0).iloc[:, 0]
            macro[source["column"]] = pd.to_numeric(series, errors="raise")
        else:
            new = pd.DataFrame(json.loads(file.read_bytes())["data"])
            old = pd.read_csv(Path(DATA_DIR) / (file.stem + ".csv"))
            if not new.date.between("2024-04-01", "2024-04-30").all():
                raise ValueError(f"unexpected source date: {name}")
            pd.concat([old, new]).to_csv(dest / (file.stem + ".csv"), index=False)
    old = pd.read_csv(Path(DATA_DIR) / "us_macro.csv", index_col=0)
    new = pd.DataFrame(macro)[old.columns].sort_index()
    if new.isna().any().any() or new.index[-1] != "2024-04-30":
        raise ValueError("incomplete April US macro panel")
    pd.concat([old.loc[:"2024-03-29"], new]).to_csv(dest / "us_macro.csv")


def run(cache, control_dir, flow_dir):
    torch.set_num_threads(1)
    cache.mkdir(parents=True, exist_ok=True)
    stocks = {}
    with tempfile.TemporaryDirectory() as tmp:
        prepare_files(cache, Path(tmp))
        for stock, meta in STOCK_METADATA.items():
            prices, inst, macro, foreign, context, _ = load_real_market_feeds(stock, tmp, mode="estimated")
            builder = HistoricalSnapshotBuilder(prices, inst, macro, foreign, context,
                                                event_records=[], shares_outstanding=meta["shares_outstanding"],
                                                mode="estimated")
            april = prices.loc["2024-04"]
            dates = april.index.normalize() + pd.Timedelta(hours=20)
            stocks[stock] = (dates, [builder.build_snapshot(d) for d in dates],
                             april[["open", "high", "low", "close"]].to_numpy())
    results = []
    for seed in (42, 100, 2024):
        arms = {}
        hashes = {}
        for label, directory, omitted in (("without_institutional", control_dir, slice(None)),
                                          ("flow_only", flow_dir, slice(5, 8))):
            path = directory / f"{label}-seed-{seed}.pt"
            state = torch.load(path, map_location="cpu", weights_only=True)
            if (state["schema_version"] != "v8.0-original" or state["seed"] != seed
                    or (label == "flow_only" and state.get("institutional_only") != "flow")
                    or (label == "without_institutional" and state.get("omitted_features") != "I01-I08")):
                raise ValueError("unexpected checkpoint")
            hashes[label] = hashlib.sha256(path.read_bytes()).hexdigest()
            norm = FeatureNormalizer(schema_version=state["schema_version"])
            norm.load_state_dict(state["normalizer"])
            model = StockRLFullModel()
            model.load_state_dict(state["model_state_dict"])
            model.eval()
            def action(obs, index):
                with torch.inference_mode():
                    return int(model(obs)[0][0].argmax())
            arms[label] = {}
            for stock, (dates, snapshots, ohlc) in stocks.items():
                batches = []
                for snapshot in snapshots:
                    batch = snapshot.clone()
                    batch.institutional[..., omitted].zero_()
                    batch.institutional_missing_mask[..., omitted].fill_(True)
                    batches.append(norm.transform(batch))
                arms[label][stock] = evaluate((dates, batches, ohlc, None), action)
        results.append(dict(seed=seed, checkpoint_sha256=hashes, arms=arms,
                            comparison=compare(arms["without_institutional"], arms["flow_only"])))
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cache", "control_dir", "flow_dir"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.cache, args.control_dir, args.flow_dir), indent=2, allow_nan=False))
