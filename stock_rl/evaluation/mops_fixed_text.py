"""Probe headline or timing inputs on the factual trajectory of a frozen policy."""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import torch

from stock_rl.config.schema import get_event_age_index

from . import mops_news_aware_pilot as pilot


@torch.inference_mode()
def paired_replay(model, sample, placebo):
    dates, batches, _, _ = sample
    other_dates, alternatives, _, _ = placebo
    if not dates.equals(other_dates) or len(batches) != len(alternatives):
        raise ValueError("Mismatched decision sessions")
    rows = []
    model.eval()

    def choose(obs, index):
        other = obs.clone()
        replacement = alternatives[index]
        if not torch.equal(obs.event_mask, replacement.event_mask):
            raise ValueError("Mismatched event slots")
        other.recent_event_emb.copy_(replacement.recent_event_emb)
        other.active_event_emb.copy_(replacement.active_event_emb)
        real, constant = (model(item)[0][0] for item in (obs, other))
        if not (real.isfinite().all() and constant.isfinite().all()):
            raise ValueError("Non-finite action values")
        real_action, constant_action = int(real.argmax()), int(constant.argmax())
        rows.append(dict(date=dates[index].date().isoformat(), events=int(obs.event_mask.sum()),
                         real_gap=float(real[1]-real[0]), constant_gap=float(constant[1]-constant[0]),
                         real_action=real_action, constant_action=constant_action))
        return real_action

    factual = pilot.evaluate(sample, choose)
    return rows, factual


def timing_view(obs, label):
    if label == "empty":
        return pilot.mask_event_inputs(obs, "M0")
    if label == "fresh":
        fresh = obs.clone()
        age = get_event_age_index(obs.schema_version)
        fresh.event_metadata[..., age].masked_fill_(obs.event_mask, 0)
        return fresh
    raise ValueError(f"Unknown timing intervention: {label}")


@torch.inference_mode()
def timing_replay(model, sample):
    dates = sample[0]
    rows = []
    model.eval()

    def choose(obs, index):
        # A zero-age intervention treats every visible event as newly published.
        fresh = timing_view(obs, "fresh")
        empty = timing_view(obs, "empty")
        row = {"date": dates[index].date().isoformat(), "events": int(obs.event_mask.sum())}
        for label, view in (("real", obs), ("fresh", fresh), ("empty", empty)):
            q = model(view)[0][0]
            if not q.isfinite().all():
                raise ValueError("Non-finite action values")
            row[label + "_gap"] = float(q[1] - q[0])
            row[label + "_action"] = int(q.argmax())
        rows.append(row)
        return row["real_action"]

    actual = pilot.evaluate(sample, choose)
    return rows, actual


@torch.inference_mode()
def strategy_replays(model, sample, actual):
    """Allow each counterfactual action to drive its own subsequent position."""
    outcomes = {"real": actual}
    for label in ("fresh", "empty"):
        def choose(obs, index):
            q = model(timing_view(obs, label))[0][0]
            if not q.isfinite().all():
                raise ValueError("Non-finite action values")
            return int(q.argmax())
        outcomes[label] = pilot.evaluate(sample, choose)
    return {label: {key: metrics[key] for key in ("net_return", "executed_position_changes", "costs")}
            for label, metrics in outcomes.items()}


def prepare_inputs(cache, april, group):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    protocol = json.loads((pilot.ROOT / "configs/v8_22_study.json").read_text())
    dates = [pd.Timestamp(day, tz=protocol["timezone"]) + pd.Timedelta(hours=protocol["decision_hour"])
             for phase in pilot.PHASES for day in protocol["splits"][phase]]
    digests = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
               for p in sorted(Path(pilot.DATA_DIR).glob("*.csv"))}
    manifest = pilot.SplitManifest.create("mops-news-aware-pilot", pilot.CURRENT_ACTIVE_SCHEMA_VERSION,
                                          *dates, raw_dataset_content_hash=pilot.SplitManifest.compute_content_hash(digests))
    engine = pilot.load_trusted_pretrained_embedding_engine(cache_dir=cache)
    if engine is None:
        raise RuntimeError("Verified BGE weights required")
    ledgers = pilot.build_mops_event_ledgers(pilot.load_mops_disclosures(pilot.MOPS_JSON), engine)
    groups = pilot.build_study_inputs(protocol, manifest, *ledgers)
    factual = groups[group]["inputs"]
    periods = {"test": (factual["test"], groups["D"]["inputs"]["test"])}
    if april:
        april.mkdir(exist_ok=True, parents=True)
        inputs = pilot.build_april_holdout_inputs(protocol, {k: g["normalizer"] for k, g in groups.items()},
                                                  *ledgers, april)
        periods["holdout"] = inputs[group], inputs["D"]
    return protocol, factual, periods


def weights_digest(model):
    digest = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        digest.update(key.encode() + b"\0")
        digest.update(value.detach().contiguous().numpy().tobytes())
    return digest.hexdigest()


def run(cache, april, checkpoints=None, probe="text"):
    group = {"text": "C", "timing": "B"}[probe]
    protocol, factual, periods = prepare_inputs(cache, april, group)
    reference = json.loads(pilot.RESULTS_JSON.read_text())["retrained_models"][group]
    result = []
    for seed in protocol["seeds"]:
        agent, epoch, _ = pilot.fit_seed(seed, protocol, factual)
        model = agent.online_model.cpu().eval()
        digest = weights_digest(model)
        if checkpoints:
            checkpoints.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), checkpoints / f"group-{group.lower()}-seed-{seed}.pt")
        for phase, (real, placebo) in periods.items():
            for stock in protocol["stocks"]:
                if probe == "timing":
                    rows, actual = timing_replay(model, real[stock])
                else:
                    rows, actual = paired_replay(model, real[stock], placebo[stock])
                expected = reference[phase][str(seed)]["per_stock"][stock]
                matches = (actual["target_actions"] == expected["target_actions"]
                           and actual["session_dates"] == expected["session_dates"]
                           and abs(actual["net_return"] - expected["net_return"]) < 1e-8)
                result.append(dict(seed=seed, phase=phase, stock=stock, epoch=epoch,
                                   weights_sha256=digest, reproduces_v847=matches,
                                   decisions=rows))
                if probe == "timing":
                    result[-1]["strategies"] = strategy_replays(model, real[stock], actual)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", choices=("text", "timing"), default="text")
    parser.add_argument("--bge-cache")
    parser.add_argument("--april-cache", type=Path)
    parser.add_argument("--checkpoints", type=Path, help="Save reconstructed model weights")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(run(args.bge_cache, args.april_cache, args.checkpoints, args.probe), indent=2, allow_nan=False) + "\n")
