import json

import pandas as pd

from stock_rl.config.schema import ACTION_CASH, CURRENT_ACTIVE_SCHEMA_VERSION
from stock_rl.data.normalizer import SplitManifest
from stock_rl.evaluation.market_drl_pilot import PHASES, ROOT, evaluate, prepare_inputs


def test_pilot_splits_and_normalizer_are_chronological():
    protocol = json.loads((ROOT / "configs/v8_22_study.json").read_text())
    protocol["stocks"] = ["2330"]
    boundaries = [
        pd.Timestamp(f"{day} 20:00", tz=protocol["timezone"])
        for phase in PHASES for day in protocol["splits"][phase]
    ]
    manifest = SplitManifest.create("test-m0", CURRENT_ACTIVE_SCHEMA_VERSION, *boundaries)
    inputs, normalizer = prepare_inputs(protocol, manifest)
    train, validation, test = (inputs[phase]["2330"] for phase in PHASES)
    assert train[0][-1] < validation[0][0] < validation[0][-1] < test[0][0]
    assert normalizer.fitted_train_window == (boundaries[0], boundaries[1])
    assert all(not observation.event_mask.any() for sample in (train, validation, test)
               for observation in sample[1])
    cash = evaluate(test, lambda *_: ACTION_CASH)
    assert len(cash["target_actions"]) == len(test[0]) - 1
    assert cash["session_dates"][-1] == "2024-03-29"
    assert cash["net_return"] == 0.0
    assert cash["sharpe"] is None
