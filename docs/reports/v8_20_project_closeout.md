# V8.20 research closeout

## Reproduce

Run from the repository root with `requirements.txt` installed.

```bash
python -m stock_rl.evaluation.market_baseline > market_baseline.json
python main.py
python -m pytest -q
```

The first command uses the cached TWSE market files and writes a deterministic research record.
The existing main entry point completed all three training stages and evaluation on synthetic market and event data.
The market-only result matched byte-for-byte across two runs.
Full pytest passed 260 tests with two dependency warnings (CPU PyTorch 2.5.1, pinned BGE snapshot).
Full-suite semantic checks also need Transformers and the pinned BGE snapshot in the local Hugging Face cache.
For the CPU environment used here, install `transformers==4.44.2`, then run:

```bash
python - <<'PY'
from huggingface_hub import snapshot_download
from stock_rl.evaluation.provenance import BGE_SMALL_ZH_MANIFEST, load_trusted_pretrained_embedding_engine

manifest = BGE_SMALL_ZH_MANIFEST
snapshot_download(
    manifest.model_id,
    revision=manifest.revision,
    allow_patterns=[
        "config.json", "model.safetensors", "vocab.txt",
        "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
    ],
)
assert load_trusted_pretrained_embedding_engine() is not None
PY
```

## Market-only replay

The fixed evaluation period contains 56 sessions per stock from 2024-01-02 through
2024-03-29. Earlier cached sessions provide the historical context window.
The replay compares three independent long/cash policies with identical simulated costs.
The 20-day moving-average rule uses closes available by the decision at 20:00 Taipei.
The existing execution simulator applies each target position at the next recorded trading
session's open. The final session contributes closing equity and has no subsequent order.
Return is measured from starting cash to final marked equity, with no terminal liquidation.
The replay uses cached prices and the simulator cost model. Realized fills and total returns
with distributions require separate evidence. No policy parameters were fitted to this sample.

| Stock | Cash return | Buy-and-hold return | SMA20 return | Buy-and-hold max drawdown | SMA20 max drawdown |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2330 | 0.00% | 33.21% | 24.18% | -3.95% | -3.95% |
| 2603 | 0.00% | 11.18% | -5.46% | -13.87% | -19.51% |
| 2881 | 0.00% | 8.93% | 3.93% | -2.81% | -4.59% |

The [result JSON](../results/v8_20_market_baseline.json) records file hashes, the shared
stock/TAIEX sessions, target actions, costs, and daily equity. Each stock makes 55
decisions, with zero populated event slots. Buy-and-hold led the rule on these three
stocks in this period. This is a small descriptive benchmark.

## Research completion criteria

| Component | Status |
| --- | --- |
| Three-stage model and tensor contract | Synthetic integration demonstration |
| Market-only policy replay | Real cached OHLCV, estimated market-data availability |
| Historical disclosure intake | Existing CSV/JSON CLI, private candidate export and formal gate |
| Historical disclosure evidence | One matched SEC filing for 2330, zero verified MOPS originals and zero historical provider delivery receipts |
| Head-to-head news contribution and Formal PIT historical training | Pending dated, version-specific, licensed announcement exports and historical provider availability evidence |

The [sample request](../inquiries/twse_historical_disclosure_inquiry.md) records the
authorized acquisition route. The [JASM evidence record](../evidence/ev_2330_jasm_expansion_20240206.md)
covers the historical case study. A second phase can use a paired market-only and news-aware
experiment when both policies can observe the same historical market sessions and event
availability can be established for each original version.
