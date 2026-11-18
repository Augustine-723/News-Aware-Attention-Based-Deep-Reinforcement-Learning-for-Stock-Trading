# V8.23 M0–M3 事件輸入與固定 checkpoint 驗證

沿用 V8.21 的 2330 台積電市場快照與三筆隔離合成事件。每個決策時點的市場、持倉及市場缺值遮罩都相同。使用 V8.22 seed 42 的固定 M0 checkpoint 及其訓練期正規化統計，對四份觀測執行 CPU 推論。

| 輸入 | 事件遮罩與年齡 | 結構化事件欄位 | 合成文字向量 |
|---|---|---|---|
| M0 | 清空 | 清空 | 清空 |
| M1 | 保留，可從遮罩計數 | 清空 | 清空 |
| M2 | 保留 | 保留 | 清空 |
| M3 | 保留 | 保留 | 保留 |

| 決策時間（台北） | M1–M3 可用事件數 | M0／M1／M2／M3 貪婪動作 | M3 與 M0 最大絕對 Q 值差 |
|---|---:|---|---:|
| 2024-02-02 20:00 | 0 | Cash／Cash／Cash／Cash | 0 |
| 2024-02-05 20:00 | 1 | Cash／Cash／Cash／Cash | 0.003389 |
| 2024-02-15 20:00 | 3 | Long／Long／Long／Long | 0.003341 |

無事件時四路 Q 值完全一致。兩個有事件的決策點出現數值差異，但同日四路仍選同一動作。固定模型在訓練時僅看過空事件輸入。本輪驗證事件遮罩、模型輸入與輸出，未使用新聞訓練或執行報酬比較。M4 仍需另外接入可追溯的事件重要性訓練標籤。

## 重現

若本機已有 V8.22 checkpoint，可直接執行第二個指令。

```bash
python -m stock_rl.evaluation.market_drl_pilot --checkpoints checkpoints/v8_22 > market_drl_pilot.json
python -m stock_rl.evaluation.news_input_pilot --checkpoint checkpoints/v8_22/m0-seed-42.pt
python -m pytest -q tests/test_news_input_pilot.py tests/test_paired_observation.py
```

[逐筆輸入與 Q 值結果](../results/v8_23_news_input_pilot.json)
