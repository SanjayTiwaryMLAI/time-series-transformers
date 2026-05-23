# 📈 Time Series Transformers

Temporal Fusion Transformer (TFT), PatchTST, and iTransformer for real-world forecasting — stock prices, energy demand, and weather data.

## 🚀 Models Covered
| Model | Strength | Use Case |
|-------|----------|----------|
| Temporal Fusion Transformer | Interpretability | Multi-horizon forecasting |
| PatchTST | Patch-based attention | Long-sequence time series |
| iTransformer | Inverted attention | Multivariate forecasting |

## 📁 Structure
```
time-series-transformers/
├── models/
│   ├── patch_tst.py          # PatchTST implementation
│   ├── tft.py                # Temporal Fusion Transformer
│   └── forecaster.py         # Unified forecasting interface
├── data/
│   └── preprocess.py         # Data loading & normalization
├── requirements.txt
└── README.md
```

## ⚡ Quick Start
```python
from models.forecaster import TimeSeriesForecaster
model = TimeSeriesForecaster(model_type="patchtst")
model.fit(train_df, target_col="price", horizon=24)
preds = model.predict(test_df)
```
