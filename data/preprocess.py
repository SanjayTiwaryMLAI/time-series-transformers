"""Data loading, normalization and sliding-window dataset."""
import numpy as np
import pandas as pd
import yfinance as yf
from torch.utils.data import Dataset
import torch


def fetch_stock_data(ticker: str = "AAPL", period: str = "5y") -> pd.DataFrame:
    df = yf.download(ticker, period=period)[["Open", "High", "Low", "Close", "Volume"]]
    df.dropna(inplace=True)
    return df


class TimeSeriesDataset(Dataset):
    def __init__(self, data: np.ndarray, seq_len: int = 512, pred_len: int = 96):
        self.seq_len  = seq_len
        self.pred_len = pred_len
        # Normalize
        self.mean = data.mean(0)
        self.std  = data.std(0) + 1e-8
        self.data = (data - self.mean) / self.std

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_len]
        y = self.data[idx + self.seq_len : idx + self.seq_len + self.pred_len]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


if __name__ == "__main__":
    df  = fetch_stock_data("AAPL")
    ds  = TimeSeriesDataset(df.values)
    x, y = ds[0]
    print(f"Input seq: {x.shape}, Target: {y.shape}")
