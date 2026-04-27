#Importing all the important libraries
import math
import os
import warnings
import logging
import tempfile
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import TensorDataset, DataLoader
import yfinance as yf
import pandas as pd
import numpy as np

# We're muting warnings here because PyTorch Lightning loves to yell about minor
# hardware configuration stuff that doesn't actually stop the script from running.
warnings.filterwarnings("ignore")
logging.getLogger("lightning").setLevel(logging.ERROR)
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)

# Rich is just for making the terminal output look pretty instead of a wall of white text.
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.theme import Theme

console = Console(theme=Theme({
    "up":   "bold green",
    "down": "bold red",
    "flat": "bold yellow",
    "info": "dim cyan",
    "good": "bold green",
    "bad":  "bold red",
    "warn": "bold yellow",
}))

# ─── v7 Core Constants ────────────────────────────────────────────────────────
# These are the hardcoded rules of the game.
TEMPORAL_GAP  = 30    # If we train on Monday, we don't test on Tuesday. We wait 30 days to avoid "data leakage" (the model memorizing recent trends).
HIDDEN_SIZE   = 64    # The size of the LSTM's memory. Smaller (64 vs 128) prevents the model from overfitting on small stock datasets.
BASE_POS_SIZE = 0.15  # We risk a flat 15% of our capital per trade. No more dynamic sizing that blows up accounts.
BATCH_SIZE    = 32    # We feed the network 32 days of sequences at a time to average out the noise before updating weights.
PATIENCE      = 25    # If the model doesn't improve for 25 epochs, we kill the training to save time.
MAX_EPOCHS    = 150   # Maximum number of times we loop through the entire dataset.


# ─── Hardware ─────────────────────────────────────────────────────────────────
# Let's figure out what engine we're running on. Nvidia GPU (cuda), Apple Silicon (mps), or just a standard CPU.
def detect_device() -> str:
    if torch.cuda.is_available():        return "cuda"
    if (hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()): return "mps"
    return "cpu"

DEVICE = detect_device()

# Just a quick check to see if we're trading in India (NSE/BSE) so we can format currency and taxes right.
INDIAN_SUFFIXES = (".NS", ".BO", ".BSE")

def is_indian(ticker: str) -> bool:
    return any(ticker.upper().endswith(s) for s in INDIAN_SUFFIXES)

def get_currency(ticker: str) -> str:
    return "₹" if is_indian(ticker) else "$"


# ─── Regime Filter ────────────────────────────────────────────────────────────
# The market tide lifts all boats. We need to know if the broader market (S&P 500, etc.) is trending up or down.
def infer_us_regime_index(ticker: str, period: str = "12y") -> tuple:
    # If it's a US stock, we test it against S&P, NASDAQ, and DOW to see which one it mimics the most.
    candidates = [("^GSPC", "S&P 500"), ("^NDX", "NASDAQ 100"), ("^DJI", "Dow Jones")]
    try:
        stock = yf.download(ticker, period=period, progress=False, auto_adjust=True)
        if stock.empty:
            return "^GSPC", "S&P 500"
        
        # Clean up weird Yahoo Finance multi-index formatting
        if isinstance(stock.columns, pd.MultiIndex):
            stock.columns = stock.columns.get_level_values(0)
            
        # Get daily percentage changes to compare correlation
        s_ret = stock["Close"].ffill().pct_change().dropna()
        if len(s_ret) < 100:
            return "^GSPC", "S&P 500"
            
        best_sym, best_name, best_corr = "^GSPC", "S&P 500", -1.0
        
        # Loop through indices, find the highest correlation (which index does the stock follow most closely?)
        for sym, name in candidates:
            idx = yf.download(sym, period=period, progress=False, auto_adjust=True)
            if idx.empty: continue
            if isinstance(idx.columns, pd.MultiIndex):
                idx.columns = idx.columns.get_level_values(0)
            i_ret = idx["Close"].ffill().pct_change().dropna()
            
            aligned = pd.concat([s_ret, i_ret], axis=1, join="inner").dropna()
            if len(aligned) < 60: continue
            corr = float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1]))
            if np.isfinite(corr) and corr > best_corr:
                best_corr, best_sym, best_name = corr, sym, name
        return best_sym, best_name
    except Exception:
        return "^GSPC", "S&P 500"


def fetch_regime_series(ticker: str, period: str = "max") -> pd.Series:
    # Generates a series of 1s (Bull Market), -1s (Bear Market), or 0s (Chop).
    if is_indian(ticker):
        index_ticker, regime_name = "^NSEI", "NIFTY 50"
    else:
        index_ticker, regime_name = infer_us_regime_index(ticker, period=period)
    try:
        raw = yf.download(index_ticker, period=period, progress=False, auto_adjust=True)
        if raw.empty: return pd.Series(dtype=int)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        close  = raw["Close"].ffill()
        
        # Logic: If the index is > 2% above its 200-day EMA, it's a bull regime.
        # If it's > 2% below, bear regime. Otherwise, flat.
        ema200 = close.ewm(span=200, adjust=False).mean()
        ratio  = close / ema200 - 1.0
        regime = pd.Series(0, index=close.index)
        regime[ratio >  0.02] =  1
        regime[ratio < -0.02] = -1
        
        # Attach some metadata so we can print it later
        regime.attrs["regime_symbol"] = index_ticker
        regime.attrs["regime_name"]   = regime_name
        return regime
    except Exception:
        return pd.Series(dtype=int)


# ─── Indicators ───────────────────────────────────────────────────────────────
# These are the mathematical inputs for our neural network. Think of these as the "senses" of the AI.

def compute_rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    # Relative Strength Index. Formula: 100 - [100 / ( 1 + (Average Gain / Average Loss) )]
    # Tells us if an asset is overbought (usually > 70) or oversold (usually < 30).
    delta    = np.diff(close, prepend=close[0]) # Daily price changes
    gain     = np.where(delta > 0,  delta, 0.0) # Separate the green days...
    loss     = np.where(delta < 0, -delta, 0.0) # ...from the red days.
    avg_gain = pd.Series(gain).ewm(span=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-8) # +1e-8 prevents division by zero errors
    return 100.0 - (100.0 / (1.0 + rs))

def compute_macd_hist(close: np.ndarray, fast=12, slow=26, signal=9) -> np.ndarray:
    # MACD Histogram = MACD Line (12EMA - 26EMA) - Signal Line (9EMA of MACD Line)
    # Measures momentum. If it's crossing above 0, momentum is swinging bullish.
    c  = pd.Series(close)
    ml = c.ewm(span=fast, adjust=False).mean() - c.ewm(span=slow, adjust=False).mean()
    sl = ml.ewm(span=signal, adjust=False).mean()
    # We normalize it by dividing by close price so it works across $1 stocks and $1000 stocks equally.
    return np.nan_to_num((ml - sl).values / (close + 1e-8), nan=0.0)

def compute_bollinger_pct(close: np.ndarray, window: int = 20) -> np.ndarray:
    # Where is the price relative to Bollinger Bands? (Moving average +/- 2 standard deviations)
    # 1.0 means touching the top band, 0.0 means touching the bottom band.
    c   = pd.Series(close)
    ma  = c.rolling(window).mean()
    std = c.rolling(window).std()
    pct = ((c - (ma - 2*std)) / (4*std + 1e-8)).values
    return np.clip(np.nan_to_num(pct, nan=0.5), 0.0, 1.0) - 0.5 # Shifted to center around 0

def compute_atr_ratio(high, low, close, period: int = 14) -> np.ndarray:
    # Average True Range (ATR) measures volatility. How wild are the price swings?
    # True Range = max(High-Low, High-PrevClose, Low-PrevClose)
    prev_close = np.concatenate([[close[0]], close[:-1]])
    tr  = np.maximum(high - low,
          np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    atr = pd.Series(tr).ewm(span=period, adjust=False).mean().values
    return np.nan_to_num(atr / (close + 1e-8), nan=0.0) # Normalized by price

def compute_obv_momentum(close, volume, period: int = 5) -> np.ndarray:
    # On-Balance Volume. If price closes up, add volume. If down, subtract volume.
    # We are calculating the *momentum* of OBV over the last 5 days to spot hidden buying/selling pressure.
    direction = np.sign(np.diff(close, prepend=close[0]))
    obv       = np.cumsum(direction * volume)
    obv_prev  = pd.Series(obv).shift(period).values
    mom = np.where(np.abs(obv_prev) > 1e-8,
                   (obv - obv_prev) / (np.abs(obv_prev) + 1e-8), 0.0)
    return np.nan_to_num(np.clip(mom, -1.0, 1.0), nan=0.0)

def compute_52w_position(close, window: int = 252) -> np.ndarray:
    # Where is the current price compared to its 52-week (252 trading days) high and low?
    # 0 = at 52w low. 1 = at 52w high. (We subtract 0.5 to center it around 0 for the neural network).
    c     = pd.Series(close)
    hi_52 = c.rolling(window, min_periods=1).max().values
    lo_52 = c.rolling(window, min_periods=1).min().values
    pos   = (close - lo_52) / (hi_52 - lo_52 + 1e-8)
    return np.nan_to_num(pos - 0.5, nan=0.0)

# We use "Robust Stats" (Median and Interquartile Range) instead of standard Mean/Variance scaling.
# Why? Because stock markets have insane outliers (like a stock dropping 50% in a day).
# Standard scaling gets wrecked by outliers. Robust scaling ignores them.
def fit_robust_stats(features: np.ndarray, eps: float = 1e-8):
    med = np.nanmedian(features, axis=0)
    iqr = np.maximum(
        np.nanpercentile(features, 75, axis=0) - np.nanpercentile(features, 25, axis=0),
        eps)
    return med.astype(np.float32), iqr.astype(np.float32)

def apply_robust_stats(features: np.ndarray, med, iqr) -> np.ndarray:
    # Apply the math: (Value - Median) / IQR. Then clip extreme values to -4 and 4 to protect the network.
    return np.clip(
        np.nan_to_num((features - med) / (iqr + 1e-8), nan=0.0),
        -4.0, 4.0).astype(np.float32)


# ─── Feature Engineering ──────────────────────────────────────────────────────

N_FEATURES = 22

def build_features(close, open_, high, low, vol) -> np.ndarray:
    """
    Here we assemble the 22 inputs that the AI will look at.
    CRITICAL RULE: Every feature looks STRICTLY backwards.
    If you include today's close to predict today's close, that's "Lookahead Bias" 
    and your backtest will lie to you by looking into the future.
    """
    # 1-5: Basic returns. How much did price/volume change from yesterday?
    close_ret = (close[1:] / close[:-1]) - 1
    open_ret  = (open_[1:] / close[:-1]) - 1
    high_ret  = (high[1:]  / close[:-1]) - 1
    low_ret   = (low[1:]   / close[:-1]) - 1
    vol_ret   = np.where(vol[:-1] > 0, (vol[1:] / vol[:-1]) - 1, 0.0)

    # 6: RSI scaled to -0.5 to 0.5
    rsi_raw   = np.nan_to_num(compute_rsi(close, 14)[1:] / 100.0 - 0.5, nan=0.0)

    # 7: Moving Average Ratio. Is the short-term trend (5-day) beating the medium-term (20-day)?
    ma5      = pd.Series(close).rolling(5).mean().values
    ma20     = pd.Series(close).rolling(20).mean().values
    ma_ratio = np.nan_to_num(((ma5 / (ma20 + 1e-8)) - 1)[1:], nan=0.0)

    # 8-13: Injecting our indicator functions from earlier
    vol5      = pd.Series(close_ret).rolling(5).std().fillna(0).values
    macd_norm = compute_macd_hist(close)[1:]
    bb_pct    = compute_bollinger_pct(close)[1:]
    atr_ratio = compute_atr_ratio(high, low, close)[1:]
    obv_mom   = compute_obv_momentum(close, vol)[1:]
    pos_52w   = compute_52w_position(close)[1:]

    # 14-16: Longer term momentum and position against the 50-day moving average
    close_s  = pd.Series(close)
    mom5     = np.nan_to_num(((close_s / close_s.shift(5)) - 1).values[1:],  nan=0.0)
    mom20    = np.nan_to_num(((close_s / close_s.shift(20)) - 1).values[1:], nan=0.0)
    ma50     = close_s.rolling(50, min_periods=10).mean()
    pos50    = np.nan_to_num(((close_s / (ma50 + 1e-8)) - 1).values[1:],     nan=0.0)

    # 17: Volatility Regime. Is the stock acting crazier over the last 5 days compared to the last 20?
    vol20      = pd.Series(close_ret).rolling(20, min_periods=5).std().fillna(0)
    vol_regime = np.nan_to_num((pd.Series(vol5) / (vol20 + 1e-8) - 1).values, nan=0.0)

    # 18: Dollar Volume Momentum. Is big money flowing in recently? (Price * Volume)
    dollar_vol = pd.Series(close[1:] * vol[1:])
    dv20       = dollar_vol.rolling(20, min_periods=5).mean()
    dv_mom     = np.nan_to_num(
        np.clip((dollar_vol / (dv20 + 1e-8) - 1).values, -5.0, 5.0), nan=0.0)

    # 19-22: Japanese Candlestick structure.
    # Tells the AI if buyers or sellers dominated the trading session today.
    candle_range  = high[1:] - low[1:] + 1e-8
    overnight_gap = np.nan_to_num(open_[1:] / close[:-1] - 1.0, nan=0.0)
    candle_body   = np.nan_to_num(
        np.clip((close[1:] - open_[1:]) / candle_range, -1.0, 1.0), nan=0.0)
    upper_wick    = np.nan_to_num(
        np.clip((high[1:] - np.maximum(close[1:], open_[1:])) / candle_range, 0.0, 1.0), nan=0.0)
    lower_wick    = np.nan_to_num(
        np.clip((np.minimum(close[1:], open_[1:]) - low[1:]) / candle_range, 0.0, 1.0), nan=0.0)

    # Stack all 22 columns together into a single matrix.
    feats = np.column_stack([
        open_ret, high_ret, low_ret, close_ret, vol_ret,
        rsi_raw, ma_ratio, vol5, macd_norm, bb_pct,
        atr_ratio, obv_mom, pos_52w,
        mom5, mom20, pos50, vol_regime, dv_mom,
        overnight_gap, candle_body, upper_wick, lower_wick,
    ]).astype(np.float32)
    assert feats.shape[1] == N_FEATURES
    return feats


# ─── Data Pipeline ────────────────────────────────────────────────────────────

def fetch_and_prepare(
    ticker: str,
    period: str       = "12y",
    seq_len: int      = 60,   # AI looks at the past 60 days of data to make 1 prediction
    train_frac: float = 0.65, # 65% of data to learn
    val_frac:   float = 0.175,# 17.5% of data to tune itself and prevent memorization
):
    # Grab the raw ticker data from Yahoo Finance
    raw = yf.download(ticker, period=period, progress=False, auto_adjust=True)
    if raw.empty or len(raw) < seq_len + 200:
        raise ValueError(
            f"Not enough data for {ticker} "
            f"(got {len(raw)} rows, need ≥{seq_len + 200}). Try a longer period.")
            
    # Clean up Yahoo's messy headers
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw = raw.ffill().dropna()

    # Convert to pure numpy arrays for speed
    close = raw["Close"].values.astype(np.float64)
    open_ = raw["Open"].values.astype(np.float64)
    high  = raw["High"].values.astype(np.float64)
    low   = raw["Low"].values.astype(np.float64)
    vol   = raw["Volume"].values.astype(np.float64)

    # Run the feature engineering function we just built
    raw_features = build_features(close, open_, high, low, vol)

    # Because we need 60 days of history to make a prediction, we can't start at day 0.
    min_k = seq_len - 1
    max_k = len(raw_features) - 2

    # Math to split the timeline into Train, Validation, and Test blocks.
    n_total = max_k - min_k + 1
    n_train = int(n_total * train_frac)
    n_val   = int(n_total * val_frac)

    # We add 'TEMPORAL_GAP' (30 days) of dead space between the sets.
    # This ensures that an event happening on the last day of training doesn't immediately 
    # bleed its effects into the first day of testing. It's an honesty check.
    tr_start_k   = min_k
    tr_end_k     = min_k + n_train - 1
    val_start_k  = tr_end_k  + 1 + TEMPORAL_GAP
    val_end_k    = val_start_k + n_val - 1
    test_start_k = val_end_k  + 1 + TEMPORAL_GAP

    if test_start_k > max_k:
        raise ValueError(
            f"Not enough data after 2×{TEMPORAL_GAP}-day gaps. "
            f"Use period='max' or reduce TEMPORAL_GAP.")

    features = raw_features

    # This loop builds "windows" of 60 days.
    # Input X: The last 60 days of features.
    # Target yd: 1.0 if the price goes up the next day, 0.0 if it drops.
    # Target yr: The exact percentage return from open to close (used for backtesting later).
    def build_split(k_start: int, k_end: int) -> dict:
        X_list, yd_list, yr_list = [], [], []
        dates_list, close_list   = [], []
        for k in range(k_start, min(k_end + 1, max_k + 1)):
            X_list.append(features[k - seq_len + 1 : k + 1])
            dir_ret = (close[k + 2] / (close[k + 1] + 1e-8)) - 1.0 # Close-to-close return
            bt_ret  = (close[k + 2] / (open_[k + 2]  + 1e-8)) - 1.0  # Open-to-close return
            
            # Did it go up? 1.0 for yes, 0.0 for no. This is binary classification.
            yd_list.append(1.0 if dir_ret > 0 else 0.0)
            yr_list.append(float(bt_ret))
            dates_list.append(raw.index[k + 2])
            close_list.append(float(close[k + 2]))
            
        # Convert lists to PyTorch Tensors (the data structure AI runs on)
        X_t  = torch.tensor(np.array(X_list),             dtype=torch.float32)
        yd_t = torch.tensor(np.array(yd_list, dtype=np.float32))
        yr_t = torch.tensor(np.array(yr_list, dtype=np.float32))
        return {"X": X_t, "yd": yd_t, "yr": yr_t,
                "dates": dates_list, "closes": np.array(close_list)}

    tr = build_split(tr_start_k, tr_end_k)
    vl = build_split(val_start_k, val_end_k)
    ts = build_split(test_start_k, max_k)

    # The absolute most recent 60 days to predict TOMORROW's action
    last_window = features[-seq_len:]

    if len(tr["X"]) == 0:
        raise ValueError("No training sequences were built — check seq_len and data length")
        
    # Scale all the data based ONLY on the training data to prevent lookahead bias
    train_vals = tr["X"].cpu().numpy().reshape(-1, N_FEATURES)
    med, iqr = fit_robust_stats(train_vals)

    # Apply the scaling
    tr["X"] = torch.tensor(apply_robust_stats(tr["X"].cpu().numpy(), med, iqr), dtype=torch.float32)
    vl["X"] = torch.tensor(apply_robust_stats(vl["X"].cpu().numpy(), med, iqr), dtype=torch.float32)
    ts["X"] = torch.tensor(apply_robust_stats(ts["X"].cpu().numpy(), med, iqr), dtype=torch.float32)
    last_window = apply_robust_stats(last_window, med, iqr)

    console.print(
        f"  [info]Sequences → Train: {len(tr['X'])} | "
        f"Val: {len(vl['X'])} | Test: {len(ts['X'])} "
        f"| Gaps: {TEMPORAL_GAP}d each[/]"
    )

    last_close  = float(raw["Close"].iloc[-1])

    return {
        "tr": tr, "vl": vl, "ts": ts,
        "last_window":     last_window,
        "last_close":      last_close,
        "n_train_samples": len(tr["X"]),
        "n_val_samples":   len(vl["X"]),
        "n_test_samples":  len(ts["X"]),
    }

# DataLoader wraps our tensors so they can be fed into the AI in chunks (batches of 32).
# We shuffle the training data so the AI doesn't memorize the chronological order.
def make_dataloaders(data: dict):
    pin = DEVICE == "cuda"
    tr, vl = data["tr"], data["vl"]
    train_dl = DataLoader(
        TensorDataset(tr["X"], tr["yd"]),
        batch_size=BATCH_SIZE, shuffle=True,
        pin_memory=pin, num_workers=0,
    )
    val_dl = DataLoader(
        TensorDataset(vl["X"], vl["yd"]),
        batch_size=BATCH_SIZE, pin_memory=pin, num_workers=0,
    )
    return train_dl, val_dl


# ─── Focal Loss ───────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    # Standard AI loss functions (like Binary Cross Entropy) get lazy. 
    # If a stock goes up 60% of the time, standard AI will just guess "UP" every day.
    # Focal Loss fixes this. It mathematical scales down the penalty for "easy" predictions
    # and severely punishes the AI for getting "hard" predictions wrong.
    def __init__(self, gamma: float = 1.5, alpha: float = 0.5):
        super().__init__()
        self.gamma = gamma # Focusing parameter: higher means it focuses more on hard examples
        self.alpha = alpha # Balances the weight between UP and DOWN classes

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        # Basic prediction error
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        # How confident was the AI?
        prob = torch.sigmoid(logits)
        pt = prob * targets + (1.0 - prob) * (1.0 - targets)
        
        alpha_t = torch.where(targets == 1.0,
                              torch.tensor(self.alpha, device=targets.device),
                              torch.tensor(1.0 - self.alpha, device=targets.device))
                              
        # The magic formula: (1 - prediction_confidence) ^ gamma
        # If the AI was 99% confident and right, weight is near 0.
        # If AI was 10% confident and right, weight is high, forcing it to learn from it.
        focal_weight = (1.0 - pt) ** self.gamma
        return (alpha_t * focal_weight * bce).mean()


# ─── Model ───────────────────────────────────────────────────────────────────

# This teaches the LSTM which days in the 60-day window actually matter. 
# Did earnings hit 5 days ago? Pay "attention" to that day, ignore the flat chop from 40 days ago.
class ScaledDotAttention(nn.Module):
    def __init__(self, H: int):
        super().__init__()
        # The query vector learns what an "important" day looks like in the hidden state space
        self.query = nn.Parameter(torch.randn(H) * 0.02)
        self.scale = math.sqrt(H)

    def forward(self, hiddens: torch.Tensor) -> torch.Tensor:
        # Score each day's hidden state against the query vector
        scores  = torch.matmul(hiddens, self.query) / self.scale
        # Softmax turns the scores into percentages that add up to 100%
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)
        # Multiply the days by their importance weight and sum them up into one final "context" vector
        return (hiddens * weights).sum(dim=1)


class LSTM_v7(L.LightningModule):
    # This is the actual Brain. 
    def __init__(
        self,
        input_size:    int   = N_FEATURES, # 22
        hidden_size:   int   = HIDDEN_SIZE, # 64
        lr:            float = 3e-4,       # Learning rate (how big of steps it takes when learning)
        dropout:       float = 0.4,        # Randomly turns off 40% of neurons to prevent memorizing the data
        warmup_epochs: int   = 5,
        max_epochs:    int   = MAX_EPOCHS,
        focal_gamma:   float = 1.5,
    ):
        super().__init__()
        self.save_hyperparameters()
        H = hidden_size
        self.lr        = lr
        self.warmup_ep = warmup_epochs
        self.max_ep    = max_epochs

        self.input_ln      = nn.LayerNorm(input_size) # Normalizes the data again as it enters
        self.lstm          = nn.LSTM(input_size, H, num_layers=2, # A 2-layer LSTM reading the time sequence
                                     dropout=dropout, batch_first=True)
        self.lstm_ln       = nn.LayerNorm(H)
        self.attn          = ScaledDotAttention(H) # Our custom attention mechanism
        
        # A "Skip Connection" or Residual projection. 
        # It takes the very last day (day 60) and pipes it directly to the output, bypassing the LSTM memory.
        # This guarantees the model doesn't "forget" what the price is literally right now.
        self.residual_proj = nn.Linear(input_size, H, bias=False)
        self.drop          = nn.Dropout(dropout)
        
        # The final decision maker. Takes the memory + attention + current day, and squashes it down to 1 number.
        self.head = nn.Sequential(
            nn.LayerNorm(H),
            nn.Linear(H, H // 2),
            nn.GELU(), # Smoother version of ReLU, helps gradients flow
            nn.Dropout(dropout),
            nn.Linear(H // 2, 1), # Output layer
        )
        self.focal = FocalLoss(gamma=focal_gamma)
        self._init_weights()

    def _init_weights(self):
        # We initialize the network weights with specific mathematical distributions (Orthogonal, Xavier)
        # to prevent the "vanishing gradient" problem where the AI just stops learning altogether.
        for name, p in self.lstm.named_parameters():
            if "weight_hh" in name:   nn.init.orthogonal_(p)
            elif "weight_ih" in name: nn.init.xavier_uniform_(p)
            elif "bias" in name:
                p.data.zero_()
                n = p.size(0)
                # Forget gate bias initialization to 1.0 (an old LSTM trick to make it remember long-term better)
                p.data[n // 4 : n // 2].fill_(1.0)
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None: nn.init.zeros_(m.bias)
        nn.init.xavier_uniform_(self.residual_proj.weight, gain=0.1)

    def _encode(self, X: torch.Tensor) -> torch.Tensor:
        X_norm     = self.input_ln(X)
        hiddens, _ = self.lstm(X_norm)
        context    = self.attn(self.lstm_ln(hiddens))
        skip       = self.residual_proj(X_norm[:, -1, :]) # Grab the last row (current day)
        return self.drop(context + skip) # Combine memory context with the skip connection

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # Feed data through encoder, then through the final linear head
        return self.head(self._encode(X)).squeeze(-1)

    def training_step(self, batch, _):
        # This is what happens under the hood during the train loop
        x, y = batch
        logits = self(x) # AI's raw guesses
        loss = self.focal(logits, y) # Grade the guesses
        preds = (torch.sigmoid(logits) > 0.5).float() # Convert raw numbers to binary 1 or 0
        acc = (preds == y).float().mean() # Did it match the target?
        
        self.log("train_loss", loss, on_epoch=True, prog_bar=False)
        self.log("train_acc", acc, on_epoch=True, prog_bar=False)
        return loss

    def validation_step(self, batch, _):
        # Same as training, but we don't update weights. We just use this to see if it's actually learning
        # or just memorizing the training data.
        x, y = batch
        self.eval() # Turn off dropout
        with torch.no_grad(): # Don't track gradients (saves memory)
            logits = self(x)
            loss = self.focal(logits, y)
            preds = (torch.sigmoid(logits) > 0.5).float()
            acc = (preds == y).float().mean()
        self.log("val_loss", loss, on_epoch=True, prog_bar=False)
        self.log("val_acc", acc, on_epoch=True, prog_bar=False)

    def configure_optimizers(self):
        # AdamW is the optimizer making the actual adjustments to the weights based on the loss.
        opt = AdamW(self.parameters(), lr=self.lr, weight_decay=1e-4, amsgrad=True)
        
        # Learning Rate Scheduler:
        # Warmup for 5 epochs (slowly increase learning rate so we don't break early fragile weights)
        # Then Cosine Annealing (smoothly decay learning rate to 0 to "settle" into the global minimum)
        def lr_lambda(epoch):
            if epoch < self.warmup_ep:
                return (epoch + 1) / max(self.warmup_ep, 1)
            t = (epoch - self.warmup_ep) / max(1, self.max_ep - self.warmup_ep)
            return 0.5 * (1.0 + math.cos(math.pi * t))
            
        return {"optimizer": opt, "lr_scheduler": {
            "scheduler": torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda),
            "interval": "epoch"}}


# ─── Evaluation ───────────────────────────────────────────────────────────────

@torch.no_grad()
def get_probabilities(model: LSTM_v7, X: torch.Tensor) -> np.ndarray:
    # Feed all data through the trained model to get the final confidence scores (0.0 to 1.0)
    model.eval()
    p_list = []
    # Doing it in batches of 512 so we don't run out of RAM/VRAM
    for i in range(0, len(X), 512):
        logit = model(X[i:i+512].to(model.device))
        p_list.append(torch.sigmoid(logit).cpu().numpy())
    return np.concatenate(p_list)


def compute_metrics(p_up: np.ndarray, y_dir: torch.Tensor,
                    y_raw: torch.Tensor, threshold: float = 0.50) -> dict:
    # Calculates our precision, recall, and total accuracy. 
    y_bin   = y_dir.cpu().numpy()
    actuals = y_raw.cpu().numpy()

    pred_up   = p_up > threshold
    actual_up = y_bin > 0.5
    correct   = int(np.sum(pred_up == actual_up))
    dir_acc   = correct / len(pred_up) * 100

    # True positives, false positives, etc.
    tp_up = int(np.sum( pred_up &  actual_up))
    fp_up = int(np.sum( pred_up & ~actual_up))
    fn_up = int(np.sum(~pred_up &  actual_up))
    tp_dn = int(np.sum(~pred_up & ~actual_up))
    fp_dn = int(np.sum(~pred_up &  actual_up))

    prec_up = tp_up / (tp_up + fp_up + 1e-8) * 100
    rec_up  = tp_up / (tp_up + fn_up + 1e-8) * 100
    prec_dn = tp_dn / (tp_dn + fp_dn + 1e-8) * 100
    brier   = float(np.mean((p_up - y_bin) ** 2)) # Brier score measures the *accuracy* of the probabilities. Lower is better.

    return {
        "n": len(p_up), "correct": correct, "wrong": len(p_up) - correct,
        "dir_acc":     round(dir_acc, 2),
        "prec_up":     round(prec_up, 2), "rec_up": round(rec_up, 2),
        "prec_dn":     round(prec_dn, 2),
        "avg_win":     round((prec_up + prec_dn) / 2, 2),
        "brier_score": round(brier, 4),
        "called_up":   int(np.sum(pred_up)),
        "called_dn":   int(np.sum(~pred_up)),
        "p_up_raw":    p_up,
        "actuals_raw": actuals,
    }

# v7.1 FIX: Enforce highly restrictive trade rate (0.20)
def compute_adaptive_thresholds(p_up_val: np.ndarray,
                                 target_trade_frac: float = 0.20) -> tuple:
    # Instead of blindly trading if prediction > 50%, we dynamically figure out what the top 10%
    # and bottom 10% (for a total of 20% trade rate) of the AI's confidence levels are.
    # We only trade those hyper-confident signals.
    p = np.asarray(p_up_val[np.isfinite(p_up_val)], dtype=np.float64)
    if len(p) < 50:
        return 0.55, 0.45, 0.20
    target = float(np.clip(target_trade_frac, 0.10, 0.40))
    tail   = target / 2.0
    dn     = float(np.quantile(p, tail))
    up     = float(np.quantile(p, 1.0 - tail))
    centre = float(np.median(p))
    dn     = min(dn, centre - 0.01)
    up     = max(up, centre + 0.01)
    # Clip them so the thresholds never get absurdly loose or incredibly tight
    dn     = float(np.clip(dn, 0.20, 0.48))
    up     = float(np.clip(up, 0.52, 0.80))
    actual_rate = float(np.mean((p >= up) | (p <= dn)))
    return up, dn, actual_rate


# ─── Training ─────────────────────────────────────────────────────────────────

def train_model(data: dict, seed: int = 42) -> tuple:
    # Seed everything so if we run this script twice, we get the exact same results.
    import random as _random
    torch.manual_seed(seed)
    np.random.seed(seed)
    _random.seed(seed)
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass

    train_dl, val_dl = make_dataloaders(data)

    n_tr              = data["n_train_samples"]
    batches_per_epoch = max(1, n_tr // BATCH_SIZE)
    console.print(
        f"  [info]Training: {n_tr} samples | {batches_per_epoch} batches/epoch | "
        f"batch={BATCH_SIZE} | max_epochs={MAX_EPOCHS} | patience={PATIENCE}[/]"
    )

    # Setup saving. We monitor the validation loss. The epoch where val_loss is lowest
    # is the "best" brain, so we save it to disk.
    ckpt_dir   = tempfile.mkdtemp()
    checkpoint = ModelCheckpoint(
        dirpath=ckpt_dir, filename="best", monitor="val_loss",
        mode="min", save_top_k=1, verbose=False)
        
    # Early stopping: If 25 epochs go by and val_loss doesn't improve, pull the plug.
    early_stop = EarlyStopping(
        monitor="val_loss", patience=PATIENCE, min_delta=1e-5,
        mode="min", verbose=False)

    accelerator = ("gpu" if DEVICE == "cuda" else
                   "mps" if DEVICE == "mps" else "cpu")
                   
    trainer = L.Trainer(
        accelerator=accelerator, devices=1,
        enable_model_summary=False, enable_progress_bar=False,
        logger=False, log_every_n_steps=5, max_epochs=MAX_EPOCHS,
        callbacks=[early_stop, checkpoint],
        gradient_clip_val=1.0, gradient_clip_algorithm="norm", # Stops gradients from exploding to infinity
        precision="16-mixed" if DEVICE == "cuda" else "32", # Half-precision math to speed up GPUs
        deterministic=True,
    )

    # Instantiate the brain
    model = LSTM_v7(
        input_size=N_FEATURES, hidden_size=HIDDEN_SIZE,
        lr=3e-4, dropout=0.4, warmup_epochs=5,
        max_epochs=MAX_EPOCHS, focal_gamma=1.5,
    )

    # Let her rip.
    trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)

    # Reload the absolute best checkpoint from that run
    best_ckpt = checkpoint.best_model_path
    if best_ckpt and os.path.exists(best_ckpt):
        model = LSTM_v7.load_from_checkpoint(best_ckpt)
        console.print(f"  [info]Loaded best ckpt val_loss={checkpoint.best_model_score:.6f}[/]")

    epoch    = trainer.current_epoch
    val_loss = trainer.callback_metrics.get("val_loss", torch.tensor(float("nan"))).item()
    val_acc  = trainer.callback_metrics.get("val_acc",  torch.tensor(float("nan"))).item()
    n_updates = epoch * batches_per_epoch
    console.print(
        f"  [info]Stopped epoch {epoch}/{MAX_EPOCHS} | "
        f"val_loss={val_loss:.6f} | val_acc={val_acc*100:.1f}% | "
        f"~{n_updates:,} gradient updates[/]"
    )

    tr_m = compute_metrics(get_probabilities(model, data["tr"]["X"]),
                            data["tr"]["yd"], data["tr"]["yr"])
    vl_m = compute_metrics(get_probabilities(model, data["vl"]["X"]),
                            data["vl"]["yd"], data["vl"]["yr"])
    ts_m = compute_metrics(get_probabilities(model, data["ts"]["X"]),
                            data["ts"]["yd"], data["ts"]["yr"])

    return model, val_loss, tr_m, vl_m, ts_m


# ─── Today's Signal ───────────────────────────────────────────────────────────

@torch.no_grad()
def predict_today(model: LSTM_v7, last_window: np.ndarray,
                  n_mc: int = 10) -> tuple:
    # We grab the most recent 60 days of real data, right up to market close.
    base = torch.tensor(last_window, dtype=torch.float32,
                        device=model.device).unsqueeze(0)

    was_training = model.training
    model.eval() # Turn off dropout for the real, deterministic prediction
    with torch.no_grad():
        p_det = float(torch.sigmoid(model(base)).item())

    # "Monte Carlo Dropout"
    # We turn dropout back on (randomly killing neurons) and run it 10 more times.
    # If the network is highly confident, it will output the same prediction even with dead neurons.
    # If it's guessing, the predictions will swing wildly. This gives us our standard deviation (uncertainty).
    p_mc = []
    model.train()
    with torch.no_grad():
        for _ in range(max(1, n_mc)):
            p_mc.append(float(torch.sigmoid(model(base)).item()))

    if not was_training:
        model.eval()

    return p_det, float(np.std(p_mc))


# ─── Backtester ───────────────────────────────────────────────────────────────

@dataclass
class TradeRecord:
    # A simple container to hold individual trade data so we can log it.
    day_idx: int; date: object; direction: int
    p_up: float; actual_ret: float; trade_ret: float
    capital_after: float; correct: bool; regime: int; pos_size: float


@dataclass
class BacktestResult:
    # Container for all our final statistics.
    ticker: str; initial_capital: float; final_capital: float
    total_return_pct: float; cagr_pct: float
    sharpe_all: float; sharpe_active: float
    sortino: float; max_drawdown_pct: float; calmar: float
    n_trades: int; n_wins: int; n_losses: int
    win_rate_pct: float; profit_factor: float
    avg_win_pct: float; avg_loss_pct: float
    n_long: int; n_short: int; n_flat: int
    test_days: int; test_years: float; trades_per_year: float
    contra_trend_trades: int
    rt_cost_pct_of_final: float; tax_pct_of_final: float
    total_drag_pct: float
    equity_curve: list; trade_log: list
    bh_return_pct: float; bh_cagr_pct: float; alpha_pct: float
    avg_pos_size_pct: float
    verdict: str; verdict_color: str; issues: str

# v7.1 FIX: Default hold_days=3, take_profit_pct=0.06, stop_loss_pct=0.025
def run_backtest(
    ticker: str,
    p_up_arr:      np.ndarray,
    actuals:       np.ndarray,
    dates:         list,
    close_prices:  np.ndarray,
    regime_series: pd.Series,
    signal_up:    float = 0.55,
    signal_dn:    float = 0.45,
    initial_capital: float = 100_000.0,
    test_dir_acc: float = 50.0,
    n_test_samples: int = 0,
    slippage_rt_override: float | None = None,
    take_profit_pct: float | None = 0.06,  # 6% Take Profit
    stop_loss_pct:  float | None = 0.025, # 2.5% Stop Loss
    hold_days: int = 3, # Hold for 3 days to beat the spread
    hold_until_flip: bool = False,
) -> BacktestResult:

    indian = is_indian(ticker)

    # Taxes and fees. The invisible account killers.
    def get_rt_cost(direction: int) -> float:
        # If India, 0.45% cost to short, 0.22% to go long.
        if indian: return 0.0045 if direction == -1 else 0.0022
        # US is cheaper
        else:      return 0.0018 if direction == -1 else 0.0010

    # Slippage: You never get the exact price you want. The market slips against you.
    slippage_rt = 0.0010 if indian else 0.0005
    if slippage_rt_override is not None:
        slippage_rt = float(slippage_rt_override)

    tax_rate = 0.025 if indian else 0.015

    # Setup the tracking variables
    capital      = initial_capital
    peak_capital = initial_capital
    equity       = [capital]
    all_rets     = []
    trade_log    = []
    n_long = n_short = n_flat = contra_trend = 0
    wins, losses = [], []
    rt_costs_abs = 0.0
    slippage_abs = 0.0
    pos_sizes    = []

    i = 0
    n = len(p_up_arr)
    # The actual backtest loop steps forward day by day.
    while i < n:
        p = float(p_up_arr[i])
        actual = float(actuals[i])
        date = dates[i] if dates else i

        # Check regime.
        regime = 0
        if not regime_series.empty and hasattr(date, "date"):
            try:   regime = int(regime_series.asof(date))
            except Exception: regime = 0

        # Are we long (1), short (-1), or sitting cash (0)?
        raw_dir   = 1 if p > signal_up else (-1 if p < signal_dn else 0)
        direction = raw_dir

        # Drawdown logic. If our account is taking a beating, scale down our position sizes aggressively to protect capital.
        curr_dd = capital / (peak_capital + 1e-8) - 1.0
        if   curr_dd <= -0.20: direction = 0;  dd_scale = 0.0
        elif curr_dd <= -0.12: dd_scale = 0.40
        elif curr_dd <= -0.07: dd_scale = 0.70
        else:                  dd_scale = 1.0
        if direction == 0: dd_scale = 0.0

        # Regime Logic. If the AI wants to short a bull market, cut position size in half. (Contra-trend trade)
        regime_scale = 1.0
        if regime == 1 and raw_dir == -1:
            regime_scale = 0.50; contra_trend += 1
        elif regime == -1 and raw_dir == 1:
            regime_scale = 0.50; contra_trend += 1

        # Calculate final position size
        pos_size = float(np.clip(
            BASE_POS_SIZE * dd_scale * regime_scale, 0.0, 0.25))

        if   direction == 1:  n_long  += 1
        elif direction == -1: n_short += 1
        else:                 n_flat  += 1

        # If a trade was triggered...
        if direction != 0 and pos_size > 5e-4:
            entry_idx = i
            # We hold for `hold_days` (3), or until we hit the end of the array.
            exit_idx = min(n - 1, entry_idx + max(1, int(hold_days)) - 1)

            if hold_until_flip:
                for j in range(entry_idx + 1, n):
                    p_future = float(p_up_arr[j])
                    raw_dir_future = 1 if p_future > signal_up else (-1 if p_future < signal_dn else 0)
                    if raw_dir_future == -direction:
                        exit_idx = j
                        break

            # Calculate compounded return over those 3 days
            cum_ret = 1.0
            for j in range(entry_idx, exit_idx + 1):
                r = float(actuals[j])
                cum_ret *= (1.0 + direction * r)
            pnl_ret = (cum_ret - 1.0) * pos_size

            # Check if we hit our Take Profit or Stop Loss ceilings/floors.
            if take_profit_pct is not None:
                tp = float(take_profit_pct)
                if pnl_ret > tp * pos_size:
                    pnl_ret = tp * pos_size
            if stop_loss_pct is not None:
                sl = float(stop_loss_pct)
                if pnl_ret < -sl * pos_size:
                    pnl_ret = -sl * pos_size

            # Deduct the cost of doing business
            rt_cost = get_rt_cost(direction)
            slip = float(slippage_rt)
            trade_notional = capital * pos_size
            rt_cost_abs = trade_notional * rt_cost
            slippage_cost_abs = trade_notional * slip

            # Final net trade return
            trade_ret = pnl_ret - (rt_cost + slip) * pos_size
            capital = capital * (1.0 + trade_ret)

            rt_costs_abs += rt_cost_abs
            slippage_abs += slippage_cost_abs

            all_rets.append(trade_ret)
            is_win = trade_ret > 0
            (wins if is_win else losses).append(trade_ret)
            pos_sizes.append(pos_size)
            trade_log.append(TradeRecord(
                day_idx=entry_idx, date=dates[entry_idx] if dates else entry_idx,
                direction=direction, p_up=float(p), actual_ret=float(pnl_ret),
                trade_ret=float(trade_ret), capital_after=float(capital),
                correct=is_win, regime=regime, pos_size=pos_size,
            ))

            # Skip our loop index forward so we aren't opening new trades while holding this one
            i = exit_idx + 1
            peak_capital = max(peak_capital, capital)
            equity.append(max(capital, 1.0))
            continue
        else:
            all_rets.append(0.0)

        # Update highest watermark
        peak_capital = max(peak_capital, capital)
        equity.append(max(capital, 1.0))
        i += 1

    # End of test calculations
    gross_profit = max(equity[-1] - initial_capital, 0.0)
    tax_abs      = gross_profit * tax_rate
    net_final    = max(equity[-1] - tax_abs, 1.0)
    equity[-1]   = net_final

    total_costs_abs = rt_costs_abs + slippage_abs
    rt_cost_pct_f = (total_costs_abs / max(net_final, 1.0)) * 100.0
    tax_pct_f     = (tax_abs      / max(net_final, 1.0)) * 100.0
    total_drag    = rt_cost_pct_f + tax_pct_f

    total_ret = (net_final / initial_capital - 1.0) * 100.0
    n_days    = len(p_up_arr)
    n_years   = n_days / 252.0
    # Compound Annual Growth Rate
    cagr      = ((net_final / initial_capital) ** (1.0 / max(n_years, 0.1)) - 1.0) * 100.0

    # Sharpe ratio is Risk Adjusted Return. Formula: Mean Return / Standard Deviation of Return.
    # Anything over 1.0 is generally considered a viable edge.
    arr = np.array(all_rets)
    sharpe_all    = (arr.mean() / arr.std() * math.sqrt(252)) if arr.std() > 1e-10 else 0.0
    active        = arr[arr != 0] # Only look at days we actually made a trade
    sharpe_active = (active.mean() / active.std() * math.sqrt(252 / max(1, hold_days))) if (
        len(active) > 1 and active.std() > 1e-10) else 0.0
        
    # Sortino ratio is like Sharpe, but it only penalizes downward volatility. 
    dd_sq   = np.minimum(arr, 0.0) ** 2
    dd_dev  = math.sqrt(float(np.mean(dd_sq)) + 1e-16) * math.sqrt(252)
    sortino = arr.mean() * 252 / (dd_dev + 1e-8)

    # Max Drawdown: The deepest valley our account equity fell into from its peak.
    eq_arr = np.array(equity)
    peak   = np.maximum.accumulate(eq_arr)
    max_dd = float(((eq_arr - peak) / (peak + 1e-8) * 100).min())
    calmar = cagr / (abs(max_dd) + 1e-8)

    n_tr     = len(trade_log)
    win_rate = (len(wins) / n_tr * 100) if n_tr > 0 else 0.0
    avg_win  = float(np.mean(wins)   * 100) if wins   else 0.0
    avg_loss = float(np.mean(losses) * 100) if losses else 0.0
    # Profit factor: Gross win money / Gross loss money. Needs to be > 1.0 to survive.
    pf       = sum(wins) / (abs(sum(losses)) + 1e-8)

    # Buy and hold calculation for comparison
    if len(close_prices) >= 2:
        bh_ret  = (close_prices[-1] / (close_prices[0] + 1e-8) - 1.0) * 100.0
        bh_cagr = ((close_prices[-1] / (close_prices[0] + 1e-8)) ** (
                   1.0 / max(n_years, 0.1)) - 1.0) * 100.0
    else:
        bh_ret = bh_cagr = 0.0

    # Alpha: The percentage points our strategy beat simply holding the stock.
    alpha = total_ret - bh_ret

    # Sanity checks. If it fails these, the model is junk and shouldn't be traded.
    issues_list = []
    tradable    = True
    if test_dir_acc < 53.0:
        tradable = False; issues_list.append(f"Test acc {test_dir_acc:.1f}% < 53%")
    if n_test_samples < 150:
        issues_list.append(f"Only {n_test_samples} test samples")
    if sharpe_active < 0.8:
        tradable = False; issues_list.append(f"Active Sharpe {sharpe_active:.2f} < 0.8")
    if total_ret < 0:
        tradable = False; issues_list.append(f"Negative return {total_ret:.1f}%")
    if max_dd < -25.0:
        issues_list.append(f"Max DD {max_dd:.1f}% > 25%")
    if pf < 1.1 and n_tr > 0:
        tradable = False; issues_list.append(f"Profit factor {pf:.2f} < 1.1")
    if n_tr < 5:
        issues_list.append(f"Only {n_tr} trades (low significance)")
    if contra_trend > 0:
        issues_list.append(f"{contra_trend} contra-trend signals at 50% size")

    if tradable and not issues_list:   verdict = "✅ TRADABLE"; color = "good"
    elif tradable:                     verdict = "⚠  MARGINAL"; color = "warn"
    else:                              verdict = "❌ NOT TRADABLE"; color = "bad"

    avg_ps = float(np.mean(pos_sizes)) * 100 if pos_sizes else 0.0

    return BacktestResult(
        ticker=ticker, initial_capital=initial_capital,
        final_capital=round(net_final, 2),
        total_return_pct=round(total_ret, 2), cagr_pct=round(cagr, 2),
        sharpe_all=round(sharpe_all, 3), sharpe_active=round(sharpe_active, 3),
        sortino=round(sortino, 3),
        max_drawdown_pct=round(max_dd, 2), calmar=round(calmar, 3),
        n_trades=n_tr, n_wins=len(wins), n_losses=len(losses),
        win_rate_pct=round(win_rate, 2), profit_factor=round(pf, 3),
        avg_win_pct=round(avg_win, 4), avg_loss_pct=round(avg_loss, 4),
        n_long=n_long, n_short=n_short, n_flat=n_flat,
        test_days=n_days, test_years=round(n_years, 2),
        trades_per_year=round(n_tr / max(n_years, 1e-6), 2),
        contra_trend_trades=contra_trend,
        rt_cost_pct_of_final=round(rt_cost_pct_f, 2),
        tax_pct_of_final=round(tax_pct_f, 2),
        total_drag_pct=round(total_drag, 2),
        equity_curve=equity, trade_log=trade_log,
        bh_return_pct=round(bh_ret, 2), bh_cagr_pct=round(bh_cagr, 2),
        alpha_pct=round(alpha, 2),
        avg_pos_size_pct=round(avg_ps, 2),
        verdict=verdict, verdict_color=color,
        issues=" | ".join(issues_list) if issues_list else "None",
    )


# ─── Display ──────────────────────────────────────────────────────────────────
# This entire section is just using the `rich` library to print neat tables in the terminal.
# It grabs the metrics dictionaries we built above and maps them to row/columns.

def print_model_report(ticker, tr_m, vl_m, ts_m, n_tr, n_vl, n_ts):
    tbl = Table(box=box.SIMPLE_HEAVY, show_header=True,
                header_style="bold white", border_style="bright_black", expand=True)
    tbl.add_column("Metric",             style="dim white")
    tbl.add_column(f"Train ({n_tr}d)",   justify="right", style="cyan")
    tbl.add_column(f"Val   ({n_vl}d)",   justify="right", style="yellow")
    tbl.add_column(f"TEST  ({n_ts}d)",   justify="right", style="bold green")

    def pct(v, g=53.0):
        return f"[{'green' if v>=g else 'red'}]{v:.2f}%[/]"
    def bs(v):
        return f"[{'green' if v<0.25 else 'red'}]{v:.4f}[/]"

    rows = [
        ("Samples",          str(tr_m["n"]),           str(vl_m["n"]),           str(ts_m["n"])),
        ("Correct / Wrong",  f"{tr_m['correct']}/{tr_m['wrong']}",
                              f"{vl_m['correct']}/{vl_m['wrong']}",
                              f"{ts_m['correct']}/{ts_m['wrong']}"),
        ("──","──","──","──"),
        ("Dir accuracy ★",   pct(tr_m["dir_acc"]),     pct(vl_m["dir_acc"]),     pct(ts_m["dir_acc"])),
        ("Brier score (↓)",  bs(tr_m["brier_score"]),  bs(vl_m["brier_score"]),  bs(ts_m["brier_score"])),
        ("──","──","──","──"),
        ("Called UP",        str(tr_m["called_up"]),   str(vl_m["called_up"]),   str(ts_m["called_up"])),
        ("Prec UP",          pct(tr_m["prec_up"]),     pct(vl_m["prec_up"]),     pct(ts_m["prec_up"])),
        ("Recall UP",        pct(tr_m["rec_up"]),      pct(vl_m["rec_up"]),      pct(ts_m["rec_up"])),
        ("──","──","──","──"),
        ("Called DOWN",      str(tr_m["called_dn"]),   str(vl_m["called_dn"]),   str(ts_m["called_dn"])),
        ("Prec DOWN",        pct(tr_m["prec_dn"]),     pct(vl_m["prec_dn"]),     pct(ts_m["prec_dn"])),
        ("──","──","──","──"),
        ("Avg win rate",     pct(tr_m["avg_win"]),     pct(vl_m["avg_win"]),     pct(ts_m["avg_win"])),
    ]
    for r in rows:
        tbl.add_row(*r)
    console.print(Panel(tbl,
        title=f"[bold cyan]{ticker} — Model Performance "
              f"(v7.1 | stride=1 | 30d gaps)[/]",
        padding=(0, 1)))


def print_backtest_panel(bt: BacktestResult):
    currency = get_currency(bt.ticker)
    ic = bt.initial_capital

    # Generates a tiny sparkline graph out of unicode blocks to show equity curve.
    eq    = np.array(bt.equity_curve)
    bars  = "▁▂▃▄▅▆▇█"
    rng   = max(eq.max() - eq.min(), 1e-8)
    idx   = np.round(np.linspace(0, len(eq)-1, 52)).astype(int)
    spark = "".join(bars[min(int((eq[j] - eq.min()) / rng * 7), 7)] for j in idx)
    sc    = "green" if bt.final_capital >= ic else "red"

    def c(v, tg, tr, fmt=".2f", sfx=""):
        s = f"{v:{fmt}}{sfx}"
        return (f"[green]{s}[/]" if v >= tg else
                f"[red]{s}[/]"   if v <= tr else
                f"[yellow]{s}[/]")

    tbl = Table(box=box.SIMPLE_HEAVY, show_header=True,
                header_style="bold white", border_style="bright_black", expand=True)
    tbl.add_column("Metric",     style="dim white")
    tbl.add_column("Strategy",   justify="right", style="cyan")
    tbl.add_column("Buy & Hold", justify="right", style="yellow")

    bh_final = ic * (1 + bt.bh_return_pct / 100)
    rows = [
        ("Initial Capital",
         f"{currency}{ic:,.0f}", "—"),
        ("Final Capital ✅ (after all costs+tax)",
         f"{currency}{bt.final_capital:,.0f}",
         f"{currency}{bh_final:,.0f}"),
        ("──","──","──"),
        ("Total Return",
         c(bt.total_return_pct, 15, 0, ".2f","%"),
         c(bt.bh_return_pct,   15, 0, ".2f","%")),
        ("CAGR",
         c(bt.cagr_pct,    10, 0, ".2f","%"),
         c(bt.bh_cagr_pct, 10, 0, ".2f","%")),
        ("Alpha vs B&H",
         c(bt.alpha_pct, 5, -5, ".2f","%"), "—"),
        ("──","──","──"),
        ("Sharpe (all days)",
         c(bt.sharpe_all,    1.0, 0.3, ".3f"), "—"),
        ("Sharpe (active ★)",
         c(bt.sharpe_active, 1.2, 0.5, ".3f"), "—"),
        ("Sortino",
         c(bt.sortino, 1.5, 0.5, ".3f"), "—"),
        ("Max Drawdown",
         f"[{'green' if bt.max_drawdown_pct>-15 else 'red'}]{bt.max_drawdown_pct:.2f}%[/]",
         "—"),
        ("Calmar",
         c(bt.calmar, 0.5, 0.2, ".3f"), "—"),
        ("──","──","──"),
        ("Total Trades",  str(bt.n_trades), "—"),
        ("Long / Short / Flat", f"{bt.n_long}/{bt.n_short}/{bt.n_flat}", "—"),
        ("Test span",  f"{bt.test_days}d ({bt.test_years:.2f}y)", "—"),
        ("Trades/year", f"{bt.trades_per_year:.1f}", "—"),
        ("Avg position size", f"{bt.avg_pos_size_pct:.1f}%", "—"),
        ("Contra-trend (50% size)", f"[yellow]{bt.contra_trend_trades}[/]", "—"),
        ("──","──","──"),
        ("Win Rate",
         c(bt.win_rate_pct, 53, 45, ".2f","%"), "—"),
        ("Profit Factor",
         c(bt.profit_factor, 1.2, 1.0, ".3f"), "—"),
        ("──","──","──"),
        ("[bold]COST BREAKDOWN (% of FINAL capital)[/]", "", ""),
        ("  Transaction costs", f"[red]-{bt.rt_cost_pct_of_final:.2f}%[/]", "—"),
        ("  Tax reserve",       f"[red]-{bt.tax_pct_of_final:.2f}%[/]",     "—"),
        ("  Total drag",        f"[red]-{bt.total_drag_pct:.2f}%[/]",        "—"),
        ("  (Already deducted from Final Capital above)", "", ""),
        ("──","──","──"),
        ("Avg Win  / trade", f"[green]+{bt.avg_win_pct:.4f}%[/]", "—"),
        ("Avg Loss / trade", f"[red]{bt.avg_loss_pct:.4f}%[/]",   "—"),
    ]
    for r in rows: tbl.add_row(*r)

    vc = bt.verdict_color
    console.print(f"\n[{sc}]{spark}[/]  [{vc}]{bt.verdict}[/{vc}]")
    console.print(tbl)
    if bt.issues != "None":
        console.print(f"  [warn]⚠  {bt.issues}[/]")
    console.print()


def permutation_test(p_up: np.ndarray, actuals: np.ndarray,
                     signal_up: float = 0.55, signal_dn: float = 0.45,
                     n_perm: int = 200, metric: str = "sharpe") -> dict:
    # A statistical test. Is the AI actually smart, or just lucky?
    # We shuffle the predictions 200 times randomly and see how often random guessing
    # beats the AI's actual performance. If p_val < 0.05, the AI has a real edge.
    p_up = np.asarray(p_up)
    actuals = np.asarray(actuals)
    def compute_metric(p_up_local):
        mask_long = p_up_local > signal_up
        mask_short = p_up_local < signal_dn
        directions = np.where(mask_long, 1, np.where(mask_short, -1, 0))
        pos_size = BASE_POS_SIZE
        rets = []
        for d, r in zip(directions, actuals):
            if d == 0:
                continue
            rets.append(d * r * pos_size - pos_size * (0.001 + 0.001))
        arr = np.array(rets)
        if arr.size == 0:
            return 0.0
        if metric == "sharpe":
            return float((arr.mean() / (arr.std() + 1e-10)) * math.sqrt(252))
        else:
            return float(arr.sum())

    obs = compute_metric(p_up)
    perms = []
    rng = np.random.default_rng(42)
    for _ in range(max(10, int(n_perm))):
        p_shuf = rng.permutation(p_up) # Shuffle the array randomly
        perms.append(compute_metric(p_shuf))
    perms = np.array(perms)
    mean_p = float(perms.mean())
    std_p = float(perms.std())
    p_val = float(np.mean(perms >= obs)) # Calculate probability
    return {"observed": obs, "perm_mean": mean_p, "perm_std": std_p, "p_value": p_val,
            "n_perm": len(perms)}


def build_summary_table(results: list, bt_results: list) -> Table:
    # Rolls up all the processed tickers into one neat master table.
    tbl = Table(title="LSTM Peak v7.1 — Honest Summary (Test Set)",
                box=box.DOUBLE_EDGE, header_style="bold white on #0d1117",
                border_style="bright_black", show_lines=True, expand=True)
    for name, style, just, mw in [
        ("Ticker",  "bold cyan", "center", 12),
        ("Signal",  "",          "center", 8),
        ("P(UP)",   "",          "center", 7),
        ("TestAcc", "",          "center", 8),
        ("Sharpe",  "",          "center", 8),
        ("Return",  "",          "center", 9),
        ("MDD",     "",          "center", 7),
        ("Trades",  "",          "center", 8),
        ("Costs%",  "",          "center", 9),
        ("Verdict", "",          "center", 14),
    ]:
        tbl.add_column(name, style=style, justify=just, min_width=mw)

    def g(v, t): return "green" if v >= t else "red"

    for r, bt in zip(results, bt_results):
        if r["status"] != "OK" or bt is None:
            tbl.add_row(r["ticker"], *(["-"] * 9))
            continue
        sig  = r["signal"]
        p_up = r["p_up"]
        sc   = "up" if sig=="LONG" else "down" if sig=="SHORT" else "flat"
        tm   = r["test_metrics"]
        tbl.add_row(
            r["ticker"],
            f"[{sc}]{'▲ LONG' if sig=='LONG' else '▼ SHORT' if sig=='SHORT' else '→ FLAT'}[/]",
            f"[{g(p_up*100,55)}]{p_up*100:.1f}%[/]",
            f"[{g(tm['dir_acc'],53)}]{tm['dir_acc']:.1f}%[/]",
            f"[{g(bt.sharpe_active,1.0)}]{bt.sharpe_active:.2f}[/]",
            f"[{g(bt.total_return_pct,0)}]{bt.total_return_pct:.1f}%[/]",
            f"[{'green' if bt.max_drawdown_pct>-15 else 'red'}]{bt.max_drawdown_pct:.1f}%[/]",
            str(bt.n_trades),
            f"[yellow]-{bt.total_drag_pct:.1f}%[/]",
            f"[{bt.verdict_color}]{bt.verdict}[/]",
        )
    return tbl


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    # The brain of the script. This fires off when you run the python file.
    console.rule("[bold cyan]LSTM Stock Predictor — PEAK EDITION v7.1 (Actually Tradable)[/]")
    console.print(
        "\n[info]v7.1 FIXES: Swing trading (hold=3) · High-conviction thresholding (top 20%) · "
        "Strict Take-Profit (6%) and Stop-Loss (2.5%)[/]"
    )
    console.print(
        f"[info]Hardware: [bold white]{DEVICE.upper()}[/][info] | "
        f"PyTorch {torch.__version__}[/]\n"
    )

    # Grabbing user inputs
    raw_input = console.input("[bold white]Tickers (comma-separated): [/]").strip()
    tickers   = [t.upper().strip() for t in raw_input.split(",") if t.strip()]
    if not tickers:
        console.print("[bold red]No tickers entered.[/]"); return

    period = console.input(
        "[bold white]History period (e.g. 5y 10y 12y max, Enter=12y): [/]"
    ).strip() or "12y"

    ic_str = console.input(
        "[bold white]Paper capital (Enter=₹1,00,000): [/]"
    ).strip()
    try:
        initial_capital = float(
            ic_str.replace(",","").replace("₹","").replace("$",""))
    except Exception:
        initial_capital = 100_000.0

    results, bt_results = [], []
    regime_cache: dict  = {}

    # Loops through every stock you asked for
    for ticker in tickers:
        console.rule(f"[bold yellow]{ticker}[/]")
        currency = get_currency(ticker)

        # Spins up a progress bar so you know it didn't crash while thinking
        with Progress(SpinnerColumn(),
                      TextColumn("[progress.description]{task.description}"),
                      BarColumn(bar_width=26), TimeElapsedColumn(),
                      console=console, transient=True) as prog:
            task = prog.add_task("", total=4)

            # Step 1. Fetch & prepare data
            prog.update(task, description=f"  Fetching {ticker} ({period})...")
            try:
                data = fetch_and_prepare(ticker, period=period)
            except Exception as e:
                console.print(f"  [bad]✗ {ticker}: {e}[/]")
                results.append({"ticker": ticker, "status": str(e)[:80]})
                bt_results.append(None); continue
            prog.advance(task)

            # Step 2. Regime index
            prog.update(task, description="  Fetching regime index...")
            if ticker not in regime_cache:
                regime_cache[ticker] = fetch_regime_series(ticker, period=period)
            regime_series = regime_cache[ticker]
            reg_name   = regime_series.attrs.get("regime_name",
                         "NIFTY 50" if is_indian(ticker) else "S&P 500")
            reg_symbol = regime_series.attrs.get("regime_symbol", "^NSEI")
            console.print(
                f"  [info]Regime: {reg_name} ({reg_symbol})[/]"
            )
            prog.advance(task)

            # Step 3. Train the neural net
            prog.update(task, description="  Training LSTM v7.1...")
            try:
                model, val_loss, tr_m, vl_m, ts_m = train_model(data)
            except Exception as e:
                console.print(f"  [bad]✗ Training failed: {e}[/]")
                results.append({"ticker": ticker, "status": "Train error"})
                bt_results.append(None); continue
            prog.advance(task)

            # Calculate adaptive thresholds (the top/bottom 10% bounds discussed earlier)
            signal_up, signal_dn, val_tr = compute_adaptive_thresholds(
                vl_m["p_up_raw"], target_trade_frac=0.20)
            console.print(
                f"  [info]Thresholds: LONG>{signal_up:.3f} "
                f"SHORT<{signal_dn:.3f} | Val trade rate={val_tr*100:.1f}%[/]"
            )

            # Step 4. Check today's signal + run historical backtest
            prog.update(task, description="  Inference + backtest...")
            p_up_today, p_std_today = predict_today(model, data["last_window"])

            signal = ("LONG"  if p_up_today > signal_up  else
                      "SHORT" if p_up_today < signal_dn  else "FLAT")
            conf_val   = abs(p_up_today - 0.5) * 200
            conf_label = "HIGH" if conf_val >= 20 else "MED" if conf_val >= 10 else "LOW"

            ts       = data["ts"]
            test_pu  = ts_m["p_up_raw"]
            test_act = ts_m["actuals_raw"]
            test_dt  = ts["dates"]
            test_cl  = ts["closes"]
            min_len  = min(len(test_pu), len(test_act), len(test_dt), len(test_cl))

            bt = run_backtest(
                ticker=ticker,
                p_up_arr=test_pu[:min_len],
                actuals=test_act[:min_len],
                dates=test_dt[:min_len],
                close_prices=test_cl[:min_len],
                regime_series=regime_series,
                signal_up=signal_up, signal_dn=signal_dn,
                initial_capital=initial_capital,
                test_dir_acc=ts_m["dir_acc"],
                n_test_samples=ts_m["n"],
            )
            bt_results.append(bt)
            prog.advance(task)

        # Print outputs to console
        p_col = "up" if signal=="LONG" else "down" if signal=="SHORT" else "flat"
        console.print(
            f"  [info]Last close:[/] [white]{currency}{data['last_close']:.2f}[/]   "
            f"[info]P(UP):[/] [{p_col}]{p_up_today*100:.1f}%[/{p_col}]"
            f" ±{p_std_today*100:.1f}%   "
            f"[info]Signal:[/] [{p_col}]"
            f"{'▲ LONG' if signal=='LONG' else '▼ SHORT' if signal=='SHORT' else '→ FLAT'}"
            f"[/{p_col}]   [info]Conf:[/] {conf_label} ({conf_val:.0f})   "
            f"[info]TEST acc:[/] {ts_m['dir_acc']:.1f}%"
        )
        print_model_report(ticker, tr_m, vl_m, ts_m,
                           data["n_train_samples"],
                           data["n_val_samples"],
                           data["n_test_samples"])
        print_backtest_panel(bt)

        # Log it for the summary table
        results.append({
            "ticker":      ticker,
            "last_close":  data["last_close"],
            "p_up":        p_up_today,
            "p_std":       p_std_today,
            "signal":      signal,
            "confidence":  conf_label,
            "conf_val":    conf_val,
            "train_m":     tr_m,
            "val_m":       vl_m,
            "test_metrics":ts_m,
            "status":      "OK",
        })

    # Render summary table at the very end
    console.print()
    valid = [(r, b) for r, b in zip(results, bt_results)
             if r["status"] == "OK" and b is not None]
    if valid:
        rs, bs = zip(*valid)
        console.print(Panel(build_summary_table(list(rs), list(bs)), padding=(1, 2)))

if __name__ == "__main__":
    main()