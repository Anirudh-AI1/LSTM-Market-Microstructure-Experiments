"""
Sector-aware cross-stock LSTM with walk-forward validation.

Optimized with Volatility-Normalized Targets (ATR-Scaling) to solve signal compression.
Adapts dynamically to high-beta (TSLA) and low-beta (Banks) stocks seamlessly.
"""

from __future__ import annotations

import copy
import math
import random
import re
import warnings
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import yfinance as yf

warnings.filterwarnings("ignore")

INDIAN_SUFFIXES = (".NS", ".BO", ".BSE")
OHLCV_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]

MANUAL_INDUSTRY_PEERS = {
    ("us", "auto-manufacturers"): [
        "GM", "F", "RIVN", "LCID", "STLA", "TM", "HMC", "VWAGY", "BYDDY", "NIO", "LI", "XPEV",
    ],
    ("us", "semiconductors"): [
        "NVDA", "AMD", "AVGO", "QCOM", "MU", "INTC", "TXN", "ADI", "MRVL", "NXPI", "ON", "ASML",
    ],
    ("india", "banks-regional"): [
        "SBIN.NS", "BANKBARODA.NS", "CANBK.NS", "PNB.NS", "UNIONBANK.NS", "INDIANB.NS",
        "BANKINDIA.NS", "MAHABANK.NS", "CENTRALBK.NS", "UCOBANK.NS", "IOB.NS", "PSB.NS",
    ],
    ("india", "aerospace-defense"): [
        "HAL.NS", "BEL.NS", "BDL.NS", "GRSE.NS", "MAZDOCK.NS", "COCHINSHIP.NS",
        "MIDHANI.NS", "DATAPATTNS.NS", "MTARTECH.NS", "PARAS.NS",
    ],
}

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    RICH_AVAILABLE = True
    console = Console()
except Exception:
    RICH_AVAILABLE = False
    console = None


@dataclass
class RunConfig:
    start_date: str = "2000-01-01"
    end_date: Optional[str] = None
    
    # 1. SHORTER MEMORY: 21 Days (1 Trading Month) for sharper momentum reaction
    seq_len: int = 21  
    holding_period_days: int = 5  
    min_price_rows: int = 450
    min_peers: int = 4
    max_peers: int = 12
    train_years: int = 10
    test_years: int = 2
    step_years: int = 1
    expanding_train: bool = False
    desired_folds: int = 8
    min_folds: int = 1
    min_train_years: int = 2 
    
    val_fraction: float = 0.15
    batch_size: int = 128
    hidden_size: int = 48
    dropout: float = 0.30
    learning_rate: float = 1e-3
    max_epochs: int = 30 # Increased slightly for normalized target
    early_stopping_patience: int = 6
    
    # 2. VOLATILITY EDGES: These are now ATR multiples, not strict percentages.
    # 0.5 means the model expects a move equal to 50% of the stock's average daily range over 5 days.
    signal_long_edge: float = 0.5 
    signal_short_edge: float = -0.5
    
    execution_delay_days: int = 1
    allow_short: bool = True
    min_trades_per_year: float = 12.0
    max_active_fraction: float = 0.65
    slippage_pct_us: float = 0.0020
    spread_pct_us: float = 0.0005
    slippage_pct_india: float = 0.0025
    spread_pct_india: float = 0.0010
    include_target_in_train_if_few_peers: bool = True
    random_seed: int = 42


@dataclass
class ResolvedTicker:
    symbol: str
    long_name: str
    exchange: str
    quote_type: str
    market_hint: str


@dataclass
class PeerDiscovery:
    target: ResolvedTicker
    sector: str
    sector_key: str
    industry: str
    industry_key: str
    market: str
    peers: List[str]
    notes: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    used_target_in_fallback: bool = False


@dataclass
class FoldSpec:
    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


@dataclass
class FoldSummary:
    fold_id: int
    train_range: str
    test_range: str
    train_samples: int
    val_samples: int
    test_samples: int
    loss_score: float
    directional_acc: float
    long_rate_pct: float 
    mean_pred_ret: float
    peers_used: int
    policy_mode: str


@dataclass
class BacktestResult:
    total_return_pct: float
    annual_return_pct: float
    sharpe: float
    max_drawdown_pct: float
    win_rate_pct: float
    total_trades: int
    active_days: int
    buy_hold_return_pct: float
    buy_hold_final_equity: float
    alpha_pct: float
    final_equity: float


@dataclass
class SampleBundle:
    X: np.ndarray
    y: np.ndarray
    next_ret: np.ndarray
    dates: np.ndarray
    symbols: np.ndarray


@dataclass
class FoldPlan:
    train_years: int
    test_years: int
    step_years: int
    expanding_train: bool
    total_folds: int
    adapted: bool


@dataclass
class TradingPolicy:
    mode: str
    signal_long_edge: float
    signal_short_edge: float
    allow_short: bool
    validation_score: float


class SimpleLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, dropout: float):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_size)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            dropout=0.0,
            batch_first=True,
        )
        attn_hidden = max(hidden_size // 2, 16)
        self.attn = nn.Sequential(
            nn.Linear(hidden_size, attn_hidden),
            nn.Tanh(),
            nn.Linear(attn_hidden, 1),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, attn_hidden),
            nn.GELU(),
            nn.Linear(attn_hidden, 1), 
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_norm(x)
        out, _ = self.lstm(x)
        attn_scores = self.attn(out).squeeze(-1)
        attn_weights = torch.softmax(attn_scores, dim=1).unsqueeze(-1)
        context = (out * attn_weights).sum(dim=1)
        last = out[:, -1, :]
        combined = 0.7 * context + 0.3 * last
        return self.head(combined).squeeze(-1)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def market_currency_symbol(market: str) -> str:
    return "Rs" if market == "india" else "$"


def format_money(value: float, market: str) -> str:
    return f"{market_currency_symbol(market)} {value:,.2f}"


def print_rule(title: str) -> None:
    if RICH_AVAILABLE:
        console.rule(title)
    else:
        line = "=" * max(20, len(title) + 4)
        print(f"\n{line}\n{title}\n{line}")


def print_note(message: str) -> None:
    if RICH_AVAILABLE:
        console.print(message)
    else:
        print(message)


def is_indian_symbol(symbol: str) -> bool:
    symbol_up = (symbol or "").upper()
    return any(symbol_up.endswith(sfx) for sfx in INDIAN_SUFFIXES)


def base_symbol(symbol: str) -> str:
    up = symbol.upper()
    for suffix in INDIAN_SUFFIXES:
        if up.endswith(suffix):
            return up[: -len(suffix)]
    return up


def same_market(symbol: str, market: str) -> bool:
    if market == "india":
        return is_indian_symbol(symbol)
    return not is_indian_symbol(symbol)


def prefer_market_symbol(symbols: Iterable[str], market: str) -> List[str]:
    deduped: Dict[str, str] = {}
    for symbol in symbols:
        if not symbol:
            continue
        up = str(symbol).upper().strip()
        if market == "india" and not is_indian_symbol(up):
            continue
        if market == "us" and is_indian_symbol(up):
            continue
        key = base_symbol(up)
        if key not in deduped:
            deduped[key] = up
            continue
        existing = deduped[key]
        if market == "india":
            if existing.endswith(".BO") and up.endswith(".NS"):
                deduped[key] = up
    return list(deduped.values())


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    return df


def score_search_quote(quote: dict, query: str) -> int:
    query_up = query.upper().strip()
    symbol = str(quote.get("symbol", "")).upper()
    long_name = str(quote.get("longname", "")).upper()
    short_name = str(quote.get("shortname", "")).upper()
    exchange = str(quote.get("exchange", "")).upper()
    score = 0
    if quote.get("quoteType") == "EQUITY":
        score += 50
    if symbol == query_up:
        score += 100
    if base_symbol(symbol) == base_symbol(query_up):
        score += 30
    if query_up and query_up in symbol:
        score += 20
    if query_up and query_up in short_name:
        score += 15
    if query_up and query_up in long_name:
        score += 10
    if symbol.endswith(".NS"):
        score += 3
    if exchange in {"NMS", "NYQ", "NSI"}:
        score += 2
    return score


def prompt_user_to_choose(options: Sequence[dict], query: str) -> dict:
    print("\nMultiple ticker matches found:")
    for idx, quote in enumerate(options, start=1):
        symbol = quote.get("symbol", "?")
        short_name = quote.get("shortname") or quote.get("longname") or "Unknown"
        exchange = quote.get("exchangeDisp") or quote.get("exchange") or "Unknown"
        quote_type = quote.get("quoteType") or "Unknown"
        print(f"  {idx}. {symbol:<12} {short_name} [{exchange}, {quote_type}]")
    while True:
        raw = input(f"Choose 1-{len(options)} for '{query}' (Enter = 1): ").strip()
        if raw == "":
            return options[0]
        if raw.isdigit():
            choice = int(raw)
            if 1 <= choice <= len(options):
                return options[choice - 1]
        print("Invalid choice. Try again.")


def resolve_ticker(query: str) -> ResolvedTicker:
    cleaned = query.strip()
    if not cleaned:
        raise ValueError("Ticker/company name cannot be blank.")

    search = yf.Search(
        cleaned,
        max_results=12,
        news_count=0,
        lists_count=0,
        include_research=False,
        enable_fuzzy_query=True,
        recommended=12,
        raise_errors=False,
    )
    quotes = [q for q in (search.quotes or []) if q.get("quoteType") == "EQUITY"]
    if not quotes:
        raise ValueError(f"No equity ticker match found for '{cleaned}'.")

    ranked = sorted(quotes, key=lambda q: score_search_quote(q, cleaned), reverse=True)
    top = ranked[0]
    exact_symbol = str(top.get("symbol", "")).upper() == cleaned.upper()
    top_score = score_search_quote(top, cleaned)
    second_score = score_search_quote(ranked[1], cleaned) if len(ranked) > 1 else -999

    if exact_symbol or top_score >= second_score + 20:
        chosen = top
    else:
        chosen = prompt_user_to_choose(ranked[:5], cleaned)

    symbol = str(chosen.get("symbol", "")).upper().strip()
    exchange = str(chosen.get("exchangeDisp") or chosen.get("exchange") or "").strip()
    market_hint = "india" if is_indian_symbol(symbol) or exchange.upper() in {"NSI", "BSE"} else "us"
    long_name = str(chosen.get("longname") or chosen.get("shortname") or symbol).strip()
    return ResolvedTicker(
        symbol=symbol,
        long_name=long_name,
        exchange=exchange or "Unknown",
        quote_type=str(chosen.get("quoteType") or "EQUITY"),
        market_hint=market_hint,
    )


@lru_cache(maxsize=256)
def safe_info(ticker: str) -> dict:
    info = yf.Ticker(ticker).info
    return info or {}


def is_us_otc_exchange(exchange: str) -> bool:
    return exchange.upper() in {"PNK", "OQB", "OTC", "OTCM", "OTCQB", "OTCQX", "OEM"}


def filter_peer_candidates(
    candidates: Sequence[str],
    market: str,
    target_sector: str,
    target_sector_key: str,
    target_industry: str,
    target_industry_key: str,
    config: RunConfig,
) -> List[str]:
    strict: List[str] = []
    loose: List[str] = []

    for symbol in candidates:
        try:
            info = safe_info(symbol)
        except Exception:
            continue

        peer_exchange = str(info.get("exchange", "")).upper()
        peer_sector = str(info.get("sector", "")).strip()
        peer_sector_key = str(info.get("sectorKey", "")).strip()
        peer_industry = str(info.get("industry", "")).strip()
        peer_industry_key = str(info.get("industryKey", "")).strip()

        if market == "india" and not is_indian_symbol(symbol):
            continue
        if market == "us" and is_indian_symbol(symbol):
            continue
        if market == "us" and is_us_otc_exchange(peer_exchange):
            continue

        same_industry = False
        same_sector = False

        if target_industry_key and peer_industry_key:
            same_industry = peer_industry_key == target_industry_key
        elif target_industry and peer_industry:
            same_industry = peer_industry.lower() == target_industry.lower()

        if target_sector_key and peer_sector_key:
            same_sector = peer_sector_key == target_sector_key
        elif target_sector and peer_sector:
            same_sector = peer_sector.lower() == target_sector.lower()

        if same_industry:
            strict.append(symbol)
        elif same_sector:
            loose.append(symbol)

    strict = prefer_market_symbol(strict, market)
    loose = prefer_market_symbol(loose, market)

    if len(strict) >= config.min_peers:
        return strict[: config.max_peers]

    merged = list(dict.fromkeys(strict + loose))
    return merged[: config.max_peers]


def extract_symbols_from_table(table: Optional[pd.DataFrame]) -> List[str]:
    if table is None or table.empty:
        return []

    candidate_cols = ["symbol", "Symbol", "ticker", "Ticker"]
    for col in candidate_cols:
        if col in table.columns:
            values = table[col].dropna().astype(str).tolist()
            return [v.upper().strip() for v in values if v]

    if table.index.dtype == object:
        values = [str(v).upper().strip() for v in table.index.tolist()]
        if sum(1 for v in values if any(ch.isalpha() for ch in v)) >= max(1, len(values) // 2):
            return values
    return []


def parse_screen_symbols(response: object) -> List[str]:
    if response is None:
        return []
    if isinstance(response, dict):
        if isinstance(response.get("quotes"), list):
            return [
                str(item.get("symbol", "")).upper().strip()
                for item in response["quotes"]
                if isinstance(item, dict) and item.get("symbol")
            ]
        finance = response.get("finance")
        if isinstance(finance, dict):
            result = finance.get("result")
            if isinstance(result, list) and result:
                first = result[0]
                if isinstance(first, dict) and isinstance(first.get("quotes"), list):
                    return [
                        str(item.get("symbol", "")).upper().strip()
                        for item in first["quotes"]
                        if isinstance(item, dict) and item.get("symbol")
                    ]
    return []


def infer_market(symbol: str, info: dict, resolved: ResolvedTicker) -> str:
    if is_indian_symbol(symbol):
        return "india"
    country = str(info.get("country", "")).lower()
    exchange = str(info.get("exchange", "")).upper()
    if country == "india" or exchange in {"NSI", "BSE"}:
        return "india"
    return resolved.market_hint


def discover_peers(resolved: ResolvedTicker, config: RunConfig) -> PeerDiscovery:
    info = safe_info(resolved.symbol)
    market = infer_market(resolved.symbol, info, resolved)

    sector = str(info.get("sector") or "").strip()
    sector_key = str(info.get("sectorKey") or "").strip()
    industry = str(info.get("industry") or "").strip()
    industry_key = str(info.get("industryKey") or "").strip()
    if not sector and not industry:
        raise ValueError(
            f"Yahoo Finance did not return sector/industry metadata for {resolved.symbol}."
        )

    industry_candidates: List[str] = []
    sector_candidates: List[str] = []
    screener_candidates: List[str] = []
    notes: List[str] = []
    warns: List[str] = []

    manual_key = (market, industry_key)
    if manual_key in MANUAL_INDUSTRY_PEERS:
        industry_candidates.extend(MANUAL_INDUSTRY_PEERS[manual_key])
        notes.append(f"Manual same-industry peer map for {market}/{industry_key}")

    if industry_key:
        try:
            industry_obj = yf.Industry(industry_key)
            for table_name in ("top_companies", "top_growth_companies", "top_performing_companies"):
                industry_candidates.extend(extract_symbols_from_table(getattr(industry_obj, table_name, None)))
            notes.append(f"Industry peer discovery via yfinance.Industry('{industry_key}')")
        except Exception as exc:
            warns.append(f"Industry discovery failed: {exc}")

    if sector_key:
        try:
            sector_obj = yf.Sector(sector_key)
            sector_candidates.extend(extract_symbols_from_table(getattr(sector_obj, "top_companies", None)))
            notes.append(f"Sector peer discovery via yfinance.Sector('{sector_key}')")
        except Exception as exc:
            warns.append(f"Sector discovery failed: {exc}")

    if sector:
        try:
            region = "in" if market == "india" else "us"
            q = yf.EquityQuery(
                "and",
                [
                    yf.EquityQuery("eq", ["region", region]),
                    yf.EquityQuery("eq", ["sector", sector]),
                ],
            )
            response = yf.screen(
                q,
                size=min(max(config.max_peers * 3, 25), 80),
                sortField="intradaymarketcap",
                sortAsc=False,
            )
            screener_candidates.extend(parse_screen_symbols(response))
            notes.append(f"Screener discovery via sector='{sector}', region='{region}'")
        except Exception as exc:
            warns.append(f"Screener discovery failed: {exc}")

    primary_pool = [
        symbol for symbol in industry_candidates
        if symbol and symbol.upper() != resolved.symbol.upper()
    ]
    fallback_pool = [
        symbol for symbol in sector_candidates + screener_candidates
        if symbol and symbol.upper() != resolved.symbol.upper()
    ]

    primary_pool = prefer_market_symbol(primary_pool, market)
    fallback_pool = prefer_market_symbol(fallback_pool, market)
    primary_peers = filter_peer_candidates(
        candidates=primary_pool,
        market=market,
        target_sector=sector,
        target_sector_key=sector_key,
        target_industry=industry,
        target_industry_key=industry_key,
        config=config,
    )

    peers = primary_peers
    if len(peers) < config.min_peers:
        combined_pool = list(dict.fromkeys(primary_pool + fallback_pool))
        peers = filter_peer_candidates(
            candidates=combined_pool,
            market=market,
            target_sector=sector,
            target_sector_key=sector_key,
            target_industry=industry,
            target_industry_key=industry_key,
            config=config,
        )
        if peers and not primary_peers:
            warns.append(
                "Peer universe needed sector/screener fallback because strict same-industry peers were limited."
            )

    used_target_fallback = False
    if len(peers) < config.min_peers and config.include_target_in_train_if_few_peers:
        used_target_fallback = True
        warns.append(
            f"Only {len(peers)} peers found automatically. The target will be allowed in training fallback."
        )

    return PeerDiscovery(
        target=resolved,
        sector=sector or "Unknown",
        sector_key=sector_key or "Unknown",
        industry=industry or "Unknown",
        industry_key=industry_key or "Unknown",
        market=market,
        peers=peers,
        notes=notes,
        warnings=warns,
        used_target_in_fallback=used_target_fallback,
    )


def download_price_history(symbol: str, config: RunConfig) -> pd.DataFrame:
    df = yf.download(
        symbol,
        start=config.start_date,
        end=config.end_date,
        auto_adjust=True,
        progress=False,
        threads=False,
        timeout=10, 
    )
    if df is None or df.empty:
        raise ValueError(f"No price history returned for {symbol}.")
    df = flatten_columns(df)
    missing = [col for col in OHLCV_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"{symbol} is missing OHLCV columns: {missing}")
    df = df[OHLCV_COLUMNS].copy()
    df = df.ffill().dropna()
    if len(df) < config.min_price_rows:
        raise ValueError(f"{symbol} only has {len(df)} rows. Need at least {config.min_price_rows}.")
    return df


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff().fillna(0.0)
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-8)
    return 100 - (100 / (1 + rs))


def compute_atr_ratio(frame: pd.DataFrame, period: int = 14) -> pd.Series:
    prev_close = frame["Close"].shift(1)
    tr = pd.concat(
        [
            frame["High"] - frame["Low"],
            (frame["High"] - prev_close).abs(),
            (frame["Low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.ewm(alpha=1 / period, adjust=False).mean()
    return atr / (frame["Close"] + 1e-8)


def build_feature_frame(frame: pd.DataFrame, config: RunConfig) -> pd.DataFrame:
    close = frame["Close"]
    open_ = frame["Open"]
    high = frame["High"]
    low = frame["Low"]
    volume = frame["Volume"].replace(0, np.nan).ffill()

    feature_df = pd.DataFrame(index=frame.index)
    feature_df["ret_1d"] = close.pct_change()
    feature_df["ret_5d"] = close.pct_change(5)
    feature_df["ret_20d"] = close.pct_change(20)
    feature_df["overnight_gap"] = open_.div(close.shift(1)).sub(1.0)
    feature_df["intraday_ret"] = close.div(open_).sub(1.0)
    feature_df["range_pct"] = high.div(low).sub(1.0)
    feature_df["dist_sma20"] = close.div(close.rolling(20).mean()).sub(1.0)
    feature_df["dist_sma50"] = close.div(close.rolling(50).mean()).sub(1.0)
    feature_df["vol_20d"] = close.pct_change().rolling(20).std()
    
    atr_ratio = compute_atr_ratio(frame)
    feature_df["atr_ratio"] = atr_ratio
    feature_df["rsi_14"] = compute_rsi(close).div(100.0).sub(0.5)

    log_volume = np.log(volume.clip(lower=1.0))
    vol_mean = log_volume.rolling(20).mean()
    vol_std = log_volume.rolling(20).std()
    feature_df["volume_z20"] = (log_volume - vol_mean) / (vol_std + 1e-8)
    
    # 3. MACRO REGIME: Give the LSTM explicit context since we shortened the sequence length to 21
    feature_df["trend_200"] = close.div(close.rolling(200).mean()).sub(1.0)

    # Backtest target
    feature_df["next_ret"] = close.shift(-1).div(close).sub(1.0)
    
    # THE SECRET WEAPON: VOLATILITY-NORMALIZED TARGET
    # Instead of predicting "1.5%", it predicts "0.8 ATR".
    # This standardizes TSLA and a regional bank so they learn on the exact same scale.
    raw_5_day_return = close.shift(-config.holding_period_days).div(close).sub(1.0)
    feature_df["target_ret"] = raw_5_day_return / (atr_ratio + 1e-8)

    feature_df = feature_df.replace([np.inf, -np.inf], np.nan).dropna()
    return feature_df


def fit_robust_stats(frame: pd.DataFrame, feature_cols: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
    if frame.empty:
        raise ValueError("Training frame is empty, cannot fit scaler.")
    values = frame.loc[:, feature_cols].to_numpy(dtype=np.float32)
    med = np.nanmedian(values, axis=0)
    q75 = np.nanpercentile(values, 75, axis=0)
    q25 = np.nanpercentile(values, 25, axis=0)
    iqr = np.maximum(q75 - q25, 1e-8)
    return med.astype(np.float32), iqr.astype(np.float32)


def apply_robust_stats(frame: pd.DataFrame, feature_cols: Sequence[str], med: np.ndarray, iqr: np.ndarray) -> pd.DataFrame:
    out = frame.copy()
    values = out.loc[:, feature_cols].to_numpy(dtype=np.float32)
    scaled = (values - med) / (iqr + 1e-8)
    out.loc[:, feature_cols] = np.clip(np.nan_to_num(scaled, nan=0.0), -5.0, 5.0)
    return out


def build_sequences(frame: pd.DataFrame, feature_cols: Sequence[str], seq_len: int, symbol: str) -> SampleBundle:
    features = frame.loc[:, feature_cols].to_numpy(dtype=np.float32)
    target_ret = frame["target_ret"].to_numpy(dtype=np.float32)
    next_ret = frame["next_ret"].to_numpy(dtype=np.float32)
    dates = frame.index.to_numpy()

    X_list: List[np.ndarray] = []
    y_list: List[float] = []
    ret_list: List[float] = []
    date_list: List[np.datetime64] = []
    symbol_list: List[str] = []

    for idx in range(seq_len - 1, len(frame)):
        seq = features[idx - seq_len + 1 : idx + 1]
        
        y_val = float(target_ret[idx])
        ret = float(next_ret[idx])
        
        X_list.append(seq)
        y_list.append(y_val)
        ret_list.append(ret)
        date_list.append(dates[idx])
        symbol_list.append(symbol)

    if not X_list:
        return SampleBundle(
            X=np.empty((0, seq_len, len(feature_cols)), dtype=np.float32),
            y=np.empty((0,), dtype=np.float32),
            next_ret=np.empty((0,), dtype=np.float32),
            dates=np.empty((0,), dtype="datetime64[ns]"),
            symbols=np.empty((0,), dtype=object),
        )

    return SampleBundle(
        X=np.asarray(X_list, dtype=np.float32),
        y=np.asarray(y_list, dtype=np.float32),
        next_ret=np.asarray(ret_list, dtype=np.float32),
        dates=np.asarray(date_list),
        symbols=np.asarray(symbol_list, dtype=object),
    )


def subset_bundle(bundle: SampleBundle, mask: np.ndarray) -> SampleBundle:
    return SampleBundle(
        X=bundle.X[mask],
        y=bundle.y[mask],
        next_ret=bundle.next_ret[mask],
        dates=bundle.dates[mask],
        symbols=bundle.symbols[mask],
    )


def concat_bundles(bundles: Sequence[SampleBundle], seq_len: int, n_features: int) -> SampleBundle:
    non_empty = [b for b in bundles if len(b.y) > 0]
    if not non_empty:
        return SampleBundle(
            X=np.empty((0, seq_len, n_features), dtype=np.float32),
            y=np.empty((0,), dtype=np.float32),
            next_ret=np.empty((0,), dtype=np.float32),
            dates=np.empty((0,), dtype="datetime64[ns]"),
            symbols=np.empty((0,), dtype=object),
        )
    return SampleBundle(
        X=np.concatenate([b.X for b in non_empty], axis=0),
        y=np.concatenate([b.y for b in non_empty], axis=0),
        next_ret=np.concatenate([b.next_ret for b in non_empty], axis=0),
        dates=np.concatenate([b.dates for b in non_empty], axis=0),
        symbols=np.concatenate([b.symbols for b in non_empty], axis=0),
    )


def sort_bundle_by_date(bundle: SampleBundle) -> SampleBundle:
    if len(bundle.y) == 0:
        return bundle
    order = np.argsort(bundle.dates.astype("datetime64[ns]"))
    return subset_bundle(bundle, order)


def make_train_val_split(bundle: SampleBundle, val_fraction: float) -> Tuple[SampleBundle, SampleBundle]:
    if len(bundle.y) < 20:
        return bundle, SampleBundle(
            X=np.empty((0, bundle.X.shape[1], bundle.X.shape[2]), dtype=np.float32),
            y=np.empty((0,), dtype=np.float32),
            next_ret=np.empty((0,), dtype=np.float32),
            dates=np.empty((0,), dtype="datetime64[ns]"),
            symbols=np.empty((0,), dtype=object),
        )
    bundle = sort_bundle_by_date(bundle)
    split_idx = max(1, int(len(bundle.y) * (1.0 - val_fraction)))
    split_idx = min(split_idx, len(bundle.y) - 1)
    train_mask = np.arange(len(bundle.y)) < split_idx
    val_mask = ~train_mask
    return subset_bundle(bundle, train_mask), subset_bundle(bundle, val_mask)


def build_data_loader(bundle: SampleBundle, batch_size: int, shuffle: bool) -> DataLoader:
    ds = TensorDataset(
        torch.tensor(bundle.X, dtype=torch.float32),
        torch.tensor(bundle.y, dtype=torch.float32),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def train_model(
    train_bundle: SampleBundle,
    val_bundle: SampleBundle,
    config: RunConfig,
    device: torch.device,
) -> SimpleLSTM:
    import random as _random
    _seed = getattr(config, "random_seed", 42)
    torch.manual_seed(_seed)
    np.random.seed(_seed)
    _random.seed(_seed)
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass
    
    input_size = train_bundle.X.shape[-1]
    model = SimpleLSTM(input_size=input_size, hidden_size=config.hidden_size, dropout=config.dropout).to(device)

    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=1e-5)

    train_loader = build_data_loader(train_bundle, config.batch_size, shuffle=True)
    val_loader = build_data_loader(val_bundle, config.batch_size, shuffle=False)

    best_state = copy.deepcopy(model.state_dict())
    best_val_loss = float("inf")
    patience_left = config.early_stopping_patience

    for _epoch in range(config.max_epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        if len(val_bundle.y) == 0:
            best_state = copy.deepcopy(model.state_dict())
            continue

        model.eval()
        val_losses: List[float] = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                val_losses.append(float(criterion(logits, yb).item()))
        mean_val_loss = float(np.mean(val_losses)) if val_losses else float("inf")

        if mean_val_loss + 1e-6 < best_val_loss:
            best_val_loss = mean_val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_left = config.early_stopping_patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    model.load_state_dict(best_state)
    model.eval()
    return model


def predict_returns(model: SimpleLSTM, X: np.ndarray, device: torch.device) -> np.ndarray:
    if len(X) == 0:
        return np.empty((0,), dtype=np.float32)
    preds: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(X), 512):
            xb = torch.tensor(X[start : start + 512], dtype=torch.float32, device=device)
            logits = model(xb)
            preds.append(logits.cpu().numpy())
    return np.concatenate(preds).astype(np.float32)


def directional_accuracy_pct(preds: np.ndarray, y_true: np.ndarray) -> float:
    if len(y_true) == 0:
        return float("nan")
    correct = (np.sign(preds) == np.sign(y_true)).astype(np.float32)
    return float(np.mean(correct) * 100.0)


def _generate_fold_specs(
    first_date: pd.Timestamp,
    last_date: pd.Timestamp,
    train_years: int,
    test_years: int,
    step_years: int,
    expanding_train: bool,
) -> List[FoldSpec]:
    folds: List[FoldSpec] = []
    anchor = first_date
    fold_id = 1
    while True:
        train_start = first_date if expanding_train else anchor
        train_end = anchor + pd.DateOffset(years=train_years) - pd.Timedelta(days=1)
        test_start = train_end + pd.Timedelta(days=1)
        test_end = test_start + pd.DateOffset(years=test_years) - pd.Timedelta(days=1)
        if test_start >= last_date:
            break
        if test_end > last_date:
            test_end = last_date
        folds.append(
            FoldSpec(
                fold_id=fold_id,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
            )
        )
        fold_id += 1
        anchor = anchor + pd.DateOffset(years=step_years)
        if anchor >= last_date:
            break
    return folds


def make_fold_specs(target_index: pd.DatetimeIndex, config: RunConfig) -> Tuple[List[FoldSpec], FoldPlan]:
    first_date = max(pd.Timestamp(config.start_date), pd.Timestamp(target_index.min()))
    last_date = pd.Timestamp(target_index.max())

    candidate_settings: List[Tuple[int, int, int, bool, bool]] = []
    
    candidate_settings.append((config.train_years, config.test_years, config.step_years, config.expanding_train, False))
    
    for train_years in range(config.train_years - 1, config.min_train_years - 1, -1):
        candidate_settings.append((train_years, config.test_years, config.step_years, config.expanding_train, True))
        
    for test_years in range(config.test_years, 0, -1):
        for train_years in range(config.train_years, config.min_train_years - 1, -1):
            candidate_settings.append((train_years, test_years, config.step_years, config.expanding_train, True))

    seen_settings = set()
    for train_years, test_years, step_years, expanding_train, adapted in candidate_settings:
        key = (train_years, test_years, step_years, expanding_train)
        if key in seen_settings:
            continue
        seen_settings.add(key)

        folds = _generate_fold_specs(
            first_date=first_date,
            last_date=last_date,
            train_years=train_years,
            test_years=test_years,
            step_years=step_years,
            expanding_train=expanding_train,
        )
        
        required_folds = config.min_folds if not adapted else 1
        if len(folds) < required_folds:
            continue

        if len(folds) > config.desired_folds:
            folds = folds[-config.desired_folds :]
            for idx, fold in enumerate(folds, start=1):
                fold.fold_id = idx

        plan = FoldPlan(
            train_years=train_years,
            test_years=test_years,
            step_years=step_years,
            expanding_train=expanding_train,
            total_folds=len(folds),
            adapted=adapted,
        )
        return folds, plan

    return [], FoldPlan(
        train_years=config.train_years,
        test_years=config.test_years,
        step_years=config.step_years,
        expanding_train=config.expanding_train,
        total_folds=0,
        adapted=True,
    )


def build_fold_datasets(
    fold: FoldSpec,
    feature_frames: Dict[str, pd.DataFrame],
    feature_cols: Sequence[str],
    target_symbol: str,
    peer_symbols: Sequence[str],
    config: RunConfig,
) -> Tuple[SampleBundle, SampleBundle, SampleBundle, SampleBundle]:
    train_symbols = list(peer_symbols)
    if len(train_symbols) < config.min_peers and config.include_target_in_train_if_few_peers:
        train_symbols = list(dict.fromkeys(list(train_symbols) + [target_symbol]))

    train_bundles: List[SampleBundle] = []
    val_bundles: List[SampleBundle] = []

    train_start_dt = pd.Timestamp(fold.train_start)
    train_end_dt = pd.Timestamp(fold.train_end)

    for symbol in train_symbols:
        frame = feature_frames[symbol]
        train_frame = frame.loc[train_start_dt:train_end_dt]
        
        if len(train_frame) < config.seq_len:
            continue
        
        med_sym, iqr_sym = fit_robust_stats(train_frame, feature_cols)
        
        scaled_frame = apply_robust_stats(frame, feature_cols, med_sym, iqr_sym)
        bundle_scaled = build_sequences(scaled_frame, feature_cols, config.seq_len, symbol)
        
        train_mask = (
            (bundle_scaled.dates >= np.datetime64(fold.train_start))
            & (bundle_scaled.dates <= np.datetime64(fold.train_end))
        )
        bundle_train_full = subset_bundle(bundle_scaled, train_mask)
        
        if len(bundle_train_full.y) > 0:
            t_bundle, v_bundle = make_train_val_split(bundle_train_full, config.val_fraction)
            if len(t_bundle.y) > 0: train_bundles.append(t_bundle)
            if len(v_bundle.y) > 0: val_bundles.append(v_bundle)

    train_all = concat_bundles(train_bundles, config.seq_len, len(feature_cols))
    val_all = concat_bundles(val_bundles, config.seq_len, len(feature_cols))

    target_frame = feature_frames[target_symbol]
    target_train_frame = target_frame.loc[train_start_dt:train_end_dt]
    
    if len(target_train_frame) < config.seq_len:
        raise ValueError(f"Target has insufficient training data in fold {fold.fold_id}.")
        
    med_t, iqr_t = fit_robust_stats(target_train_frame, feature_cols)
    target_scaled_frame = apply_robust_stats(target_frame, feature_cols, med_t, iqr_t)
    target_bundle = build_sequences(target_scaled_frame, feature_cols, config.seq_len, target_symbol)

    target_train_mask = (
        (target_bundle.dates >= np.datetime64(fold.train_start))
        & (target_bundle.dates <= np.datetime64(fold.train_end))
    )
    target_train = subset_bundle(target_bundle, target_train_mask)
    
    if len(target_train.y) >= 40:
        _, target_val_bundle = make_train_val_split(target_train, config.val_fraction)
    else:
        target_val_bundle = val_all

    test_mask = (
        (target_bundle.dates >= np.datetime64(fold.test_start))
        & (target_bundle.dates <= np.datetime64(fold.test_end))
    )
    test_bundle = subset_bundle(target_bundle, test_mask)
    
    return train_all, val_all, target_val_bundle, test_bundle


def run_backtest(
    pred_ret: np.ndarray,
    next_ret: np.ndarray,
    dates: np.ndarray,
    market: str,
    config: RunConfig,
    initial_equity: float = 100_000.0,
    signal_long_edge: Optional[float] = None,
    signal_short_edge: Optional[float] = None,
    allow_short: Optional[bool] = None,
) -> BacktestResult:
    pred_ret = np.asarray(pred_ret, dtype=np.float64)
    next_ret = np.asarray(next_ret, dtype=np.float64)
    if len(pred_ret) == 0:
        raise ValueError("No predictions available for backtest.")

    if signal_long_edge is None: signal_long_edge = config.signal_long_edge
    if signal_short_edge is None: signal_short_edge = config.signal_short_edge
    if allow_short is None: allow_short = config.allow_short

    swing_pos = np.zeros(len(pred_ret), dtype=np.float32)
    current_pos = 0.0
    hold_days = 0
    target_holding_days = max(int(config.holding_period_days), 1)

    for i, p_now in enumerate(pred_ret):
        signal = 0.0
        # The threshold is an ATR multiple. So if p_now >= 0.5, we enter.
        if p_now >= signal_long_edge:
            signal = 1.0
        elif allow_short and p_now <= signal_short_edge:
            signal = -1.0

        if current_pos == 0.0:
            if signal != 0.0:
                current_pos = signal
                hold_days = 1
        else:
            hold_days += 1
            exit_now = False

            if hold_days > target_holding_days:
                exit_now = True

            if exit_now:
                if signal == -current_pos and signal != 0.0:
                    current_pos = signal
                    hold_days = 1
                else:
                    current_pos = 0.0
                    hold_days = 0

        swing_pos[i] = current_pos

    delay = max(int(config.execution_delay_days), 0)
    pos = np.roll(swing_pos, delay)
    if delay > 0:
        pos[:delay] = 0.0

    prev_pos = np.roll(pos, 1)
    prev_pos[0] = 0.0
    turnover = np.abs(pos - prev_pos)

    if market == "india":
        per_side_cost = config.slippage_pct_india + (config.spread_pct_india / 2.0)
    else:
        per_side_cost = config.slippage_pct_us + (config.spread_pct_us / 2.0)

    costs = turnover * per_side_cost
    strat_ret = (pos * next_ret) - costs

    equity_curve = np.empty(len(strat_ret) + 1, dtype=np.float64)
    equity_curve[0] = initial_equity
    for i, daily_ret in enumerate(strat_ret, start=1):
        equity_curve[i] = equity_curve[i - 1] * (1.0 + float(daily_ret))

    bh_curve = np.empty(len(next_ret) + 1, dtype=np.float64)
    bh_curve[0] = initial_equity
    for i, daily_ret in enumerate(next_ret, start=1):
        bh_curve[i] = bh_curve[i - 1] * (1.0 + float(daily_ret))

    years = max(len(strat_ret) / 252.0, 1e-6)
    total_return = (equity_curve[-1] / initial_equity - 1.0) * 100.0
    annual_return = ((equity_curve[-1] / initial_equity) ** (1.0 / years) - 1.0) * 100.0
    buy_hold_return = (bh_curve[-1] / initial_equity - 1.0) * 100.0

    if np.std(strat_ret) > 1e-10:
        sharpe = float(np.mean(strat_ret) / np.std(strat_ret) * math.sqrt(252))
    else:
        sharpe = 0.0

    peak = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve / (peak + 1e-8)) - 1.0
    max_dd = float(drawdown.min() * 100.0)

    active_mask = pos != 0
    active_days = int(active_mask.sum())
    trade_starts = int(np.sum((pos != 0) & (prev_pos == 0)))
    active_rets = strat_ret[active_mask]
    win_rate = float(np.mean(active_rets > 0) * 100.0) if len(active_rets) else 0.0

    return BacktestResult(
        total_return_pct=round(total_return, 2),
        annual_return_pct=round(annual_return, 2),
        sharpe=round(sharpe, 3),
        max_drawdown_pct=round(max_dd, 2),
        win_rate_pct=round(win_rate, 2),
        total_trades=trade_starts,
        active_days=active_days,
        buy_hold_return_pct=round(buy_hold_return, 2),
        buy_hold_final_equity=round(float(bh_curve[-1]), 2),
        alpha_pct=round(total_return - buy_hold_return, 2),
        final_equity=round(float(equity_curve[-1]), 2),
    )


def select_trading_policy(
    pred_ret: np.ndarray,
    next_ret: np.ndarray,
    market: str,
    config: RunConfig,
) -> TradingPolicy:
    if len(pred_ret) == 0:
        return TradingPolicy(
            mode="long-only", signal_long_edge=0.5, signal_short_edge=-0.5, allow_short=False, validation_score=-1e18
        )

    # Grid search ATR multiples (0.3x to 0.9x ATR expectation)
    long_thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    best_policy = TradingPolicy(mode="long-only", signal_long_edge=0.5, signal_short_edge=-0.5, allow_short=False, validation_score=-1e18)
    best_score = -1e18

    for l_thr in long_thresholds:
        bt = run_backtest(
            pred_ret=pred_ret,
            next_ret=next_ret,
            dates=np.arange(len(pred_ret)),
            market=market,
            config=config,
            signal_long_edge=l_thr,
            signal_short_edge=-0.5,
            allow_short=False,
        )
        
        years = max(len(pred_ret) / 252.0, 0.25)
        trades_per_year = bt.total_trades / years
        
        min_trade_penalty = max(0.0, config.min_trades_per_year - trades_per_year) * 10.0
        drawdown_penalty = max(0.0, abs(bt.max_drawdown_pct) - 15.0) * 2.0
        
        score = bt.total_return_pct + (bt.sharpe * 20.0) - min_trade_penalty - drawdown_penalty
        
        if score > best_score:
            best_score = score
            best_policy = TradingPolicy(
                mode="long-only", signal_long_edge=l_thr, signal_short_edge=-0.5, allow_short=False, validation_score=score
            )

    if config.allow_short:
        short_thresholds = [-0.3, -0.4, -0.5, -0.6, -0.7, -0.8, -0.9]
        for s_thr in short_thresholds:
            for l_thr in long_thresholds:
                bt = run_backtest(
                    pred_ret=pred_ret,
                    next_ret=next_ret,
                    dates=np.arange(len(pred_ret)),
                    market=market,
                    config=config,
                    signal_long_edge=l_thr,
                    signal_short_edge=s_thr,
                    allow_short=True,
                )
                
                years = max(len(pred_ret) / 252.0, 0.25)
                trades_per_year = bt.total_trades / years
                
                min_trade_penalty = max(0.0, config.min_trades_per_year - trades_per_year) * 10.0
                drawdown_penalty = max(0.0, abs(bt.max_drawdown_pct) - 15.0) * 2.0
                
                score = bt.total_return_pct + (bt.sharpe * 20.0) - min_trade_penalty - drawdown_penalty
                
                if score > best_score:
                    best_score = score
                    best_policy = TradingPolicy(
                        mode="long-short", signal_long_edge=l_thr, signal_short_edge=s_thr, allow_short=True, validation_score=score
                    )
                    
    return best_policy


def print_run_header(config: RunConfig) -> None:
    print_rule("Magnitude Execution LSTM Walk-Forward (Volatility Normalized)")
    lines = [
        f"Sequence memory: {config.seq_len} days | Prediction Horizon: {config.holding_period_days} Days",
        "Target Label: 5-Day Forward Return normalized by Average True Range (ATR).",
        f"Execution Hurdle: Dynamic ATR Multiple (e.g., > {config.signal_long_edge} ATR move required).",
        "This equalizes TSLA and banks, removing the signal compression trap.",
    ]
    if RICH_AVAILABLE:
        console.print(Panel("\n".join(lines), border_style="cyan"))
    else:
        for line in lines:
            print(line)


def print_walk_forward_explainer(folds: Sequence[FoldSpec], plan: FoldPlan) -> None:
    if not folds:
        return
    example = folds[0]
    mode = "expanding" if plan.expanding_train else "rolling"
    text = (
        f"Walk-forward means we train on old data and test only on future unseen data.\n"
        f"Example fold 1: train on {example.train_start.date()} to {example.train_end.date()}, "
        f"then test on {example.test_start.date()} to {example.test_end.date()}.\n"
        f"Current mode: {mode} walk-forward with {plan.train_years}y train / {plan.test_years}y test / "
        f"{plan.step_years}y step across {plan.total_folds} folds."
    )
    if plan.adapted:
        text += "\nNote: the default fold setup was shortened automatically because the stock had less history."
    if RICH_AVAILABLE:
        console.print(Panel(text, title="What 'Fold' Means", border_style="yellow"))
    else:
        print("\nWhat 'Fold' Means")
        print(text)


def print_discovery_summary(discovery: PeerDiscovery) -> None:
    if RICH_AVAILABLE:
        stock_table = Table(show_header=False, box=None, pad_edge=False)
        stock_table.add_column("Label", style="bold cyan")
        stock_table.add_column("Value", style="white")
        stock_table.add_row("Symbol", discovery.target.symbol)
        stock_table.add_row("Name", discovery.target.long_name)
        stock_table.add_row("Exchange", discovery.target.exchange)
        stock_table.add_row("Market", discovery.market.upper())
        stock_table.add_row("Sector", discovery.sector)
        stock_table.add_row("Industry", discovery.industry)
        console.print(Panel(stock_table, title="Resolved Stock", border_style="green"))

        peer_text = ", ".join(discovery.peers) if discovery.peers else "No peers discovered automatically"
        extra_lines = [f"Peers ({len(discovery.peers)}): {peer_text}"]
        extra_lines.extend(f"Source: {note}" for note in discovery.notes)
        extra_lines.extend(f"Warning: {warning}" for warning in discovery.warnings)
        console.print(Panel("\n".join(extra_lines), title="Peer Universe", border_style="magenta"))
        return

    print("\nResolved stock")
    print(f"  Symbol   : {discovery.target.symbol}")
    print(f"  Name     : {discovery.target.long_name}")
    print(f"  Exchange : {discovery.target.exchange}")
    print(f"  Market   : {discovery.market}")
    print(f"  Sector   : {discovery.sector}")
    print(f"  Industry : {discovery.industry}")


def print_fold_table(summaries: Sequence[FoldSummary]) -> None:
    if not summaries:
        print("\nNo valid walk-forward folds were produced.")
        return
    if RICH_AVAILABLE:
        table = Table(title="Walk-Forward Fold Summary", border_style="bright_black")
        table.add_column("Fold", justify="right", style="bold cyan")
        table.add_column("Training Window", style="white")
        table.add_column("Testing Window", style="white")
        table.add_column("Train", justify="right")
        table.add_column("Test", justify="right")
        table.add_column("Loss", justify="right")
        table.add_column("Dir Acc", justify="right")
        table.add_column("Policy", justify="right")
        for item in summaries:
            table.add_row(
                str(item.fold_id),
                item.train_range,
                item.test_range,
                f"{item.train_samples:,}",
                f"{item.test_samples:,}",
                f"{item.loss_score:.4f}",
                f"{item.directional_acc:.2f}%",
                item.policy_mode,
            )
        console.print(table)
        return

    print("\nWalk-forward folds")
    header = (
        f"{'Fold':<6}{'Train Range':<24}{'Test Range':<24}"
        f"{'Train':>10}{'Test':>8}{'Loss':>9}{'DirAcc':>9}{'Mode':>12}"
    )
    print(header)
    print("-" * len(header))
    for item in summaries:
        print(
            f"{item.fold_id:<6}{item.train_range:<24}{item.test_range:<24}"
            f"{item.train_samples:>10}{item.test_samples:>8}"
            f"{item.loss_score:>9.4f}{item.directional_acc:>8.2f}%{item.policy_mode:>12}"
        )


def print_reality_checks(mean_acc: float, bt: BacktestResult, summaries: Sequence[FoldSummary]) -> None:
    lines = [
        f"Mean directional accuracy: {mean_acc:.2f}%",
        f"Backtest Sharpe: {bt.sharpe:.3f}",
    ]
    if mean_acc > 56.0:
        lines.append("Warning: Accuracy is high for magnitude prediction. Double check validation leakage.")
    elif mean_acc < 48.0:
        lines.append("Warning: Directional accuracy is extremely low. Model is struggling with sign.")
    else:
        lines.append("Info: Directional accuracy is inside the expected reality band for regression.")
    if bt.sharpe > 2.0:
        lines.append("Warning: Sharpe > 2 on this setup is suspicious. Audit assumptions carefully.")
    if bt.alpha_pct < 0:
        lines.append(
            f"Warning: Strategy underperformed buy & hold by {abs(bt.alpha_pct):.2f}%. "
            "That means this model did not add value on this stock."
        )

    if RICH_AVAILABLE:
        console.print(Panel("\n".join(lines), title="Reality Check", border_style="yellow"))
    else:
        print("\nReality check")
        for line in lines:
            print(f"  {line}")


def print_policy_summary(policy: TradingPolicy) -> None:
    lines = [
        f"Mode: {policy.mode}",
        f"Long execution edge: > {policy.signal_long_edge:.2f} ATR",
        f"Short execution edge: < {policy.signal_short_edge:.2f} ATR",
        f"Validation policy score: {policy.validation_score:.2f}",
    ]
    if RICH_AVAILABLE:
        console.print(Panel("\n".join(lines), title="Selected Trading Policy", border_style="cyan"))
    else:
        print("\nSelected trading policy")
        for line in lines:
            print(f"  {line}")


def print_backtest(bt: BacktestResult, initial_capital: float, config: RunConfig, market: str) -> None:
    if market == "india":
        slippage = config.slippage_pct_india * 100.0
        spread = config.spread_pct_india * 100.0
    else:
        slippage = config.slippage_pct_us * 100.0
        spread = config.spread_pct_us * 100.0

    if RICH_AVAILABLE:
        table = Table(title="Brutal Backtest Summary", border_style="bright_black")
        table.add_column("Metric", style="bold cyan")
        table.add_column("Strategy", justify="right", style="white")
        table.add_column("Buy & Hold", justify="right", style="green")
        table.add_row("Capital deployed", format_money(initial_capital, market), format_money(initial_capital, market))
        table.add_row("Final capital", format_money(bt.final_equity, market), format_money(bt.buy_hold_final_equity, market))
        table.add_row("Total return", f"{bt.total_return_pct:.2f}%", f"{bt.buy_hold_return_pct:.2f}%")
        table.add_row("Annualized return", f"{bt.annual_return_pct:.2f}%", "-")
        table.add_row("Sharpe", f"{bt.sharpe:.3f}", "-")
        table.add_row("Max drawdown", f"{bt.max_drawdown_pct:.2f}%", "-")
        table.add_row("Win rate", f"{bt.win_rate_pct:.2f}%", "-")
        table.add_row("Trades", f"{bt.total_trades}", "-")
        table.add_row("Active days", f"{bt.active_days}", "-")
        table.add_row("Alpha vs buy & hold", f"{bt.alpha_pct:.2f}%", "-")
        table.add_row(
            "Execution assumptions",
            f"slip {slippage:.2f}% | spread {spread:.2f}% | delay {config.execution_delay_days}d | "
            f"fixed hold {config.holding_period_days}d",
            "-",
        )
        console.print(table)
        return

    print("\nBrutal backtest")
    print(f"  Capital deployed   : {format_money(initial_capital, market)}")
    print(f"  Final capital      : {format_money(bt.final_equity, market)}")
    print(f"  Buy & hold capital : {format_money(bt.buy_hold_final_equity, market)}")
    print(f"  Total return       : {bt.total_return_pct:.2f}%")
    print(f"  Buy & hold return  : {bt.buy_hold_return_pct:.2f}%")
    print(f"  Annualized return  : {bt.annual_return_pct:.2f}%")
    print(f"  Sharpe             : {bt.sharpe:.3f}")
    print(f"  Max drawdown       : {bt.max_drawdown_pct:.2f}%")
    print(f"  Win rate           : {bt.win_rate_pct:.2f}%")
    print(f"  Trades             : {bt.total_trades}")
    print(f"  Active days        : {bt.active_days}")
    print(f"  Alpha vs buy/hold  : {bt.alpha_pct:.2f}%")


def prepare_feature_universe(
    symbols: Sequence[str],
    target_symbol: str,
    config: RunConfig,
) -> Dict[str, pd.DataFrame]:
    frames: Dict[str, pd.DataFrame] = {}
    for idx, symbol in enumerate(symbols, start=1):
        print(f"  Downloading {symbol} ({idx}/{len(symbols)}) ...")
        try:
            raw = download_price_history(symbol, config)
            features = build_feature_frame(raw, config)
            if len(features) < config.seq_len + 120:
                raise ValueError(f"{symbol} has too little usable feature history after engineering.")
            frames[symbol] = features
        except Exception as exc:
            if symbol == target_symbol:
                raise
            print(f"    Skipping peer {symbol}: {exc}")
    return frames


def aggregate_fold_predictions(
    rows: Sequence[Tuple[np.datetime64, float, float, pd.Timestamp]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    by_date: Dict[np.datetime64, Tuple[float, float, pd.Timestamp]] = {}
    for date, prob, ret, train_end in rows:
        existing = by_date.get(date)
        if existing is None or train_end < existing[2]:
            by_date[date] = (prob, ret, train_end)
            
    ordered_dates = np.array(sorted(by_date.keys()), dtype="datetime64[ns]")
    probs = np.array([by_date[d][0] for d in ordered_dates], dtype=np.float32)
    rets = np.array([by_date[d][1] for d in ordered_dates], dtype=np.float32)
    return ordered_dates, probs, rets


def run_pipeline(query: str, config: RunConfig, initial_capital: float) -> None:
    set_seed(config.random_seed)
    device = get_device()

    resolved = resolve_ticker(query)
    discovery = discover_peers(resolved, config)
    print_discovery_summary(discovery)

    if not discovery.peers and not discovery.used_target_in_fallback:
        raise ValueError("No peers were discovered, and target fallback is disabled.")

    all_symbols = list(dict.fromkeys(discovery.peers + [discovery.target.symbol]))
    print("\nPreparing data universe")
    feature_frames = prepare_feature_universe(all_symbols, discovery.target.symbol, config)
    available_peers = [symbol for symbol in discovery.peers if symbol in feature_frames]
    
    feature_cols = [col for col in feature_frames[discovery.target.symbol].columns if col not in ["next_ret", "target_ret"]]

    target_index = feature_frames[discovery.target.symbol].index
    folds, fold_plan = make_fold_specs(target_index, config)
    if not folds:
        raise ValueError("Not enough target history to create any walk-forward fold.")
    print_walk_forward_explainer(folds, fold_plan)

    fold_summaries: List[FoldSummary] = []
    aggregate_rows: List[Tuple[np.datetime64, float, float, pd.Timestamp]] = []
    validation_preds: List[np.ndarray] = []
    validation_rets: List[np.ndarray] = []

    print_rule(f"Training on device: {device}")
    for fold in folds:
        try:
            print_note(
                f"Fold {fold.fold_id}/{len(folds)} | train {fold.train_start.date()} -> {fold.train_end.date()} | "
                f"test {fold.test_start.date()} -> {fold.test_end.date()}"
            )
            train_bundle, val_bundle, target_val_bundle, test_bundle = build_fold_datasets(
                fold=fold,
                feature_frames=feature_frames,
                feature_cols=feature_cols,
                target_symbol=discovery.target.symbol,
                peer_symbols=available_peers,
                config=config,
            )
            if len(test_bundle.y) < 30:
                print(f"  Fold {fold.fold_id}: skipped, only {len(test_bundle.y)} target test samples.")
                continue

            model = train_model(train_bundle, val_bundle, config, device)
            
            target_val_pred = predict_returns(model, target_val_bundle.X, device)
            validation_preds.append(target_val_pred)
            validation_rets.append(target_val_bundle.next_ret) 
            
            local_policy = select_trading_policy(
                pred_ret=target_val_pred,
                next_ret=target_val_bundle.next_ret,
                market=discovery.market,
                config=config,
            )

            pred_ret = predict_returns(model, test_bundle.X, device)
            
            criterion = nn.SmoothL1Loss()
            loss_score = float(criterion(torch.tensor(pred_ret), torch.tensor(test_bundle.y)).item())
            dir_acc = directional_accuracy_pct(pred_ret, test_bundle.y)

            fold_summaries.append(
                FoldSummary(
                    fold_id=fold.fold_id,
                    train_range=f"{fold.train_start.date()} -> {fold.train_end.date()}",
                    test_range=f"{fold.test_start.date()} -> {fold.test_end.date()}",
                    train_samples=len(train_bundle.y),
                    val_samples=len(val_bundle.y),
                    test_samples=len(test_bundle.y),
                    loss_score=loss_score,
                    directional_acc=dir_acc,
                    long_rate_pct=float(np.mean(pred_ret >= local_policy.signal_long_edge) * 100.0),
                    mean_pred_ret=float(np.mean(pred_ret)),
                    peers_used=len(available_peers),
                    policy_mode=local_policy.mode,
                )
            )

            for dt, prob, ret in zip(test_bundle.dates, pred_ret, test_bundle.next_ret):
                aggregate_rows.append((dt, float(prob), float(ret), fold.train_end))

        except Exception as exc:
            print(f"  Fold {fold.fold_id}: skipped because {exc}")

    if not aggregate_rows:
        raise ValueError("No successful walk-forward predictions were produced.")

    print_fold_table(fold_summaries)
    agg_dates, agg_pred, agg_ret = aggregate_fold_predictions(aggregate_rows)
    
    global_policy = select_trading_policy(
        pred_ret=np.concatenate(validation_preds),
        next_ret=np.concatenate(validation_rets),
        market=discovery.market,
        config=config,
    )
    print_policy_summary(global_policy)
    
    bt = run_backtest(
        pred_ret=agg_pred,
        next_ret=agg_ret,
        dates=agg_dates,
        market=discovery.market,
        config=config,
        initial_equity=initial_capital,
        signal_long_edge=global_policy.signal_long_edge,
        signal_short_edge=global_policy.signal_short_edge,
        allow_short=global_policy.allow_short,
    )
    
    mean_acc = float(np.mean([item.directional_acc for item in fold_summaries])) if fold_summaries else float("nan")
    print_reality_checks(mean_acc, bt, fold_summaries)
    print_backtest(bt, initial_capital, config, discovery.market)


def parse_capital_input(raw: str) -> Optional[float]:
    text = raw.strip().lower()
    if not text:
        return None
    cleaned = (
        text.replace("rs.", "").replace("rs", "").replace("inr", "")
        .replace("$", "").replace(",", "").replace("_", "").strip()
    )
    cleaned = re.sub(r"\s+", " ", cleaned)

    multipliers = {
        "k": 1_000.0, "thousand": 1_000.0, "lakh": 100_000.0, "lac": 100_000.0,
        "l": 100_000.0, "cr": 10_000_000.0, "crore": 10_000_000.0,
        "m": 1_000_000.0, "mn": 1_000_000.0, "million": 1_000_000.0,
    }

    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)\s*([a-z]+)?", cleaned)
    if match:
        value = float(match.group(1))
        suffix = match.group(2)
        if suffix:
            value *= multipliers.get(suffix, 1.0)
        return value
    if cleaned.count(".") == 1 and cleaned.replace(".", "").isdigit():
        left, right = cleaned.split(".")
        if len(right) == 3 and left.isdigit() and right.isdigit():
            return float(left + right)
    raise ValueError(f"Could not understand capital input '{raw}'.")


def prompt_float(message: str, default: float) -> float:
    raw = input(message).strip()
    if not raw:
        return default
    try:
        value = parse_capital_input(raw)
        return default if value is None else value
    except ValueError:
        print(f"Invalid number. Using default {default}.")
        return default


def main() -> None:
    config = RunConfig()
    print_run_header(config)
    initial_capital = prompt_float("\nInitial paper capital (Enter = 100000): ", 100_000.0)
    print(f"Accepted starting capital: {initial_capital:,.2f}")

    while True:
        raw_query = input("\nEnter stock(s) to test separated by commas (blank to exit): ").strip()
        if not raw_query:
            print("Exiting.")
            return
            
        queries = [q.strip() for q in raw_query.split(",") if q.strip()]
        
        for query in queries:
            print_rule(f"Processing Ticker: {query}")
            try:
                run_pipeline(query, config, initial_capital)
            except Exception as exc:
                print(f"\nRun failed for {query}: {exc}")

        again = input("\nRun another batch? [Y/n]: ").strip().lower()
        if again in {"n", "no"}:
            print("Exiting.")
            return

if __name__ == "__main__":
    main()