# LSTM-Market-Microstructure-Experiments
A quantitative research autopsy documenting why Deep Learning (LSTMs) struggles with noisy financial time-series data. Includes two fully engineered PyTorch architectures (Sector-Peer mapping and N-Year historical sequences) complete with walk-forward validation, adaptive thresholding, and backtesting. Concludes with a post-mortem on why Gradient Boosted Trees (XGBoost) remain superior for tabular market regimes.
<div align="center">

# 📉 LSTM Market Microstructure Experiments
### A Quantitative Research Autopsy on Deep Learning in Financial Time-Series

</div>

## 🔬 Abstract
This repository documents a rigorous quantitative experiment designed to test whether Long Short-Term Memory (LSTM) neural networks can successfully capture and trade market momentum. 

After engineering two distinct PyTorch architectures, running walk-forward validation, and executing realistic backtests accounting for slippage and spread, the definitive conclusion is that both models failed to reliably beat a 50% coin-flip. 

Rather than deleting the code, this repository serves as a transparent, scientific post-mortem. It breaks down the exact network architectures, analyzes the mathematical flaws in the hypotheses, and concludes why tree-based gradient boosting remains the superior choice for tabular market regimes.

---

## 🧠 Part 1: The Core Hypothesis (Why LSTMs?)
The initial logic for deploying Deep Learning in this domain was highly intuitive: **Stock markets are sequential.** Because price action relies heavily on historical context, standard feed-forward networks (which have no memory) are ill-equipped for the task. LSTMs, however, are explicitly designed to maintain a running context vector. They possess both short-term memory (recent price action) and long-term memory (macro trends), dynamically adjusting the weight of the past to influence the present prediction. 

Theoretically, an LSTM should perfectly map the sequential nature of market momentum. We built two completely different models to prove this.

---

## 🏗️ Part 2: Experiment A — The Sector-Aware Peer LSTM
**The Logic:** Deep learning models often fail on single stocks because they memorize isolated noise. To force the model to learn the underlying "physics" of a sector, this architecture trains on a basket of competitors (e.g., General Motors, Ford, Rivian) and is then evaluated strictly on the target asset (e.g., Tesla). 

**The Engineering (`Sector_Peer_LSTM.py`):**
* **Volatility-Normalized Targets:** Instead of predicting raw percentages (which compresses the signal for slow-moving stocks), the target variable was scaled by the Average True Range (ATR). The model predicts magnitude—e.g., "Will it move 0.5x its ATR?"
* **Walk-Forward Validation:** Trained in rolling chronological chunks (10 years train, 2 years test) to prevent future data leakage.
* **Loss & Architecture:** A `SimpleLSTM` optimized with `SmoothL1Loss` to gently handle market outliers without ripping the weights apart.

### 🩸 The Autopsy: Why it Backfired
Despite heavy engineering, the model suffered massive negative alpha (e.g., underperforming Buy & Hold on GOOGL by -488%). 
1. **The Cross-Asset Fallacy:** Forcing the network to learn the market microstructure of legacy automakers and applying it to a hyper-growth tech stock broke the math. Institutional capital flows through these assets differently; the network learned a "blended" behavior that doesn't actually exist in the real world.
2. **Target vs. Evaluation Mismatch:** The network was trained as a *regression* task (minimizing the error of the exact ATR decimal) but evaluated on *directional accuracy* (up vs. down). To minimize Smooth L1 Loss on extremely noisy data, the model just predicted numbers safely close to zero, completely destroying its directional edge.

---

## ⚙️ Part 3: Experiment B — The 12-Year Deep Sequence LSTM
**The Logic:** If cross-asset training fails, what if we feed a highly advanced network the entire 12-year macroeconomic history of a single stock?

**The Engineering (`Single_Stock_LSTM.py`):**
* **Framework:** Fully encapsulated using PyTorch Lightning for mixed-precision GPU acceleration and automated gradient clipping.
* **Scaled Dot-Product Attention:** An attention head was bolted onto the LSTM to dynamically score which specific days in the 60-day lookback window actually mattered (e.g., an earnings gap vs. a flat Tuesday).
* **Focal Loss:** Replaced standard Binary Cross-Entropy with `FocalLoss` ($\gamma=1.5$) to mathematically punish the network for getting "hard" predictions wrong.
* **Execution:** A strict swing-trading policy holding for 3 days to beat the bid/ask spread, utilizing adaptive conviction thresholds (only trading the top 20% of signals).

### 🩸 The Autopsy: Why it Backfired
Despite flawless architectural compilation, the model yielded a ~49% win rate and negative Sharpe ratios.
1. **The Regime Trap:** By pulling 12 years of history and splitting it chronologically, the model trained on the mechanics of 2012–2020 (zero interest rates, massive QE) and tested on 2022–2024 (inflation, rate hikes). It was trained to drive on dry asphalt and evaluated on ice.
2. **Focal Loss Overfitting:** In computer vision, Focal Loss forces the network to focus on hard-to-classify images. In finance, "hard-to-classify" data is just black-swan random noise. The math actively forced the LSTM to dedicate its training power to memorizing unpredictable anomalies instead of learning the baseline trend.
3. **Over-Engineering:** Two layers of LSTMs, Attention heads, and GELU activations for a dataset of only ~2,000 rows. It instantly over-parameterized the space.

---

## 🏁 Part 4: The Grand Conclusion (Signal vs. Noise)
After hours of optimization, custom loss functions, and architectural tuning, the definitive takeaway from this research is that **LSTMs are structurally the wrong tool for raw tabular OHLCV data.**

### The Mathematical Reality
LSTMs were engineered to dominate datasets with a massive **Signal-to-Noise ratio** (like natural language processing, where grammar rules are strictly stationary). Financial markets are the exact inverse: **99% noise and 1% signal**, locked in a highly adversarial, non-stationary environment where the "rules" degrade the moment they are discovered.

Because Deep Learning models possess such immense mathematical capacity, they do what they are programmed to do: they connect the dots perfectly. On financial data, this means they achieve near-zero training loss by perfectly memorizing the 99% noise. When hit with unseen test data, they realize the memorized rules are useless and default to a 50/50 coin flip.

### The Next Move: Gradient Boosted Trees
To capture momentum in tabular market data, algorithms must ruthlessly segment features while resisting the urge to connect sequential noise. 

Going forward, the quantitative focus shifts entirely back to **Tree-Based Models (XGBoost, LightGBM)**. Trees are highly resistant to noise, less prone to catastrophic overfitting, and excel at finding non-linear thresholds. Furthermore, predicting absolute price direction will be abandoned in favor of **Cross-Sectional Ranking**—training the algorithm to rank a universe of stocks by relative momentum to construct market-neutral, alpha-generating portfolios.
