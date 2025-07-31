# Chicago Quant Alley: Crypto Trading Simulator & Strategy Optimizer

**Chicago Quant Alley** is a **crypto trading simulator and strategy optimizer** that bridges academic concepts from stochastic finance with real-world crypto markets. The project allows you to simulate, backtest, and optimize derivatives strategies using historical data, stochastic modeling, and reinforcement-based tuning techniques.

---

## Project Overview

This repository includes:
- Historical data collection from real crypto exchanges
- Strategy implementation for crypto forwards and options
- Backtesting engine with performance evaluation metrics
- Simulation framework using stochastic models
- Multi-Armed Bandit (MAB) algorithms for parameter tuning and adaptive strategy optimization

---

## Weekly Progress

### Foundations of Quant Trading  
- Read: _**Quantitative Trading**_ by Ernest P. Chan  
- Key concepts covered:
  - Strategy design and execution architecture
  - Risk management and capital allocation
  - Building a research and execution pipeline

---

### Data Collection  
- Collected **Options** and **Forwards** data from [Delta Exchange API](https://www.delta.exchange/)  
- Cleaned and stored data in structured formats (`pandas` DataFrames and `.parquet`)  
- Built utilities for querying, transforming, and visualizing this data  

---

### Theoretical Foundations  
- Completed NPTEL courses:
  - [Probability and Statistics – IIT Kanpur](https://nptel.ac.in/courses/111104089)
  - [Introduction to Research – IIT Madras](https://nptel.ac.in/courses/109104104)
- Reading in progress: _Stochastic Finance with Python_ by Avishek Nag  
- Key focus:
  - Stochastic differential equations (SDEs)
  - Brownian motion, geometric Brownian motion (GBM)
  - Option pricing theory

---

### Simulator Development  
- Implemented Python-based **core simulation engine** for trading  
- Modeled crypto asset price paths using stochastic processes (e.g., GBM)  
- Developed modular system for testing custom strategies on synthetic and real data  
- Plug-and-play strategy interface created for future extension

---

### Strategy Optimization with MAB  
- Integrated **Multi-Armed Bandit algorithms** (e.g., Epsilon-Greedy, UCB, Exp3)  
- Simulated different reward environments (stationary & adversarial)  
- Used bandits to optimize:
  - Strike price selection
  - Entry/exit timing
  - Hedging ratios
- Added performance plots and logging to analyze convergence behavior of bandit arms  
- Compared MAB-optimized vs. static strategies in terms of Sharpe ratio, drawdown, and win rate

---

## Resources

- **Books**:
  - _Quantitative Trading_ – Ernest P. Chan
  - _Stochastic Finance with Python_ – Avishek Nag
- **Courses**:
  - [NPTEL - Probability and Statistics](https://nptel.ac.in/courses/111104089)
  - [NPTEL - Introduction to Research](https://nptel.ac.in/courses/109104104)
- **Data API**:
  - [Delta Exchange](https://www.delta.exchange/)

---

## 🛠 Tech Stack

- **Language**: Python  
- **Libraries**:  
  `pandas`, `numpy`, `matplotlib`, `scipy`, `seaborn`, `scikit-learn`, `gym`, `optuna`  
- **Tools**:
  - Jupyter Notebooks (research + prototyping)
  - Delta Exchange API (data source)
  - Modular architecture for strategy and simulation logic

---

## Final Roadmap

| Task                                                             | Status        |
|------------------------------------------------------------------|---------------|
| Read Quant Trading by Ernest Chan                                | ✅ Completed  |
| Collect and clean data from Delta Exchange                       | ✅ Completed  |
| Complete NPTEL courses                                           | ✅ Completed  |
| Read Avishek Nag’s Stochastic Finance book                       | ✅ Completed |
| Build simulator for crypto derivatives                           | ✅ Completed  |
| Implement pricing models (e.g., Black-Scholes, Bachelier)        | ✅ Completed  |
| Build strategy plug-in system                                    | ✅ Completed  |
| Add backtesting with performance metrics                         | ✅ Completed  |
| Integrate Multi-Armed Bandit algorithms for parameter tuning     | ✅ Completed  |
| Visualize performance (win rate, Sharpe, regret curves, etc.)    | ✅ Completed  |

---

## Contributing

This is a research-driven project; contributions are welcome in:
- Advanced strategy modules (e.g., volatility arbitrage, spread trading)
- Risk modeling enhancements
- Visualization and dashboarding (Plotly, Dash, Streamlit)
- Reinforcement Learning-based tuning (e.g., contextual bandits, policy gradients)

---
