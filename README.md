# Automated Quantitative Factor Generation Pipeline

This project implements an automated pipeline for generating and testing quantitative trading factors. It utilizes a recursive expression generation algorithm to create mathematical formulas based on financial market data, followed by a rigorous two-stage validation process to filter robust alpha factors. The system also includes factor library management to ensure diversity and prevent multicollinearity.

---

## 📁 Table of Contents

- [Project Overview](#-project-overview)
- [Features](#-features)
- [Directory Structure](#-directory-structure)

---

## 🚀 Project Overview

The system is designed to automatically discover predictive factors in financial markets through the following iterative process:

1. **Expression Generation:** Recursively combines base variables (price, volume, returns) using mathematical and statistical operators.
2. **Two-Stage Testing:** Validates factors on a "Small Sample" (2022-2025) and a "Full Sample" (2017-2025).
3. **Factor Library Management:** Removes highly correlated factors to maintain a diverse library.

---

## ✨ Features

- **Recursive Expression Generation:** Generates complex expressions based on complexity levels and operator arity.
- **Multi-Process Testing:** Utilizes parallel processing to accelerate factor calculation and evaluation.
- **Rigorous Validation:** Employs strict performance thresholds (Sharpe Ratio, Max Drawdown) for factor acceptance.
- **Correlation Control:** Implements a deduplication strategy to keep absolute correlation between factors below 70%.
- **Modular Design:** Separates data fetching, processing, expression generation, and testing logic.

---

## 📂 Directory Structure

The project follows a clean separation of concerns. The core structure is as follows:

```text
project_root/
├── Config.py                 # Root and Token configurations
├── requirements.txt          # Python dependencies
├── auto_factor_pipeline.py   # Main automation pipeline
├── data_download/
│   ├── data_api.py           # Tushare API data fetching
│   ├── data_process.py       # Data merging and cleaning
├── tools/
│   ├── ExpressionGenerator.py # Logic for generating expressions
│   ├── FactorTest.py          # Logic for testing factor performance
│   ├── FactorBaseManager.py   # Logic for managing factor library diversity
│   ├── DataLoader.py          # Data loading and preprocessing
│   └── FunctionSet.py         # Mathematical and statistical operators
```
