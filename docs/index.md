---
title: PriceFM - Foundation Model for Probabilistic Day-Ahead Electricity Price Forecasting
description: Foundation Model for Probabilistic Day-Ahead Electricity Price Forecasting
---

# OrderFusion

[![arXiv](https://img.shields.io/badge/arXiv-2502.06830-b31b1b.svg)](https://www.arxiv.org/abs/2508.04875)
[![Code](https://img.shields.io/badge/GitHub-Repository-181717.svg)](https://github.com/runyao-yu/PriceFM)

**Authors:** Runyao Yu, Chenhui Gu, Jochen Stiasny, Qingsong Wen, Wasim Sarwar Dilov, Lianlian Qi, Jochen L. Cremer

## Abstract
Electricity price forecasting in Europe presents unique challenges due to the continent's increasingly integrated and physically interconnected power market. While recent advances in deep learning and foundation models have led to substantial improvements in general time series forecasting, most existing approaches fail to capture the complex spatial interdependencies and uncertainty inherent in electricity markets. In this paper, we address these limitations by introducing a comprehensive and up-to-date dataset across 24 European countries (38 regions), spanning from 2022-01-01 to 2025-01-01. Building on this groundwork, we propose PriceFM, a spatiotemporal foundation model that integrates graph-based inductive biases to capture spatial interdependencies across interconnected electricity markets. The model is designed for multi-region, multi-timestep, and multi-quantile probabilistic electricity price forecasting. Extensive experiments and ablation studies confirm the model's effectiveness, consistently outperforming competitive baselines and highlighting the importance of spatial context in electricity markets. 

## Model Structure
![Model structure](assets/model_structure.PNG)

## Citation

```bibtex
@misc{yu2025orderfusion,
  title         = {OrderFusion: Encoding Orderbook for End-to-End Probabilistic Intraday Electricity Price Prediction},
  author        = {Yu, Runyao and Tao, Yuchen and Leimgruber, Fabian and Esterl, Tara and Cremer, Jochen L.},
  year          = {2025},
  eprint        = {2502.06830},
  archivePrefix = {arXiv},
  primaryClass  = {q-fin.CP},
  url           = {https://arxiv.org/abs/2502.06830}
}
```