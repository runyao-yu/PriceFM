---
title: PriceFM 
description: Foundation Model for Probabilistic Day-Ahead Electricity Price Forecasting
---

![Affiliations](assets/affiliations.PNG){: style="float:left; height:120px;" }
<div style="clear:both;"></div>

![teaser](assets/Trade.gif){: style="float:right; height:84px; margin-left:12px;" }
# PriceFM: Foundation Model for Probabilistic Day-Ahead Electricity Price Forecasting
<div style="clear:both;"></div>

[![arXiv](https://img.shields.io/badge/arXiv-2502.06830-b31b1b.svg)](https://www.arxiv.org/abs/2508.04875)
[![Code](https://img.shields.io/badge/GitHub-Repository-181717.svg)](https://github.com/runyao-yu/PriceFM)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/runyao-yu/)
[![Email](https://img.shields.io/badge/Email-Contact-D14836?logo=gmail&logoColor=white)](mailto:runyao.yu@tudelft.nl)

**Authors:** Runyao Yu, Chenhui Gu, Jochen Stiasny, Qingsong Wen, Wasim Sarwar Dilov, Lianlian Qi, Jochen L. Cremer

## Abstract
We introduce a comprehensive and up-to-date dataset across 24 European countries (38 regions), spanning from 2022 to 2025. 

Building on this groundwork, we propose PriceFM, a spatiotemporal foundation model that integrates graph-based inductive biases to capture spatial interdependencies across interconnected electricity markets. The model is designed for multi-region, multi-timestep, and multi-quantile probabilistic electricity price forecasting. 

Extensive experiments and ablation studies confirm the model's effectiveness, consistently outperforming competitive baselines and highlighting the importance of spatial context in electricity markets. 

## Model Structure
![Model structure](assets/model_structure.PNG)

## Citation

```bibtex
@misc{yu2025PriceFM,
      title={PriceFM: Foundation Model for Probabilistic Electricity Price Forecasting}, 
      author={Runyao Yu and Chenhui Gu and Jochen Stiasny and Qingsong Wen and Wasim Sarwar Dilov and Lianlian Qi and Jochen L. Cremer},
      year={2025},
      eprint={2508.04875},
      archivePrefix={arXiv},
      primaryClass={cs.CE},
      url={https://arxiv.org/abs/2508.04875}, 
}
```