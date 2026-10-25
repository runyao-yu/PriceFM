# PriceFM

An Open-Source Deep Neural Network for Day-Ahead Price Forecasting 🏆

Our **[project page](Project_page/index.html)** shows the model structure, European electricity data, [results tables](Project_page/index.html#results), and [interactive forecasts for 40 regions](Project_page/index.html#forecasts).

![Structure of PriceFM+ and its component ablations](Project_page/images/model_figure.svg)

## 📢 News

**26 Oct 2026.** PriceFM+ is upgraded from v1 to v2. The updated model uses looped cross-attention, incorporates cross-border power flows, and is benchmarked against the latest generic foundation models.

**6 Aug 2025.** PriceFM was created as the first European-level pretrained foundation model for day-ahead electricity price forecasting.

## 💾 Data and weights

The raw and processed data, model weights, and forecasts are available from the public [European_Foundation_Model_DAM dataset on Hugging Face](https://huggingface.co/datasets/RunyaoYu/European_Foundation_Model_DAM).

These data are provided for research purposes only. The raw electricity data can also be downloaded from the ENTSO-E Transparency Platform. We provide [data collection code](src/PriceFM_Plus/data_pipeline/download.py) that queries the ENTSO-E API and adds new dates through `PriceFM_Plus.add_data()`. Users must apply for their own ENTSO-E API key.

## 🚀 Repository

    ├── src/PriceFM_Plus/                 installable package, data collection, modeling, and evaluation
    ├── tutorials/                       Jupyter tutorial with saved example results
    ├── Project_page/                    project page and interactive forecast viewer
    ├── pyproject.toml                   package metadata and dependencies
    └── README.md

The [Jupyter tutorial](tutorials/PriceFM_Plus_tutorial.ipynb) shows how to download the released files, reproduce results, and train a customized model.

## 🧰 Required packages

- Python 3.10 or newer
- NumPy 1.24 or newer
- pandas 2.0 or newer
- huggingface_hub 0.30 or newer
- tqdm 4.65 or newer

Install version 0.1.0 directly from the released wheel:

```bash
python -m pip install "https://huggingface.co/datasets/RunyaoYu/European_Foundation_Model_DAM/resolve/main/pricefm_plus-0.1.0-py3-none-any.whl"
```

To install from this repository instead, run `python -m pip install -e .`. Optional dependencies for training, figure generation, and foundation-model inference are listed in [pyproject.toml](pyproject.toml).

## Package functions

```python
import PriceFM_Plus

PriceFM_Plus.download_data("processed")
PriceFM_Plus.download_weights("PriceFM_Plus")
PriceFM_Plus.download_forecasts("PriceFM_Plus")  # MLP, LSTM, Transformer, etc. are also available

# Extend the data with your own ENTSO-E API key.
PriceFM_Plus.add_data(start_date, end_date, entsoe_api_key)

# Reproduce forecasts from saved weights, metrics from downloaded forecasts,
# and figures from the released plotting data.
PriceFM_Plus.reproduce_forecasts("PriceFM_Plus")
PriceFM_Plus.reproduce_metrics()  # Download all forecast archives and processed data first
PriceFM_Plus.reproduce_figures()

# Train and evaluate a custom configuration.
PriceFM_Plus.customize(dates, feature_case="Realistic", regions=regions, hyperparams=hyperparams)
```

See the [Jupyter tutorial](tutorials/PriceFM_Plus_tutorial.ipynb) for prerequisites, examples, and saved results.

## 📖 Citation

If you find our work useful or use our forecasts, please cite us.

PriceFM+:

```bibtex
@misc{YuPriceFMPlus2026,
  title  = {Training Paradigms and Foundation Models for European Electricity Price Forecasting},
  author = {Runyao Yu and Derek W. Bunn and Rafal Weron and Jalal Kazempour and Pierre Pinson and Takuji Matsumoto and Peter Palensky and Ricardo Jorge Bessa and Qingsong Wen and Jochen L. Cremer},
  year   = {2026},
  note   = {Manuscript in preparation}
}
```

PriceFM:

```bibtex
@misc{YuPriceFM,
  title         = {PriceFM: Foundation Model for Probabilistic Electricity Price Forecasting},
  author        = {Runyao Yu and Chenhui Gu and Jochen Stiasny and Qingsong Wen and Wasim Sarwar Dilov and Lianlian Qi and Jochen L. Cremer},
  year         = {2026},
  eprint        = {2508.04875},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CE},
  url           = {https://arxiv.org/abs/2508.04875}
}
```

We will keep upgrading PriceFM by adding more features. If you have ideas to improve PriceFM and want to contribute, please contact [ryu@london.edu](mailto:ryu@london.edu). If you are interested in other market price forecasting and want to know more, please visit [www.deepprior.com](https://www.deepprior.com).
