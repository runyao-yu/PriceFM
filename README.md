# PriceFM_Plus

Python tools for reproducing European day-ahead electricity price forecasting results. The public data, model weights, and saved forecasts are hosted in the [European_Foundation_Model_DAM dataset](https://huggingface.co/datasets/RunyaoYu/European_Foundation_Model_DAM). This repository contains the installable code and a [step-by-step Jupyter tutorial](tutorials/PriceFM_Plus_tutorial.ipynb) with saved verification outputs beneath each example.

## Install

The package is version `0.1.0`. Until it is published to a Python package index, install the verified wheel directly:

```bash
python -m pip install "https://huggingface.co/datasets/RunyaoYu/European_Foundation_Model_DAM/resolve/main/pricefm_plus-0.1.0-py3-none-any.whl"
```

Alternatively, from this repository:

```bash
python -m pip install -e .
```

Install optional dependencies only for the work you intend to run: `.[figures]` for figure generation, `.[train]` for trained-model inference and custom training, or the model-specific extras `.[chronos]`, `.[timesfm]`, `.[tabpfn]`, and `.[tirex]` for foundation-model inference. A GPU is useful for full inference and training.

## Reproduce saved results

Run these commands from the directory where you want the outputs. Downloads create `Data`, `Weights`, and `Forecasts` folders there. The archives are large, so start with only the resources you need.

```python
import PriceFM_Plus

PriceFM_Plus.download_data("processed")
PriceFM_Plus.download_forecasts("PriceFM_Plus")
PriceFM_Plus.download_weights("PriceFM_Plus")
```

To reproduce the paper's comparison table **without inference**, download the saved forecasts for `MLP`, `LSTM`, `Transformer`, `PriceFM_Plus`, `TabPFN_TS_3`, `TiRex_2`, `TimesFM_3`, `Chronos_2`, and `Persistence`, then call `PriceFM_Plus.reproduce_metrics()`. This reads saved forecasts and observed prices and writes `Metrics/table3.csv` with pinball loss, interval score, MAE, and RMSE.

For the five grouped figures and four ablation panels, install `.[figures]` and call `PriceFM_Plus.reproduce_figures()`. It regenerates the PDFs in `Figures` from the released plotting source and **derived figure-input tables**. It does not recalculate those figure-input tables from the raw data.

To rerun an archived trained model on the original held-out folds, install `.[train]`, download processed data and that model's weights, then call `PriceFM_Plus.reproduce_forecasts("PriceFM_Plus")`. The default covers the Realistic, TSO, and Oracle feature cases and test folds 1–3. Fold 4 has validation weights but no held-out test forecast. Foundation-model inference uses the corresponding optional extra. Where checkpoint redistribution is unavailable, the function asks before downloading pinned upstream weights and prompts for a token only if access requires one.

## Extend data or train

`PriceFM_Plus.add_data(start_date, end_date, entsoe_api_key)` fetches new ENTSO-E records and extends the local raw and processed data. Download processed data first. Store the API key outside notebooks and source control.

`PriceFM_Plus.customize(dates, feature_case, regions, hyperparams)` trains, validates, tests, and prints region-level metrics. `dates` contains six boundaries: training start/end, validation start/end, and test start/end. End dates are exclusive. `feature_case` is `Realistic`, `TSO`, or `Oracle`. Omitted hyperparameters use the paper settings. See the tutorial for a small example.

## Repository contents

- `src/PriceFM_Plus/`: installable package and bundled preprocessing metadata.
- `tutorials/PriceFM_Plus_tutorial.ipynb`: guided download, reproduction, and customization examples.
- `pyproject.toml`: package metadata and optional dependencies.

Large research outputs, release ZIPs, local data, drafts, caches, and credentials in the working directory are excluded from Git. No model weights or forecasts are embedded in the package.
