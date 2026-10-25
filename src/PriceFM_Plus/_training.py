"""Custom training of the published three-branch model."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ._api import CASES
from ._inference import INPUT_KEYS


def customize_model(dates, feature_case, regions, hyperparams):
    import tensorflow as tf
    from . import data as D
    from .PriceFM_Plus import build_model
    from .region_connectivity import load_connectivity

    if len(dates) != 6:
        raise ValueError("dates must have train_start, train_end, val_start, val_end, test_start, test_end")
    date_values = [pd.Timestamp(value).strftime("%Y-%m-%d") for value in dates]
    if not all(date_values[i] <= date_values[i + 1] for i in range(5)):
        raise ValueError("date boundaries must be chronologically ordered")
    if not (date_values[0] < date_values[1] <= date_values[2] < date_values[3] <=
            date_values[4] < date_values[5]):
        raise ValueError("training, validation and test windows must be nonempty and disjoint")
    if feature_case not in CASES:
        raise ValueError(f"feature_case must be one of {CASES}")
    defaults = {"epochs": 500, "lr": 1e-3, "hidden_dim": 64, "loop": 2,
                "batch_size": 4096, "microbatch_size": 4096, "seed": 42}
    options = {**defaults, **(hyperparams or {})}
    unknown = set(options) - defaults.keys()
    if unknown:
        raise ValueError(f"Unknown hyperparameters: {sorted(unknown)}")
    if any(int(options[key]) < 1 for key in ("epochs", "hidden_dim", "batch_size", "microbatch_size")):
        raise ValueError("epochs, hidden_dim and batch sizes must be positive")
    if int(options["microbatch_size"]) > int(options["batch_size"]):
        raise ValueError("microbatch_size cannot exceed batch_size")
    if float(options["lr"]) <= 0:
        raise ValueError("lr must be positive")
    model_regions, adjacency = load_connectivity()
    chosen = tuple(regions or model_regions)
    if not chosen or len(chosen) != len(set(chosen)) or any(r not in model_regions for r in chosen):
        raise ValueError("regions must be distinct members of the 40-region grid")
    case = {"Realistic": "case1", "TSO": "case5", "Oracle": "case4"}[feature_case]
    D.set_variant("interp")
    D.SPLITS = {"train": (date_values[0], date_values[1]),
                "val": (date_values[2], date_values[3]),
                "test": (date_values[4], date_values[5])}
    D.TARGET_DATES = pd.date_range(date_values[0],
                                   pd.Timestamp(date_values[5]) - pd.Timedelta(days=1),
                                   freq="D", tz="UTC")
    D.SPLIT_SCHEME = "custom_" + hashlib.sha256(json.dumps(
        [date_values, chosen, feature_case]).encode()).hexdigest()[:12]
    forecasts = tuple(name for name, _ in D.CASE5_CHANNELS) if case == "case5" else ()
    D.set_scaler_mode("global_robust", chosen, forecasts, allow_empty=True)
    data_by_region, indices_by_region, scaler_by_region = {}, {}, {}
    for region in chosen:
        data, split, scalers, _ = D.build_case_region(region, case, model_regions)
        data["region_code"] = np.broadcast_to(adjacency[model_regions.index(region)],
                                               (len(D.TARGET_DATES), 40)).copy()
        data_by_region[region], indices_by_region[region] = data, split
        scaler_by_region[region] = scalers
    train = {key: np.concatenate([data_by_region[region][key][indices_by_region[region]["train"]]
                                  for region in chosen])
             for key in (*INPUT_KEYS, "target", "target_mask")}
    if len(train["target"]) == 0:
        raise ValueError("No training targets in the chosen dates")
    run_identity = hashlib.sha256(json.dumps([date_values, chosen, feature_case, options],
                                          sort_keys=True, default=str).encode()).hexdigest()[:12]
    weight_dir = Path.cwd() / "Weights" / "PriceFM_Plus" / "Custom" / run_identity
    forecast_dir = Path.cwd() / "Forecasts" / "PriceFM_Plus" / "Custom" / run_identity
    weight_dir.mkdir(parents=True, exist_ok=True)
    forecast_dir.mkdir(parents=True, exist_ok=True)
    (weight_dir / "configuration.json").write_text(json.dumps({
        "dates": date_values, "feature_case": feature_case, "regions": chosen,
        "hyperparams": options}, indent=2) + "\n")
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(int(options["seed"]))
    network = build_model(model_regions, loops=int(options["loop"]),
                          hidden_width=int(options["hidden_dim"]),
                          extra_depth=True, temporal_mix=True,
                          recent_price_context=True,
                          branch_dense_layers=2, head_dense_layers=2)
    optimizer = tf.keras.optimizers.Adam(float(options["lr"]))
    tensors = {key: tf.convert_to_tensor(value) for key, value in train.items()}
    quantiles = tf.constant([.1, .5, .9], tf.float32)

    def loss_fn(y, mask, pred):
        error = tf.where(mask, y, tf.zeros_like(y))[..., None] - pred
        loss = tf.maximum(quantiles * error, (quantiles - 1) * error)
        loss = tf.where(mask[..., None], loss, 0.)
        counts = tf.reduce_sum(tf.cast(mask, tf.float32), axis=1) * 3
        return tf.reduce_mean(tf.math.divide_no_nan(tf.reduce_sum(loss, axis=(1, 2)), counts))

    @tf.function
    def gradient_chunk(ids):
        inputs = {key: tf.gather(tensors[key], ids) for key in INPUT_KEYS}
        y = tf.gather(tensors["target"], ids)
        mask = tf.gather(tensors["target_mask"], ids)
        with tf.GradientTape() as tape:
            pred = network(inputs, training=True)
            loss = loss_fn(y, mask, pred)
        return loss, tape.gradient(loss, network.trainable_variables)

    validation = {}
    for region in chosen:
        selected = indices_by_region[region]["val"]
        data = data_by_region[region]
        validation[region] = ({key: tf.convert_to_tensor(data[key][selected]) for key in INPUT_KEYS},
                              tf.convert_to_tensor(data["target"][selected]),
                              tf.convert_to_tensor(data["eval_mask"][selected]))
        if len(selected) == 0:
            raise ValueError(f"No validation targets for {region}")
    best = {region: float("inf") for region in chosen}
    history = []
    rng = np.random.default_rng(10000 + int(options["seed"]))
    batch = int(options["batch_size"])
    microbatch = int(options["microbatch_size"])
    for epoch in range(1, int(options["epochs"]) + 1):
        if epoch == 51:
            optimizer = tf.keras.optimizers.Adam(float(options["lr"]))
            rng = np.random.default_rng(11000 + int(options["seed"]))
        order = rng.permutation(len(train["target"]))
        losses = []
        for first in range(0, len(order), batch):
            ids = order[first:first + batch]
            gradients = [tf.zeros_like(variable) for variable in network.trainable_variables]
            step_loss = 0.
            for begin in range(0, len(ids), microbatch):
                chunk = ids[begin:begin + microbatch]
                value, grads = gradient_chunk(tf.constant(chunk, tf.int32))
                fraction = len(chunk) / len(ids)
                gradients = [total + fraction * grad for total, grad in zip(gradients, grads)]
                step_loss += fraction * float(value)
            optimizer.apply_gradients(zip(gradients, network.trainable_variables))
            losses.append(step_loss)
        row = {"epoch": epoch, "train_pinball_scaled": float(np.mean(losses))}
        for region, (inputs, target, mask) in validation.items():
            val = float(loss_fn(target, mask, network(inputs, training=False)))
            row[f"val_{region}_pinball_scaled"] = val
            if val < best[region]:
                best[region] = val
                network.save_weights(weight_dir / f"best_{region}.weights.h5")
        history.append(row)
        if epoch == 1 or epoch % 10 == 0 or epoch == int(options["epochs"]):
            print(f"epoch {epoch}/{options['epochs']} train={row['train_pinball_scaled']:.4f} "
                  f"mean_val={np.mean(list(best.values())):.4f}", flush=True)
    pd.DataFrame(history).to_csv(weight_dir / "validation_history.csv", index=False)
    result = []
    for region in chosen:
        data = data_by_region[region]
        selected = indices_by_region[region]["test"]
        if len(selected) == 0:
            raise ValueError(f"No test targets for {region}")
        network.load_weights(weight_dir / f"best_{region}.weights.h5")
        inputs = {key: tf.convert_to_tensor(data[key][selected]) for key in INPUT_KEYS}
        scaled = network(inputs, training=False).numpy()
        price = scaler_by_region[region]["DayAheadPrice"]
        pred = scaled * price["scale"] + price["center"]
        truth = data["target"][selected] * price["scale"] + price["center"]
        mask = data["eval_mask"][selected] & np.isfinite(pred).all(axis=-1)
        day, slot = np.nonzero(mask)
        raw_pred = pred[day, slot]
        actual = truth[day, slot]
        error = actual[:, None] - raw_pred
        pinball = np.maximum(np.array([.1, .5, .9]) * error,
                             (np.array([.1, .5, .9]) - 1) * error).mean(axis=1)
        interval = (raw_pred[:, 2] - raw_pred[:, 0] +
                    10 * np.maximum(raw_pred[:, 0] - actual, 0) +
                    10 * np.maximum(actual - raw_pred[:, 2], 0))
        daily = pd.DataFrame({"day": day, "pinball": pinball, "interval": interval,
                              "absolute": np.abs(error[:, 1]), "squared": error[:, 1] ** 2})
        daily = daily.groupby("day").mean()
        result.append({"region": region, "Pinball": daily.pinball.mean(),
                       "Interval Score": daily.interval.mean(),
                       "MAE": daily.absolute.mean(),
                       "RMSE": np.sqrt(daily.squared.mean())})
        dates_out = np.asarray(D.TARGET_DATES[selected].strftime("%Y-%m-%d"), dtype="U10")
        np.savez_compressed(forecast_dir / f"{region}.npz",
                            dates=dates_out, day_index=day.astype(np.uint8),
                            slot_utc=slot.astype(np.uint8),
                            prediction_EUR_per_MWh=raw_pred.astype(np.float32))
    metrics = pd.DataFrame(result)
    metrics.to_csv(forecast_dir / "regional_metrics.csv", index=False)
    print(metrics.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print("Mean across regions:\n", metrics.select_dtypes("number").mean().to_string())
    return metrics
