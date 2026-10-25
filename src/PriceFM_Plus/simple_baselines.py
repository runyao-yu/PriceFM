"""Plain MLP, LSTM and Transformer baselines for the three feature cases.

Inputs retain the existing seven-day issue-time protocol. Missing or unavailable
features are replaced by zero *after* train-only robust scaling. These models
do not receive an observation mask or PriceFM+ alignment/mixing layers.
"""
from __future__ import annotations

import numpy as np
import tensorflow as tf


def _finite(x):
    return tf.where(tf.math.is_finite(x), x, tf.zeros_like(x))


class SimpleBaseline(tf.keras.Model):
    """Return 96 quarter-hour forecasts at the 0.1, 0.5 and 0.9 quantiles."""

    def __init__(self, family: str, lstm_architecture: str = "original"):
        super().__init__(name=f"simple_{family.lower()}")
        if family not in {"SimpleMLP", "SimpleLSTM", "SimpleTransformer"}:
            raise ValueError(family)
        self.family = family
        if lstm_architecture not in ("original", "patch8"):
            raise ValueError(lstm_architecture)
        if family != "SimpleLSTM" and lstm_architecture != "original":
            raise ValueError("LSTM architecture applies only to SimpleLSTM")
        self.lstm_architecture = lstm_architecture
        if family == "SimpleMLP":
            self.hidden = [tf.keras.layers.Dense(64, activation="relu")
                           for _ in range(3)]
        elif family == "SimpleLSTM":
            self.hidden = [tf.keras.layers.LSTM(64, return_sequences=i == 0)
                           for i in range(2)]
            self.region_dense = tf.keras.layers.Dense(64, activation="relu")
        else:
            self.projection = tf.keras.layers.Dense(64)
            self.attention = [tf.keras.layers.MultiHeadAttention(
                num_heads=2, key_dim=32) for _ in range(2)]
            self.region_dense = tf.keras.layers.Dense(64, activation="relu")
            positions = np.arange(96)[:, None]
            frequencies = np.exp(np.arange(0, 64, 2) * (-np.log(10000.0) / 64))
            angles = positions * frequencies[None, :]
            encoding = np.zeros((96, 64), np.float32)
            encoding[:, 0::2] = np.sin(angles)
            encoding[:, 1::2] = np.cos(angles)
            self.position_encoding = tf.constant(encoding[None, ...])
        self.head = tf.keras.layers.Dense(288)

    def call(self, inputs, training=False):
        price = _finite(inputs["price_values"])  # (B,672,1)
        lag = _finite(inputs["lag_values"])      # (B,624,6)
        lead = _finite(inputs["lead_values"])    # (B,96,6)
        region = tf.cast(inputs["region_code"], tf.float32)  # (B,40)
        if self.family == "SimpleMLP":
            x = tf.concat([tf.reshape(price, (-1, 672)),
                           tf.reshape(lag, (-1, 624 * 6)),
                           tf.reshape(lead, (-1, 96 * 6)), region], axis=-1)
            for dense in self.hidden:
                x = dense(x)
        else:
            # A plain chronological sequence: 672 historical prices; the
            # first 624 slots also have observed fundamentals; 96 future
            # slots contain only the target-day lead features.
            batch = tf.shape(price)[0]
            price_seq = tf.concat([price, tf.zeros((batch, 96, 1), price.dtype)], 1)
            lag_seq = tf.concat([lag, tf.zeros((batch, 144, 6), lag.dtype)], 1)
            lead_seq = tf.concat([tf.zeros((batch, 672, 6), lead.dtype), lead], 1)
            x = tf.concat([price_seq, lag_seq, lead_seq], -1)  # (B,768,13)
            if self.family == "SimpleLSTM":
                if self.lstm_architecture == "patch8":
                    x = tf.reshape(x, (batch, 96, 8 * 13))
                for layer in self.hidden:
                    x = layer(x, training=training)
                    if self.lstm_architecture == "original":
                        x = tf.nn.relu(x)
            else:
                # Eight adjacent quarter-hour slots form one learned token.
                # The full 768-slot input remains available to the projection.
                x = tf.reshape(x, (batch, 96, 8 * 13))
                x = self.projection(x) + self.position_encoding
                for attn in self.attention:
                    x = x + attn(x, x, training=training)
                x = x[:, -1, :]
            x = x * self.region_dense(region)
        return tf.reshape(self.head(x), (-1, 96, 3))


def build_simple_model(family: str, lstm_architecture: str = "original"):
    model = SimpleBaseline(family, lstm_architecture=lstm_architecture)
    model({"price_values": tf.zeros((1, 672, 1)),
           "lag_values": tf.zeros((1, 624, 6)),
           "lead_values": tf.zeros((1, 96, 6)),
           "region_code": tf.zeros((1, 40))})
    return model
