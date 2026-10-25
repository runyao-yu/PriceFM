"""Quarter-hour alignment for six separate actual-feature channels.

The third code702-v3 actual channel is split into wind onshore/offshore;
price timing, validity flags, and the two branch architecture are unchanged.
"""

import tensorflow as tf
from tensorflow.keras import layers


def align_price(price_values, price_mask):
    """Seven price days -> same-slot values and validity flags [B,96,14]."""
    days = tf.reshape(price_values, (-1, 7, 96, 1))
    flags = tf.reshape(price_mask, (-1, 7, 96, 1))
    days = tf.squeeze(tf.transpose(days, (0, 2, 1, 3)), axis=-1)
    flags = tf.squeeze(tf.transpose(flags, (0, 2, 1, 3)), axis=-1)
    safe = tf.where(flags, days, tf.zeros_like(days))
    return tf.concat((safe, tf.cast(flags, safe.dtype)), axis=-1)


@tf.keras.utils.register_keras_serializable(package="PriceFMPlus")
class PriceHistoryToken(layers.Layer):
    def call(self, inputs):
        return align_price(*inputs)


FEATURES = (
    "ActualLoad", "ActualSolar", "ActualWindOnshore", "ActualWindOffshore",
    "ActualPowerFlowImport", "ActualPowerFlowExport",
)
FEATURE_COUNT = len(FEATURES)


@tf.keras.utils.register_keras_serializable(package="PriceFMPlus")
class AlignedWindSplitInputs(layers.Layer):
    """Align 624 lag and 96 lead slots by quarter-hour of day."""

    def call(self, inputs):
        lag_values, lag_mask, lead_values, lead_mask, price_values, price_mask = inputs
        lag_days = tf.reshape(lag_values[:, :576, :], (-1, 6, 96, FEATURE_COUNT))
        lag_flags = tf.reshape(lag_mask[:, :576, :], (-1, 6, 96, FEATURE_COUNT))
        half_day = lag_values[:, 576:, :]
        half_flags = lag_mask[:, 576:, :]
        day_d = tf.concat((half_day, tf.zeros_like(half_day)), axis=1)
        day_d_flags = tf.concat((half_flags, tf.zeros_like(half_flags)), axis=1)
        lag_days = tf.concat((lag_days, day_d[:, None, :, :]), axis=1)
        lag_flags = tf.concat((lag_flags, day_d_flags[:, None, :, :]), axis=1)
        lag_days = tf.transpose(lag_days, (0, 2, 3, 1))  # [B,96,6,7]
        lag_flags = tf.transpose(lag_flags, (0, 2, 3, 1))
        safe_lag = tf.where(lag_flags, lag_days, tf.zeros_like(lag_days))
        lag_token = tf.concat((safe_lag, tf.cast(lag_flags, safe_lag.dtype)),
                              axis=-1)  # [B,96,6,14]
        lag_available = tf.reduce_any(lag_flags, axis=-1)  # [B,96,6]

        safe_lead = tf.where(lead_mask, lead_values,
                             tf.zeros_like(lead_values))
        lead_token = tf.stack((safe_lead,
                               tf.cast(lead_mask, safe_lead.dtype)),
                              axis=-1)  # [B,96,6,2]
        price_token = align_price(price_values, price_mask)  # [B,96,14]
        return lag_token, lag_available, lead_token, lead_mask, price_token


__all__ = ("FEATURES", "FEATURE_COUNT", "AlignedWindSplitInputs",
           "PriceHistoryToken")
