"""Shared code702-v3 layers retained exactly for PriceFM_Plus code702-v4."""

import tensorflow as tf
from tensorflow.keras import layers


@tf.keras.utils.register_keras_serializable(package="PriceFMPlus")
class DenseFeatureMixer(layers.Layer):
    def __init__(self, input_width, feature_count=6, width=48,
                 extra_depth=False, **kwargs):
        super().__init__(**kwargs)
        self.input_width = int(input_width)
        self.feature_count = int(feature_count)
        self.width = int(width)
        self.extra_depth = bool(extra_depth)
        self.dense = layers.Dense(width, activation="swish", use_bias=False)
        if self.extra_depth:
            self.extra_dense = layers.Dense(
                width, activation="swish", use_bias=False)

    def call(self, inputs):
        tokens, _ = inputs
        flat = tf.reshape(tokens, (-1, 96,
                                   self.feature_count * self.input_width))
        result = self.dense(flat)
        return self.extra_dense(result) if self.extra_depth else result

    def get_config(self):
        return {**super().get_config(), "input_width": self.input_width,
                "feature_count": self.feature_count, "width": self.width,
                "extra_depth": self.extra_depth}


@tf.keras.utils.register_keras_serializable(package="PriceFMPlus")
class PriceToActualBranchCrossAttention(layers.Layer):
    def __init__(self, width=48, **kwargs):
        super().__init__(**kwargs)
        self.width = int(width)

    def build(self, input_shape):
        for name in ("query_projection", "key_projection",
                     "value_projection", "output_projection"):
            setattr(self, name, self.add_weight(
                name=name, shape=(self.width, self.width),
                initializer="glorot_uniform", trainable=True))

    def call(self, inputs):
        price, lag, lead, lag_available, lead_available = inputs
        actuals = tf.stack((lag, lead), axis=2)
        key_available = tf.stack((tf.reduce_any(lag_available, axis=-1),
                                  tf.reduce_any(lead_available, axis=-1)),
                                 axis=-1)
        query = tf.einsum("bqh,hk->bqk", price, self.query_projection)
        key = tf.einsum("bqsh,hk->bqsk", actuals, self.key_projection)
        value = tf.einsum("bqsh,hk->bqsk", actuals, self.value_projection)
        scores = tf.einsum("bqh,bqsh->bqs", query, key)
        scores *= tf.cast(self.width, scores.dtype) ** -.5
        scores = tf.where(key_available, scores,
                          tf.fill(tf.shape(scores), tf.cast(-1e9, scores.dtype)))
        weights = tf.nn.softmax(scores, axis=-1)
        weights = tf.where(key_available, weights, tf.zeros_like(weights))
        weights = tf.math.divide_no_nan(
            weights, tf.reduce_sum(weights, axis=-1, keepdims=True))
        context = tf.einsum("bqs,bqsh->bqh", weights, value)
        context = tf.einsum("bqh,hk->bqk", context, self.output_projection)
        return context, weights

    def get_config(self):
        return {**super().get_config(), "width": self.width}


@tf.keras.utils.register_keras_serializable(package="PriceFMPlus")
class RegionEmbeddingGate(layers.Layer):
    def __init__(self, region_count=7, embedding_width=48,
                 extra_depth=False, **kwargs):
        super().__init__(**kwargs)
        self.region_count = int(region_count)
        self.embedding_width = int(embedding_width)
        self.extra_depth = bool(extra_depth)
        if self.extra_depth:
            self.extra_dense = layers.Dense(
                embedding_width, activation="swish", use_bias=False)

    def build(self, input_shape):
        channels = int(input_shape[0][-1])
        self.region_embedding = self.add_weight(
            name="region_embedding", shape=(self.region_count,
                                            self.embedding_width),
            initializer="glorot_uniform", trainable=True)
        self.channel_projection = self.add_weight(
            name="channel_projection", shape=(self.embedding_width, channels),
            initializer="zeros", trainable=True)

    def call(self, inputs):
        context, region_code = inputs
        embedding = tf.nn.swish(
            tf.matmul(region_code, self.region_embedding))
        if self.extra_depth:
            embedding = self.extra_dense(embedding)
        scale = 1.0 + 0.5 * tf.tanh(
            tf.matmul(embedding, self.channel_projection))
        return context * scale[:, None, :]

    def get_config(self):
        return {**super().get_config(),
                "region_count": self.region_count,
                "embedding_width": self.embedding_width,
                "extra_depth": self.extra_depth}


@tf.keras.utils.register_keras_serializable(package="PriceFMPlus")
class TemporalSlotMixer(layers.Layer):
    """One shared bias-free 96x96 map across output quarter-hours.

    The channel axis is preserved. A residual connection lets the model retain
    each slot's existing representation while learning information exchange
    with all other slots.
    """

    def __init__(self, slots=96, **kwargs):
        super().__init__(**kwargs)
        self.slots = int(slots)
        self.time_dense = layers.Dense(self.slots, use_bias=False,
                                       name="time_dense")

    def call(self, context):
        transposed = tf.transpose(context, (0, 2, 1))
        mixed = self.time_dense(transposed)
        return context + tf.transpose(mixed, (0, 2, 1))

    def get_config(self):
        return {**super().get_config(), "slots": self.slots}


@tf.keras.utils.register_keras_serializable(package="PriceFMPlus")
class RecentPriceResidual(layers.Layer):
    """Learn one shared state from the last known 96 price slots and flags.

    The state is broadcast across target slots; the existing per-slot
    representation and forecast head remain unchanged.
    """

    def __init__(self, width=96, **kwargs):
        super().__init__(**kwargs)
        self.width = int(width)
        self.dense = layers.Dense(width, activation="swish", use_bias=False,
                                  name="recent_price_dense")

    def call(self, inputs):
        context, price_values, price_mask = inputs
        flags = price_mask[:, -96:, :]
        safe = tf.where(flags, price_values[:, -96:, :],
                        tf.zeros_like(price_values[:, -96:, :]))
        sequence = tf.concat((safe, tf.cast(flags, safe.dtype)), axis=-1)
        flattened = tf.reshape(sequence, (-1, 192))
        state = self.dense(flattened)
        return context + state[:, None, :]

    def get_config(self):
        return {**super().get_config(), "width": self.width}


@tf.keras.utils.register_keras_serializable(package="PriceFMPlus")
class RecentPriceSequenceResidual(layers.Layer):
    """Give every target slot a learned view of the last known price day.

    A shared two-channel projection embeds each observed price and validity
    flag. One bias-free 96x96 temporal map then lets each target quarter-hour
    read any quarter-hour of that known day. No target values are supplied.
    """

    def __init__(self, width=96, **kwargs):
        super().__init__(**kwargs)
        self.width = int(width)
        self.embed = layers.Dense(
            width, activation="swish", use_bias=False,
            name="recent_price_sequence_dense")
        self.mixer = TemporalSlotMixer(
            slots=96, name="recent_price_sequence_mixer")

    def call(self, inputs):
        context, price_values, price_mask = inputs
        flags = price_mask[:, -96:, :]
        safe = tf.where(flags, price_values[:, -96:, :],
                        tf.zeros_like(price_values[:, -96:, :]))
        sequence = tf.concat((safe, tf.cast(flags, safe.dtype)), axis=-1)
        return context + self.mixer(self.embed(sequence))

    def get_config(self):
        return {**super().get_config(), "width": self.width}
