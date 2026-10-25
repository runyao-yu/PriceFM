"""PriceFM+ architecture and compatible historical code702-v4 variants."""

import tensorflow as tf
from tensorflow.keras import Model, layers

from .alignment import (
    FEATURE_COUNT, AlignedWindSplitInputs, PriceHistoryToken,
)
from .layers import (
    DenseFeatureMixer, PriceToActualBranchCrossAttention, RegionEmbeddingGate,
    RecentPriceResidual, RecentPriceSequenceResidual, TemporalSlotMixer,
)


HIDDEN_WIDTH = 48
REGION_WIDTH = 48


def build_model(regions, loops=1, hidden_width=HIDDEN_WIDTH,
                extra_depth=False, temporal_mix=False,
                recent_price_context=False, recent_price_sequence=False,
                branch_dense_layers=None, head_dense_layers=None):
    """Build the same two-branch model with six actual types instead of five."""
    loops = int(loops)
    hidden_width = int(hidden_width)
    if loops not in (0, 1, 2, 3, 4, 5, 6):
        raise ValueError("loops must be one of 0, 1, 2, 3, 4, 5, 6")
    if hidden_width <= 0:
        raise ValueError("hidden_width must be positive")
    if branch_dense_layers is None:
        branch_dense_layers = 2 if extra_depth else 1
    if head_dense_layers is None:
        head_dense_layers = 2 if extra_depth else 1
    if branch_dense_layers not in (1, 2) or head_dense_layers not in (1, 2):
        raise ValueError("Branch and head dense layer counts must be 1 or 2")
    if recent_price_context and recent_price_sequence:
        raise ValueError("Choose only one recent-price branch")
    price_values = layers.Input((672, 1), name="price_values")
    price_mask = layers.Input((672, 1), dtype="bool", name="price_mask")
    region_code = layers.Input((len(regions),), name="region_code")
    model_inputs = {"price_values": price_values, "price_mask": price_mask,
                    "region_code": region_code}

    if loops:
        lag_values = layers.Input((624, FEATURE_COUNT), name="lag_values")
        lag_mask = layers.Input((624, FEATURE_COUNT), dtype="bool", name="lag_mask")
        lead_values = layers.Input((96, FEATURE_COUNT), name="lead_values")
        lead_mask = layers.Input((96, FEATURE_COUNT), dtype="bool", name="lead_mask")
        lag_token, lag_available, lead_token, lead_available, price_token = (
            AlignedWindSplitInputs(name="align_inputs_wind_split")([
                lag_values, lag_mask, lead_values, lead_mask,
                price_values, price_mask]))
        lag_context = DenseFeatureMixer(
            14, feature_count=FEATURE_COUNT, width=hidden_width,
            extra_depth=branch_dense_layers == 2,
            name="lag_feature_dense")([lag_token, lag_available])
        lead_context = DenseFeatureMixer(
            2, feature_count=FEATURE_COUNT, width=hidden_width,
            extra_depth=branch_dense_layers == 2,
            name="lead_feature_dense")([lead_token, lead_available])
        model_inputs.update({"lag_values": lag_values, "lag_mask": lag_mask,
                             "lead_values": lead_values, "lead_mask": lead_mask})
    else:
        price_token = PriceHistoryToken(name="align_price")([
            price_values, price_mask])

    price_context = layers.Dense(
        hidden_width, activation="swish", use_bias=False,
        name="price_encoder")(price_token)
    if branch_dense_layers == 2:
        price_extra = layers.Dense(
            hidden_width, activation="swish", use_bias=False,
            name="price_encoder_extra")(price_context)
        price_context = price_extra
    enhanced_price = price_context
    if loops:
        attention = PriceToActualBranchCrossAttention(
            hidden_width, name="price_to_actual_branch_cross_attention")
        for iteration in range(loops):
            cross_context, _ = attention([
                enhanced_price, lag_context, lead_context,
                lag_available, lead_available])
            enhanced_price = layers.Add(
                name="price_residual" if iteration == 0
                     else f"price_residual_{iteration + 1}")([
                         enhanced_price, cross_context])
    if temporal_mix:
        enhanced_price = TemporalSlotMixer(
            slots=96, name="temporal_slot_mixer")(enhanced_price)
    if recent_price_context:
        enhanced_price = RecentPriceResidual(
            hidden_width, name="recent_price_residual")([
                enhanced_price, price_values, price_mask])
    if recent_price_sequence:
        enhanced_price = RecentPriceSequenceResidual(
            hidden_width, name="recent_price_sequence_residual")([
                enhanced_price, price_values, price_mask])
    x = RegionEmbeddingGate(
        len(regions), hidden_width, extra_depth=extra_depth,
        name="region_embedding_gate")([
            enhanced_price, region_code])
    if head_dense_layers == 2:
        x = layers.Dense(hidden_width, activation="swish", use_bias=False,
                         name="pre_post_attention_dense")(x)
    x = layers.Dense(hidden_width, activation="swish", use_bias=False,
                     name="post_attention_dense")(x)
    output = layers.Dense(3, use_bias=False, name="quantiles")(x)
    model_name = f"PriceFM_Plus_code702_v4_L{loops}"
    if hidden_width != HIDDEN_WIDTH:
        model_name += f"_H{hidden_width}"
    if extra_depth:
        model_name += "_extra_depth"
    if branch_dense_layers != (2 if extra_depth else 1):
        model_name += f"_branch{branch_dense_layers}"
    if head_dense_layers != (2 if extra_depth else 1):
        model_name += f"_head{head_dense_layers}"
    if temporal_mix:
        model_name += "_temporal_mix"
    if recent_price_context:
        model_name += "_recent_price_context"
    if recent_price_sequence:
        model_name += "_recent_price_sequence"
    return Model(inputs=model_inputs, outputs=output, name=model_name)


def build_pricefm_plus(regions):
    """Canonical PriceFM+: H96, one cross-attention, temporal and recent-price paths.

    Historical experiments continue to use ``build_model`` with explicit
    flags; this wrapper fixes the current architecture without changing the
    layer topology or saved-weight compatibility.
    """
    return build_model(regions, loops=1, hidden_width=96, extra_depth=True,
                       temporal_mix=True, recent_price_context=True)
