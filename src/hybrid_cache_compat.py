"""Batch-shrinking support for hybrid linear-attention caches.

The generation loops in ``src.generation_utils`` drop finished sequences out of the
running batch and shrink the KV cache to match. Models whose layers are recurrent
instead of attentive -- e.g. the Gated DeltaNet layers that make up 24 of Qwen3.5-4B's
32 layers, three for every full-attention layer -- keep a conv state and a recurrent
state rather than key/value tensors, and ship no ``batch_select_indices``. Calling it on
the cache as a whole either raises ``AttributeError`` or, if guarded by ``hasattr``,
silently leaves the cache at the old batch size while the input tensors shrink -- a shape
mismatch on the next forward.

``batch_select_hybrid_cache`` is ported from upstream SwiReasoning
(https://github.com/sdc17/SwiReasoning), which already supported Qwen3.5. It dispatches
per layer: recurrent layers go through ``reorder_cache`` (whose beam-search index_select
along dim 0 is exactly the shrink we want), attention layers keep the fast
``batch_select_indices`` path, and legacy tuple caches are rebuilt tensor by tensor.
"""

import torch


def _select_tensor_batch(value, indices):
    if value is None or not isinstance(value, torch.Tensor) or value.ndim == 0:
        return value
    return value.index_select(0, indices.to(value.device))


def _batch_select_cache_layer(layer, indices):
    has_linear_states = hasattr(layer, "conv_states") or hasattr(layer, "recurrent_states")
    if has_linear_states and hasattr(layer, "reorder_cache"):
        layer.reorder_cache(indices)
    elif hasattr(layer, "batch_select_indices"):
        layer.batch_select_indices(indices)
    elif hasattr(layer, "reorder_cache"):
        layer.reorder_cache(indices)
    else:
        for attr in ("keys", "values", "conv_states", "recurrent_states"):
            value = getattr(layer, attr, None)
            if isinstance(value, torch.Tensor):
                setattr(layer, attr, _select_tensor_batch(value, indices))

    if hasattr(layer, "max_batch_size"):
        try:
            layer.max_batch_size = int(indices.numel())
        except Exception:
            pass


def _needs_layerwise_batch_select(past_key_values):
    layers = getattr(past_key_values, "layers", None)
    if layers is None:
        return False
    for layer in layers:
        has_linear_states = hasattr(layer, "conv_states") or hasattr(layer, "recurrent_states")
        if has_linear_states or not hasattr(layer, "batch_select_indices"):
            return True
    return False


def batch_select_hybrid_cache(past_key_values, indices):
    """Keep only ``indices`` along the batch dimension of ``past_key_values``.

    Returns the cache: in-place for modern Cache objects, a rebuilt tuple for legacy ones.
    """
    if past_key_values is None:
        return past_key_values

    if hasattr(past_key_values, "batch_select_indices") and not _needs_layerwise_batch_select(past_key_values):
        past_key_values.batch_select_indices(indices)
        return past_key_values

    if hasattr(past_key_values, "layers"):
        for layer in past_key_values.layers:
            _batch_select_cache_layer(layer, indices)
        return past_key_values

    if hasattr(past_key_values, "batch_select_indices"):
        past_key_values.batch_select_indices(indices)
        return past_key_values

    if isinstance(past_key_values, tuple):
        selected_layers = []
        for layer in past_key_values:
            if isinstance(layer, tuple):
                selected_layers.append(tuple(_select_tensor_batch(v, indices) for v in layer))
            else:
                selected_layers.append(_select_tensor_batch(layer, indices))
        return tuple(selected_layers)
    return past_key_values
