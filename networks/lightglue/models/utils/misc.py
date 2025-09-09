import math
from typing import List, Optional, Tuple

import torch


def to_sequence(map):
    return map.flatten(-2).transpose(-1, -2)


def to_map(sequence):
    n = sequence.shape[-2]
    e = math.isqrt(n)
    assert e * e == n
    assert e * e == n
    sequence.transpose(-1, -2).unflatten(-1, [e, e])


def pad_to_length(
    x,
    length: int,
    pad_dim: int = -2,
    mode: str = "zeros",  # zeros, ones, random, random_c
    bounds: Tuple[Optional[int], Optional[int]] = (None, None),
):
    # Safety check for empty or invalid tensors
    if x.numel() == 0:
        # Return a tensor with the expected shape filled with zeros
        shape = list(x.size())
        shape[pad_dim] = length
        return torch.zeros(*shape, device=x.device, dtype=x.dtype)
    
    shape = list(x.size())
    d = x.size(pad_dim)
    assert d <= length
    if d == length:
        return x
    
    # Ensure padding length is positive to avoid negative tensor shapes
    pad_length = length - d
    if pad_length <= 0:
        return x
    shape[pad_dim] = pad_length

    low, high = bounds

    if mode == "zeros":
        xn = torch.zeros(*shape, device=x.device, dtype=x.dtype)
    elif mode == "ones":
        xn = torch.ones(*shape, device=x.device, dtype=x.dtype)
    elif mode == "random":
        low = low if low is not None else x.min()
        high = high if high is not None else x.max()
        # Use torch.rand for ONNX compatibility instead of uniform_
        xn = torch.rand(*shape, device=x.device, dtype=x.dtype) * (high - low) + low
    elif mode == "random_c":
        low, high = bounds  # we use the bounds as fallback for empty seq.
        
        # Ensure bounds are valid and shape dimensions are positive for ONNX compatibility
        if low is None or high is None or low >= high or any(dim <= 0 for dim in shape):
            # Fallback to zeros if bounds are invalid or shape has negative dimensions
            xn = torch.zeros(*shape, device=x.device, dtype=x.dtype)
        else:
            xn = torch.cat(
                [
                    # Use torch.rand for ONNX compatibility instead of uniform_
                    torch.rand(*shape[:-1], 1, device=x.device, dtype=x.dtype) * (
                        (x[..., i].max() if d > 0 else high) - (x[..., i].min() if d > 0 else low)
                    ) + (x[..., i].min() if d > 0 else low)
                    for i in range(shape[-1])
                ],
                dim=-1,
            )
    else:
        raise ValueError(mode)
    return torch.cat([x, xn], dim=pad_dim)


def pad_and_stack(
    sequences: List[torch.Tensor],
    length: Optional[int] = None,
    pad_dim: int = -2,
    **kwargs,
):
    if not sequences:
        raise ValueError("Cannot pad_and_stack empty sequence list")
    
    if length is None:
        length = max([x.size(pad_dim) for x in sequences])
    
    # Ensure length is positive to avoid negative tensor shapes
    if length <= 0:
        raise ValueError(f"Invalid length {length} for pad_and_stack")
    
    # Additional safety check for ONNX compatibility
    if length > 10000:  # Reasonable upper bound to prevent memory issues
        length = 10000

    y = torch.stack([pad_to_length(x, length, pad_dim, **kwargs) for x in sequences], 0)
    return y
