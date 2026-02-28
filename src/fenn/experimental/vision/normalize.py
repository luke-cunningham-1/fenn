import numpy as np
from typing import Literal

from .vision_utils import detect_format


def normalize_batch(
    array: np.ndarray,
    mode: Literal["0_1", "minus1_1", "imagenet_stats", "zscore"] = "0_1",
) -> np.ndarray:
    """
    Normalize a batch of images using the specified normalization mode.
    
    Args:
        array: Input image batch array. Must have batch dimension as first dimension:
            - (N, H, W) - batch of grayscale images
            - (N, H, W, C) - batch with channels last
            - (N, C, H, W) - batch with channels first
        mode: Normalization mode to apply:
            - "0_1": Scale values to [0, 1] range
            - "minus1_1": Scale values to [-1, 1] range
            - "imagenet_stats": Normalize using ImageNet statistics (mean=[0.485, 0.456, 0.406],
              std=[0.229, 0.224, 0.225]) for RGB channels
            - "zscore": Standardize using z-score normalization (mean=0, std=1)
    
    Returns:
        Normalized array with same shape as input, float64 dtype.
    
    Raises:
        TypeError: If array is not a numpy array
        ValueError: If array doesn't have batch dimension or has unsupported format
        ValueError: If mode is not one of the supported modes
    """
    if not isinstance(array, np.ndarray):
        raise TypeError(f"Expected numpy.ndarray, got {type(array)}")

    if array.ndim < 3 or array.ndim > 4:
        raise ValueError(
            f"Array must have batch dimension. Expected 3D greyscale (N, H, W) or 4D color "
            f"(N, H, W, C) / (N, C, H, W), got {array.ndim}D array with shape {array.shape}. "
            f"For single images, wrap with array[np.newaxis, ...] to add batch dimension."
        )
    
    if mode == "0_1":
        return _normalize_0_1(array)
    elif mode == "minus1_1":
        return _normalize_minus1_1(array)
    elif mode == "imagenet_stats":
        return _normalize_imagenet_stats(array)
    elif mode == "zscore":
        return _normalize_zscore(array)
    else:
        raise ValueError(
            f"Unsupported normalization mode: {mode}. "
            f"Expected one of: '0_1', 'minus1_1', 'imagenet_stats', 'zscore'"
        )


def _normalize_0_1(array: np.ndarray) -> np.ndarray:
    """
    Normalize array to [0, 1] range.
    
    Normalizes each image in the batch independently by scaling values so that
    the minimum becomes 0 and the maximum becomes 1. Formula: (x - min) / (max - min).
    
    If an image has constant values (min == max), it will be set to 0.5.
    All channels (including alpha) are normalized using the same min/max strategy.
    
    Args:
        array: Input array with batch dimension
    
    Returns:
        Normalized array in [0, 1] range with float64 dtype
    """
    # Convert to float64 to ensure we can represent values in [0, 1]
    array_float = array.astype(np.float64)

    format_info = detect_format(array)
    channel_location = format_info["channel_location"]
    
    # Normalize all channels using the same strategy
    if channel_location == "first":
        min_vals = np.min(array_float, axis=(2, 3), keepdims=True)
        max_vals = np.max(array_float, axis=(2, 3), keepdims=True)
    else:  # channel_location is None or "last"
        min_vals = np.min(array_float, axis=(1, 2), keepdims=True)
        max_vals = np.max(array_float, axis=(1, 2), keepdims=True)
    
    range_vals = max_vals - min_vals
    return np.where(
        range_vals > 0,
        (array_float - min_vals) / range_vals,
        0.5
    )


def _normalize_minus1_1(array: np.ndarray) -> np.ndarray:
    """
    Normalize array to [-1, 1] range.
    
    Normalizes each image in the batch independently by scaling values so that
    the minimum becomes -1 and the maximum becomes 1. Formula: 2 * (x - min) / (max - min) - 1.
    
    If an image has constant values (min == max), it will be set to 0.0.
    All channels (including alpha) are normalized using the same min/max strategy.
    
    Args:
        array: Input array with batch dimension
    
    Returns:
        Normalized array in [-1, 1] range with float64 dtype
    """
    # Convert to float64 to ensure we can represent values in [-1, 1]
    array_float = array.astype(np.float64)

    format_info = detect_format(array)
    channel_location = format_info["channel_location"]
    
    # Normalize all channels using the same strategy
    if channel_location == "first":
        min_vals = np.min(array_float, axis=(2, 3), keepdims=True)
        max_vals = np.max(array_float, axis=(2, 3), keepdims=True)
    else:  # channel_location is None or "last"
        min_vals = np.min(array_float, axis=(1, 2), keepdims=True)
        max_vals = np.max(array_float, axis=(1, 2), keepdims=True)
    
    range_vals = max_vals - min_vals
    return np.where(
        range_vals > 0,
        2.0 * (array_float - min_vals) / range_vals - 1.0,
        0.0
    )


def _normalize_imagenet_stats(array: np.ndarray) -> np.ndarray:
    """
    Normalize array using ImageNet statistics.
    
    Uses mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225] for RGB channels (standard imagenet values).
    For RGBA images, alpha channel is normalized using z-score (its own mean/std) to match the normalization strategy.
    Assumes RGB format with channels last (N, H, W, 3) or channels first (N, 3, H, W).
    Formula: (x - mean) / std, applied per channel.
    
    Automatically normalizes input to [0, 1] range first if values are > 1 (e.g., uint8 [0, 255]),
    then applies ImageNet statistics. This makes it consistent with other normalization modes.
    
    Args:
        array: Input array with batch dimension
    
    Returns:
        Normalized array using ImageNet statistics, float64 dtype
    """
    # Convert to float64 for precision
    array_float = array.astype(np.float64)

    format_info = detect_format(array)
    channel_location = format_info["channel_location"]
    
    # ImageNet statistics (per RGB channel)
    imagenet_mean = np.array([0.485, 0.456, 0.406], dtype=np.float64)
    imagenet_std = np.array([0.229, 0.224, 0.225], dtype=np.float64)
    
    # Check if this is RGBA (4 channels)
    is_rgba = False
    if channel_location == "last" and array.shape[3] == 4:
        is_rgba = True
    elif channel_location == "first" and array.shape[1] == 4:
        is_rgba = True
    
    # Auto-normalize to [0, 1] if values are outside expected [0, 1] range
    array_max = array_float.max()
    array_min = array_float.min()
    
    if array_min < 0.0:
        # Values are < 0, likely in [-1, 1] range
        if array_max <= 1.0:
            # Scale from [-1, 1] to [0, 1]: (x + 1) / 2
            array_float = (array_float + 1.0) / 2.0
        else:
            # If max > 1 and min < 0, weird range - just divide by max
            array_float = array_float / array_max
    elif array_max > 1.0:
        # Values are > 1, need to normalize
        if np.issubdtype(array.dtype, np.integer):
            # Integer types: divide by dtype max (e.g., uint8 -> divide by 255)
            dtype_max = np.iinfo(array.dtype).max
            array_float = array_float / dtype_max
        else:
            # Float types: assume [0, 255] range and divide by 255
            array_float = array_float / 255.0
    
    if is_rgba:
        # For RGBA: apply ImageNet stats to RGB, z-score to alpha
        if channel_location == "last":
            rgb_channels = array_float[..., :3]
            alpha_channel = array_float[..., 3:4]
            
            # Apply ImageNet stats to RGB
            mean_broadcast = imagenet_mean.reshape(1, 1, 1, 3)
            std_broadcast = imagenet_std.reshape(1, 1, 1, 3)
            normalized_rgb = (rgb_channels - mean_broadcast) / std_broadcast
            
            # Normalize alpha using z-score (same strategy as RGB, but with alpha's own stats)
            alpha_mean = np.mean(alpha_channel, axis=(1, 2), keepdims=True)
            alpha_std = np.std(alpha_channel, axis=(1, 2), keepdims=True)
            normalized_alpha = np.where(
                alpha_std > 0,
                (alpha_channel - alpha_mean) / alpha_std,
                0.0
            )
            
            return np.concatenate([normalized_rgb, normalized_alpha], axis=-1)
        else:  # channels first
            rgb_channels = array_float[:, :3, ...]
            alpha_channel = array_float[:, 3:4, ...]
            
            # Apply ImageNet stats to RGB
            mean_broadcast = imagenet_mean.reshape(1, 3, 1, 1)
            std_broadcast = imagenet_std.reshape(1, 3, 1, 1)
            normalized_rgb = (rgb_channels - mean_broadcast) / std_broadcast
            
            # Normalize alpha using z-score (same strategy as RGB, but with alpha's own stats)
            alpha_mean = np.mean(alpha_channel, axis=(2, 3), keepdims=True)
            alpha_std = np.std(alpha_channel, axis=(2, 3), keepdims=True)
            normalized_alpha = np.where(
                alpha_std > 0,
                (alpha_channel - alpha_mean) / alpha_std,
                0.0
            )
            
            return np.concatenate([normalized_rgb, normalized_alpha], axis=1)
    
    # For RGB: apply ImageNet stats
    if channel_location == "last":
        # (N, H, W, 3) - broadcast mean/std to (1, 1, 1, 3)
        mean_broadcast = imagenet_mean.reshape(1, 1, 1, 3)
        std_broadcast = imagenet_std.reshape(1, 1, 1, 3)
        return (array_float - mean_broadcast) / std_broadcast
    elif channel_location == "first":
        # (N, 3, H, W) - broadcast mean/std to (1, 3, 1, 1)
        mean_broadcast = imagenet_mean.reshape(1, 3, 1, 1)
        std_broadcast = imagenet_std.reshape(1, 3, 1, 1)
        return (array_float - mean_broadcast) / std_broadcast
    else:
        # Grayscale (N, H, W) - ImageNet stats don't apply, raise error
        raise ValueError(
            "ImageNet normalization requires RGB images. "
            f"Got grayscale array with shape {array.shape}. "
            "Use ensure_color_mode() to convert to RGB first."
        )


def _normalize_zscore(array: np.ndarray) -> np.ndarray:
    """
    Normalize array using z-score normalization (mean=0, std=1).
    
    Normalizes each image in the batch independently by standardizing values so that
    the mean becomes 0 and the standard deviation becomes 1. Formula: (x - mean) / std.
    
    If an image has constant values (std == 0), it will be set to zeros.
    All channels (including alpha) are normalized using the same z-score strategy.
    
    Args:
        array: Input array with batch dimension
    
    Returns:
        Standardized array with mean=0 and std=1, float64 dtype
    """
    # Convert to float64 for precision in mean/std calculations
    array_float = array.astype(np.float64)

    format_info = detect_format(array)
    channel_location = format_info["channel_location"]
    
    # Normalize all channels using the same strategy
    if channel_location == "first":
        mean_vals = np.mean(array_float, axis=(2, 3), keepdims=True)
        std_vals = np.std(array_float, axis=(2, 3), keepdims=True)
    else:  # channel_location is None or "last"
        mean_vals = np.mean(array_float, axis=(1, 2), keepdims=True)
        std_vals = np.std(array_float, axis=(1, 2), keepdims=True)
    
    # Normalize: (x - mean) / std, handle std=0 case
    return np.where(
        std_vals > 0,
        (array_float - mean_vals) / std_vals,
        0.0
    )
