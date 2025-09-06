import cv2 as cv
import numpy as np
from typing import Union, Tuple, Sequence, Optional
import plotly.graph_objects as go


def show_image(image: np.ndarray):
    """
    Show an image using plotly.
    :param image: input image
    :return: None
    """
    h, w = image.shape[:2]
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    fig = go.Figure(go.Image(z=image))
    fig.update_layout(width=w, height=h)
    fig.show()


def _SSR(channel: np.ndarray, size: Union[int, Tuple[int, int]]) -> np.ndarray:
    """
    Single Scale Retinex (SSR) algorithm.
    :param channel: input channel
    :param size: kernel size for gaussian filter
    :return: output channel after SSR
    """
    size = (size, size) if isinstance(size, int) else size
    eps = 1e-6
    I = channel.astype(np.float32)
    L = cv.GaussianBlur(channel, size, 0).astype(np.float32)
    r = np.log(I + eps) - np.log(L + eps)

    min_r, max_r, _, _ = cv.minMaxLoc(r)
    R = (r - min_r) / (max_r - min_r) * 255
    R = np.clip(R, 0, 255).astype(np.uint8)
    R = cv.convertScaleAbs(R)
    output = cv.add(channel, R)
    return output


def SSR(image: np.ndarray, size: Union[int, Tuple[int, int]] = 3) -> np.ndarray:
    if len(image.shape) == 3:
        b, g, r = cv.split(image)
        enhanced_b = _SSR(b, size)
        enhanced_g = _SSR(g, size)
        enhanced_r = _SSR(r, size)
        enhanced_image = cv.merge([enhanced_b, enhanced_g, enhanced_r])
    else:
        enhanced_image = _SSR(image, size)

    return enhanced_image


def _MSR(channel: np.ndarray, scales: Sequence[int], weights: Optional[Sequence[float]] = None) -> np.ndarray:
    """
    Multi-Scale Retinex (MSR) algorithm.
    :param channel: input channel
    :param scales: list of kernel sizes for gaussian filter
    :param weights: list of weights for each scale
    :return: output channel after MSR
    """
    eps = 1e-6
    if weights is None:
        weights = np.ones(len(scales)) / len(scales)

    I = channel.astype(np.float32)
    h, w = I.shape[:2]
    r = np.zeros((h, w), dtype=np.float32)
    R = np.zeros((h, w), dtype=np.float32)
    for i, size in enumerate(scales):
        L = cv.GaussianBlur(I, (size, size), 0).astype(np.float32)
        r += (np.log(I + eps) - np.log(L + eps)) * weights[i]

    cv.normalize(r, R, 0, 255, cv.NORM_MINMAX)
    R = cv.convertScaleAbs(R).astype(np.uint8)
    output = cv.add(channel, R)
    return output


def MSR(image: np.ndarray, scales: Sequence[int], weights: Optional[Sequence[float]] = None) -> np.ndarray:
    if len(image.shape) == 3:
        b, g, r = cv.split(image)
        enhanced_b = _MSR(b, scales, weights)
        enhanced_g = _MSR(g, scales, weights)
        enhanced_r = _MSR(r, scales, weights)
        enhanced_image = cv.merge([enhanced_b, enhanced_g, enhanced_r])
    else:
        enhanced_image = _MSR(image, scales, weights)

    return enhanced_image


def color_balance(image: np.ndarray, low_clip: float, high_clip: float) -> np.ndarray:
    """
    Color balance algorithm.
    :param image: input image
    :param low_clip: low clip ratio
    :param high_clip: high clip ratio
    :return: output image after color balance
    """
    assert len(image.shape) == 3, "Input image must be a 3-channel image."

    total = image.shape[0] * image.shape[1]
    for i in range(image.shape[2]):
        unique, counts = np.unique(image[:, :, i], return_counts=True)
        current = 0
        low_val = 0
        high_val = 255
        for u, c in zip(unique, counts):
            if current / total < low_clip:
                low_val = u
            if current / total < high_clip:
                high_val = u
            current += c

        image[:, :, i] = np.maximum(np.minimum(image[:, :, i], high_val), low_val)

    return image


def MSRCR(image: np.ndarray, scales: Sequence[int], weights: Optional[Sequence[float]] = None,
          gain: float = 5.0, bias: float = 25.0, alpha: float = 125.0, beta: float = 46.0,
          low_clip: float = 0.01, high_clip: float = 0.99) -> np.ndarray:
    """
    Multi-Scale Retinex with Color Restoration (MSRCR) algorithm.
    :param image: input image
    :param scales: list of kernel sizes for gaussian filter
    :param weights: list of weights for each scale
    :param gain: gain factor
    :param bias: bias factor
    :param alpha: intensity factor
    :param beta: gain factor
    :param low_clip: low clip ratio
    :param high_clip: high clip ratio
    :return: output image after color balance
    """
    if weights is None:
        weights = np.ones(len(scales)) / len(scales)

    I = image.astype(np.float32) + 1.0
    r = np.zeros_like(I)
    for i, size in enumerate(scales):
        L = cv.GaussianBlur(I, (size, size), 0).astype(np.float32)
        r += (np.log(I + 1.0) - np.log(L + 1.0)) * weights[i]

    image_color = beta * np.log(alpha * I / np.sum(I, axis=2, keepdims=True))
    image_msrcr = gain * (r * image_color + bias)

    for i in range(image_msrcr.shape[2]):
        min_value, max_value, _, _ = cv.minMaxLoc(image_msrcr[:, :, i])
        image_msrcr[:, :, i] = (image_msrcr[:, :, i] - min_value) / (max_value - min_value) * 255

    image_msrcr = np.clip(image_msrcr, 0, 255).astype(np.uint8)
    output = color_balance(image_msrcr, low_clip, high_clip)

    return output


def AUTOMSRCR(image: np.ndarray, scales: Sequence[int], weights: Optional[Sequence[float]] = None) -> np.ndarray:
    if weights is None:
        weights = np.ones(len(scales)) / len(scales)

    I = image.astype(np.float32) + 1.0
    r = np.zeros_like(I)
    for i, size in enumerate(scales):
        L = cv.GaussianBlur(I, (size, size), 0).astype(np.float32)
        r += (np.log(I + 1.0) - np.log(L + 1.0)) * weights[i]

    for i in range(r.shape[2]):
        zero_count = int(np.float32(r[:, :, i] == 0).sum())
        unique, counts = np.unique(r[:, :, i], return_counts=True)

        low_val = unique[0] / 100.
        high_val = unique[-1] / 100.
        for u, c in zip(unique, counts):
            if u < 0 and c < zero_count * 0.1:
                low_val = u / 100.
            if u > 0 and c < zero_count * 0.1:
                high_val = u / 100.
                break

        r[:, :, i] = np.maximum(np.minimum(r[:, :, i], high_val), low_val)
        min_value, max_value, _, _ = cv.minMaxLoc(r[:, :, i])
        r[:, :, i] = (r[:, :, i] - min_value) / (max_value - min_value) * 255

    r = np.clip(r, 0, 255).astype(np.uint8)
    return r


if __name__ == '__main__':
    img = cv.imread("/home/nju-student/mkh/datasets/LLIE/LOLv1/our485/low/9.png")
    scales = [15, 81, 201]
    enhanced_img = MSRCR(img, scales)
    temp = np.hstack((img, enhanced_img))
    show_image(temp)