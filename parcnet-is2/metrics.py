import torch
import numpy as np
from torch import Tensor
from typing import Union


def _to_numpy(y: Union[Tensor, np.ndarray]) -> np.ndarray:
    if isinstance(y, Tensor):
        y = y.detach().cpu().numpy()
    return y.squeeze()


def mse(y_pred: Union[Tensor, np.ndarray], y_true: Union[Tensor, np.ndarray]) -> np.ndarray:
    y_true = _to_numpy(y_true)
    y_pred = _to_numpy(y_pred)

    return 10 * np.log10(np.mean(np.square(y_true - y_pred)))


def sdr(y_pred: Union[Tensor, np.ndarray], y_true: Union[Tensor, np.ndarray]):
    y_true = _to_numpy(y_true)
    y_pred = _to_numpy(y_pred)

    num = np.linalg.norm(y_true) ** 2 + 1e-7
    den = np.linalg.norm(y_true - y_pred) ** 2 + 1e-7

    return 10 * np.log10(num / den)
