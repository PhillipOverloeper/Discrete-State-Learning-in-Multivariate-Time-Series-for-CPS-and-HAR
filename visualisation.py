import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap

from VQ_VAE import vq_vae_visualisation_setup
from wadi import WADI_Dataloader

# =========================
# Utilities
# =========================


def _infer_num_states(*ys: np.ndarray) -> int:
    m = 0
    for y in ys:
        y_flat = np.asarray(y).reshape(-1).astype(int)
        if y_flat.size:
            m = max(m, int(np.max(y_flat)))
    return m + 1 if m >= 0 else 0


def _default_cmap(n_states: int) -> ListedColormap:
    base = plt.get_cmap("tab20").colors
    colors = [base[i % len(base)] for i in range(max(n_states, 1))]
    return ListedColormap(colors)


def _infer_num_states_multi(*ys: np.ndarray) -> int:
    m = -1
    for y in ys:
        y_flat = np.asarray(y).reshape(-1).astype(int)
        if y_flat.size:
            m = max(m, int(np.max(y_flat)))
    return m + 1 if m >= 0 else 0


def _default_cmap(n_states: int) -> ListedColormap:
    base = plt.get_cmap("tab20").colors
    colors = [base[i % len(base)] for i in range(max(n_states, 1))]
    return ListedColormap(colors)


def plot_window_data_true_vs_learned_aligned(
    X: np.ndarray,
    y_true: np.ndarray,
    y_hat: np.ndarray,
    window_idx: int = 0,
    figsize: Tuple[float, float] = (12, 6),
    cmap_labels: Optional[ListedColormap] = None,
    show_colorbar: bool = True,
) -> plt.Figure:
    """
    3-panel plot with guaranteed alignment:
      - Top:  feature heatmap (features x time)
      - Mid:  true labels ribbon
      - Bot:  learned labels ribbon

    Colorbar goes into its own dedicated column so all panels align.
    """
    if X.ndim != 3:
        raise ValueError(f"X must be 3D [n_windows, T, D], got {X.shape}")
    if y_true.ndim != 2 or y_hat.ndim != 2:
        raise ValueError(f"y_true and y_hat must be 2D [n_windows, T]. Got {y_true.shape}, {y_hat.shape}")
    if X.shape[0] != y_true.shape[0] or X.shape[1] != y_true.shape[1]:
        raise ValueError(f"Shape mismatch: X {X.shape} vs y_true {y_true.shape}")
    if y_hat.shape != y_true.shape:
        raise ValueError(f"Shape mismatch: y_hat {y_hat.shape} must match y_true {y_true.shape}")

    W, T, D = X.shape
    if not (0 <= window_idx < W):
        raise IndexError(f"window_idx={window_idx} out of range for {W} windows")

    xw = X[window_idx]                  # [T, D]
    yt = y_true[window_idx].astype(int) # [T]
    yh = y_hat[window_idx].astype(int)  # [T]

    n_states = _infer_num_states_multi(y_true, y_hat)
    if cmap_labels is None:
        cmap_labels = _default_cmap(n_states)

    fig = plt.figure(figsize=figsize)

    # 3 rows x 2 cols: right column is reserved for the colorbar
    gs = fig.add_gridspec(
        3, 2,
        width_ratios=[1.0, 0.04],
        height_ratios=[3.0, 0.35, 0.35],
        wspace=0.10,
        hspace=0.12
    )

    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[1, 0], sharex=ax0)
    ax2 = fig.add_subplot(gs[2, 0], sharex=ax0)

    # Dedicated colorbar axis (spans only top row)
    cax = fig.add_subplot(gs[0, 1]) if show_colorbar else None

    # --- Top: heatmap
    im = ax0.imshow(
        xw.T, aspect="auto", origin="lower", interpolation="nearest"
    )
    ax0.set_title(f"Window {window_idx}: data (top), true labels (mid), learned labels (bottom)")
    ax0.set_ylabel("Feature index")
    ax0.tick_params(axis="x", labelbottom=False)

    if show_colorbar:
        fig.colorbar(im, cax=cax)

    # --- True labels ribbon
    ax1.imshow(
        yt[np.newaxis, :],
        aspect="auto",
        cmap=cmap_labels,
        interpolation="nearest",
        origin="lower",
        vmin=0,
        vmax=max(n_states - 1, 0),
    )
    ax1.set_yticks([])
    ax1.tick_params(axis="x", labelbottom=False)
    # Use text instead of ylabel (avoids changing layout width)
    ax1.text(-0.01, 0.5, "y", transform=ax1.transAxes, va="center", ha="right")

    # --- Learned labels ribbon
    ax2.imshow(
        yh[np.newaxis, :],
        aspect="auto",
        cmap=cmap_labels,
        interpolation="nearest",
        origin="lower",
        vmin=0,
        vmax=max(n_states - 1, 0),
    )
    ax2.set_yticks([])
    ax2.set_xlabel("Time index within window")
    ax2.text(-0.01, 0.5, "ŷ", transform=ax2.transAxes, va="center", ha="right")

    fig.tight_layout()
    return fig


def plot_window_data_true_vs_learned(
    X: np.ndarray,
    y_true: np.ndarray,
    y_hat: np.ndarray,
    window_idx: int = 0,
    figsize: Tuple[float, float] = (12, 6),
    cmap_labels: Optional[ListedColormap] = None,
    show_colorbar: bool = True,
) -> plt.Figure:
    """
    Top:  Feature heatmap (features x time)
    Mid:  True label ribbon
    Bot:  Learned/predicted label ribbon
    """
    if X.ndim != 3:
        raise ValueError(f"X must be 3D [n_windows, T, D], got {X.shape}")
    if y_true.ndim != 2 or y_hat.ndim != 2:
        raise ValueError(f"y_true and y_hat must be 2D [n_windows, T]. Got {y_true.shape}, {y_hat.shape}")
    if X.shape[0] != y_true.shape[0] or X.shape[1] != y_true.shape[1]:
        raise ValueError(f"Shape mismatch: X {X.shape} vs y_true {y_true.shape}")
    if y_hat.shape != y_true.shape:
        raise ValueError(f"Shape mismatch: y_hat {y_hat.shape} must match y_true {y_true.shape}")

    W, T, D = X.shape
    if not (0 <= window_idx < W):
        raise IndexError(f"window_idx={window_idx} out of range for {W} windows")

    xw = X[window_idx]                 # [T, D]
    yt = y_true[window_idx].astype(int)  # [T]
    yh = y_hat[window_idx].astype(int)   # [T]

    n_states = _infer_num_states(y_true, y_hat)
    if cmap_labels is None:
        cmap_labels = _default_cmap(n_states)

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 1, height_ratios=[3.0, 0.35, 0.35], hspace=0.10)

    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1], sharex=ax0)
    ax2 = fig.add_subplot(gs[2], sharex=ax0)

    # --- Top: heatmap features x time
    im = ax0.imshow(
        xw.T, aspect="auto", origin="lower", interpolation="nearest"
    )
    ax0.set_title(f"Window {window_idx}: data (top), true labels (mid), learned labels (bottom)")
    ax0.set_ylabel("Feature index")
    ax0.tick_params(axis="x", labelbottom=False)

    if show_colorbar:
        plt.colorbar(im, ax=ax0, fraction=0.02, pad=0.02)

    # --- Middle: true labels ribbon
    ax1.imshow(
        yt[np.newaxis, :],
        aspect="auto",
        cmap=cmap_labels,
        interpolation="nearest",
        origin="lower",
        vmin=0,
        vmax=max(n_states - 1, 0),
    )
    ax1.set_yticks([])
    ax1.set_ylabel("y", rotation=0, labelpad=15, va="center")
    ax1.tick_params(axis="x", labelbottom=False)

    # --- Bottom: learned labels ribbon
    ax2.imshow(
        yh[np.newaxis, :],
        aspect="auto",
        cmap=cmap_labels,
        interpolation="nearest",
        origin="lower",
        vmin=0,
        vmax=max(n_states - 1, 0),
    )
    ax2.set_yticks([])
    ax2.set_ylabel("ŷ", rotation=0, labelpad=15, va="center")
    ax2.set_xlabel("Time index within window")

    fig.tight_layout()
    return fig

def plot_multiple_windows_true_vs_learned_one_figure(
    X: np.ndarray,
    y_true: np.ndarray,
    y_hat: np.ndarray,
    window_indices: Optional[Sequence[int]] = None,
    k: int = 8,
    seed: Optional[int] = None,
    replace: bool = False,
    figsize: Optional[Tuple[float, float]] = None,
    cmap_labels: Optional[ListedColormap] = None,
    show_colorbar: bool = True,
) -> plt.Figure:
    """
    One figure, multiple windows:
      Columns: windows
      Rows:    data heatmap, y true ribbon, y_hat ribbon

    Uses one shared colorbar column to keep all axes aligned.
    """
    if X.ndim != 3:
        raise ValueError(f"X must be 3D [n_windows, T, D], got {X.shape}")
    if y_true.ndim != 2 or y_hat.ndim != 2:
        raise ValueError(f"y_true and y_hat must be 2D [n_windows, T]. Got {y_true.shape}, {y_hat.shape}")
    if X.shape[0] != y_true.shape[0] or X.shape[1] != y_true.shape[1]:
        raise ValueError(f"Shape mismatch: X {X.shape} vs y_true {y_true.shape}")
    if y_hat.shape != y_true.shape:
        raise ValueError(f"Shape mismatch: y_hat {y_hat.shape} must match y_true {y_true.shape}")

    W, T, D = X.shape
    rng = np.random.default_rng(seed)

    # Choose windows
    if window_indices is None:
        pool = np.arange(W)
    else:
        pool = np.array(list(window_indices), dtype=int)
        if pool.size == 0:
            raise ValueError("window_indices is empty")
        if pool.min() < 0 or pool.max() >= W:
            raise ValueError(f"window_indices must be within [0, {W-1}]")

    if (not replace) and k > pool.size:
        raise ValueError(f"k={k} larger than available windows {pool.size} with replace=False")

    chosen = rng.choice(pool, size=k, replace=replace).astype(int)

    # Colormap for labels
    n_states = _infer_num_states_multi(y_true, y_hat)
    if cmap_labels is None:
        cmap_labels = _default_cmap(n_states)

    # Set a reasonable figsize if not provided
    if figsize is None:
        # width grows with number of windows
        figsize = (max(12, 2.3 * k), 6)

    fig = plt.figure(figsize=figsize)

    # Outer GridSpec: 3 rows, (k columns + optional colorbar column)
    ncols_total = k + (1 if show_colorbar else 0)
    width_ratios = [1.0] * k + ([0.04] if show_colorbar else [])

    gs = fig.add_gridspec(
        3, ncols_total,
        width_ratios=width_ratios,
        height_ratios=[3.0, 0.35, 0.35],
        wspace=0.10,
        hspace=0.12
    )

    # Use a shared scale for the heatmap color range for comparability
    # (avoid each heatmap having arbitrary contrast)
    X_sel = X[chosen]  # [k, T, D]
    vmin = float(np.min(X_sel))
    vmax = float(np.max(X_sel))

    heatmap_axes = []
    im_for_colorbar = None

    for j, w in enumerate(chosen):
        xw = X[w]                     # [T, D]
        yt = y_true[w].astype(int)    # [T]
        yh = y_hat[w].astype(int)     # [T]

        ax0 = fig.add_subplot(gs[0, j])
        ax1 = fig.add_subplot(gs[1, j], sharex=ax0)
        ax2 = fig.add_subplot(gs[2, j], sharex=ax0)

        # Heatmap (features x time)
        im = ax0.imshow(
            xw.T,
            aspect="auto",
            origin="lower",
            interpolation="nearest",
            vmin=vmin,
            vmax=vmax
        )
        if im_for_colorbar is None:
            im_for_colorbar = im
        heatmap_axes.append(ax0)

        ax0.set_title(f"w={w}", fontsize=10)
        ax0.tick_params(axis="x", labelbottom=False)
        if j == 0:
            ax0.set_ylabel("Feature")
        else:
            ax0.set_yticks([])

        # True ribbon
        ax1.imshow(
            yt[np.newaxis, :],
            aspect="auto",
            cmap=cmap_labels,
            interpolation="nearest",
            origin="lower",
            vmin=0,
            vmax=max(n_states - 1, 0),
        )
        ax1.set_yticks([])
        ax1.tick_params(axis="x", labelbottom=False)
        if j == 0:
            ax1.text(-0.01, 0.5, "y", transform=ax1.transAxes, va="center", ha="right")

        # Learned ribbon
        ax2.imshow(
            yh[np.newaxis, :],
            aspect="auto",
            cmap=cmap_labels,
            interpolation="nearest",
            origin="lower",
            vmin=0,
            vmax=max(n_states - 1, 0),
        )
        ax2.set_yticks([])
        ax2.set_xlabel("t" if j == k // 2 else "")
        if j == 0:
            ax2.text(-0.01, 0.5, "ŷ", transform=ax2.transAxes, va="center", ha="right")

    # Shared colorbar column on the right
    if show_colorbar:
        cax = fig.add_subplot(gs[0, -1])
        fig.colorbar(im_for_colorbar, cax=cax)

        # Turn off unused axes below the colorbar (rows 1 and 2 in last column)
        fig.add_subplot(gs[1, -1]).axis("off")
        fig.add_subplot(gs[2, -1]).axis("off")

    fig.suptitle("Multiple windows: data + true labels + learned labels", y=1.02)
    fig.tight_layout()
    return fig




def plot_random_windows_true_vs_learned(
    X: np.ndarray,
    y_true: np.ndarray,
    y_hat: np.ndarray,
    k: int = 8,
    seed: Optional[int] = None,
    replace: bool = False,
    window_indices: Optional[Sequence[int]] = None,
    show: bool = True,
) -> List[plt.Figure]:
    """
    Randomly sample k windows and plot:
      - data heatmap (top)
      - true label ribbon (mid)
      - learned label ribbon (bottom)

    Parameters
    ----------
    X : np.ndarray
        Shape (n_windows, T, D)
    y_true : np.ndarray
        Shape (n_windows, T)
    y_hat : np.ndarray
        Shape (n_windows, T)
    k : int
        Number of windows to visualize
    seed : int or None
        Random seed for reproducibility
    replace : bool
        Sample with replacement (useful if k > n_windows)
    window_indices : sequence or None
        If provided, sample only from this subset of window indices
    show : bool
        If True, calls plt.show() after creating all figures

    Returns
    -------
    figs : list of matplotlib Figure
        The created figures (one per sampled window)
    """
    if X.ndim != 3:
        raise ValueError(f"X must be 3D [n_windows, T, D], got {X.shape}")
    if y_true.ndim != 2 or y_hat.ndim != 2:
        raise ValueError(f"y_true and y_hat must be 2D [n_windows, T]. Got {y_true.shape}, {y_hat.shape}")
    if X.shape[0] != y_true.shape[0] or X.shape[1] != y_true.shape[1]:
        raise ValueError(f"Shape mismatch: X {X.shape} vs y_true {y_true.shape}")
    if y_hat.shape != y_true.shape:
        raise ValueError(f"Shape mismatch: y_hat {y_hat.shape} must match y_true {y_true.shape}")

    W = X.shape[0]
    rng = np.random.default_rng(seed)

    if window_indices is None:
        pool = np.arange(W)
    else:
        pool = np.array(list(window_indices), dtype=int)
        if pool.size == 0:
            raise ValueError("window_indices is empty")
        if pool.min() < 0 or pool.max() >= W:
            raise ValueError(f"window_indices must be within [0, {W-1}]")

    if (not replace) and k > pool.size:
        raise ValueError(f"k={k} is larger than available windows ({pool.size}) with replace=False")

    chosen = rng.choice(pool, size=k, replace=replace)

    figs = []
    for w in chosen:
        fig = plot_window_data_true_vs_learned(X, y_true, y_hat, window_idx=int(w))
        figs.append(fig)

    if show:
        plt.show()

    return figs


def _check_shapes(X: np.ndarray, y: np.ndarray) -> None:
    if X.ndim != 3:
        raise ValueError(f"X must be 3D [n_windows, window_length, n_features], got shape {X.shape}")
    if y.ndim != 2:
        raise ValueError(f"y must be 2D [n_windows, window_length], got shape {y.shape}")
    if X.shape[0] != y.shape[0] or X.shape[1] != y.shape[1]:
        raise ValueError(
            f"X and y mismatch: X {X.shape}, y {y.shape}. Need X[:2] == y."
        )


def _label_boundaries(y_1d: np.ndarray) -> np.ndarray:
    """Return indices t where y[t] != y[t-1] (transition points)."""
    y_1d = np.asarray(y_1d)
    if y_1d.size < 2:
        return np.array([], dtype=int)
    return np.where(y_1d[1:] != y_1d[:-1])[0] + 1


def _runs(y_1d: np.ndarray) -> List[Tuple[int, int, int]]:
    """
    Run-length encoding for labels.
    Returns list of (label, start_idx, length).
    """
    y_1d = np.asarray(y_1d)
    if y_1d.size == 0:
        return []

    runs = []
    start = 0
    current = y_1d[0]
    for t in range(1, len(y_1d)):
        if y_1d[t] != current:
            runs.append((int(current), start, t - start))
            start = t
            current = y_1d[t]
    runs.append((int(current), start, len(y_1d) - start))
    return runs


def _window_summary_features(
    X: np.ndarray,
    agg: str = "mean",
) -> np.ndarray:
    """
    Compute window-level feature vectors from time series per window.
    Returns shape [n_windows, n_features].
    """
    if agg == "mean":
        return X.mean(axis=1)
    if agg == "std":
        return X.std(axis=1)
    if agg == "min":
        return X.min(axis=1)
    if agg == "max":
        return X.max(axis=1)
    raise ValueError(f"Unknown agg={agg}. Use one of: mean, std, min, max.")


@dataclass
class VizConfig:
    num_states: Optional[int] = None
    cmap: Optional[ListedColormap] = None
    alpha_label_band: float = 0.25
    figsize: Tuple[float, float] = (12, 4)

    def resolve(self, y: np.ndarray):
        n_states = self.num_states if self.num_states is not None else _infer_num_states(y)
        cmap = self.cmap if self.cmap is not None else _default_cmap(n_states)
        return n_states, cmap


# =========================
# 1) Single-window traces + label overlay
# =========================

def plot_window_traces_with_labels(
    X: np.ndarray,
    y: np.ndarray,
    window_idx: int = 0,
    feature_indices: Optional[Sequence[int]] = None,
    feature_names: Optional[Sequence[str]] = None,
    show_boundaries: bool = True,
    config: VizConfig = VizConfig(),
) -> plt.Figure:
    """
    Plot selected feature traces for one window, with a label band behind.
    """
    _check_shapes(X, y)
    n_states, cmap = config.resolve(y)

    W, T, D = X.shape
    if window_idx < 0 or window_idx >= W:
        raise IndexError(f"window_idx out of range: {window_idx} for W={W}")

    if feature_indices is None:
        # default: up to 5 features
        feature_indices = list(range(min(5, D)))
    feature_indices = list(feature_indices)

    xw = X[window_idx]  # [T, D]
    yw = y[window_idx]  # [T]
    t = np.arange(T)

    fig, ax = plt.subplots(figsize=config.figsize)

    # label background as colored spans
    ax.imshow(
        yw[np.newaxis, :],
        aspect="auto",
        extent=[0, T - 1, np.min(xw[:, feature_indices]), np.max(xw[:, feature_indices])],
        cmap=cmap,
        alpha=config.alpha_label_band,
        interpolation="nearest",
        origin="lower",
    )

    for k, fi in enumerate(feature_indices):
        name = f"feat_{fi}" if feature_names is None else feature_names[fi]
        ax.plot(t, xw[:, fi], label=name)

    if show_boundaries:
        b = _label_boundaries(yw)
        for bb in b:
            ax.axvline(bb, linewidth=1, alpha=0.4)

    ax.set_title(f"Window {window_idx}: Feature traces with label overlay")
    ax.set_xlabel("Time index within window")
    ax.set_ylabel("Feature value")
    ax.legend(loc="best", frameon=True)
    fig.tight_layout()
    return fig


# =========================
# 2) Window heatmap [time x features] + label ribbon
# =========================

def plot_window_heatmap_with_label_ribbon(
    X: np.ndarray,
    y: np.ndarray,
    window_idx: int = 0,
    config: VizConfig = VizConfig(figsize=(12, 5)),
) -> plt.Figure:
    """
    Heatmap of one window: X[time, features] with a label strip.
    """
    _check_shapes(X, y)
    n_states, cmap = config.resolve(y)

    xw = X[window_idx]  # [T, D]
    yw = y[window_idx]  # [T]
    T, D = xw.shape

    fig = plt.figure(figsize=config.figsize)
    gs = fig.add_gridspec(2, 1, height_ratios=[0.25, 3], hspace=0.05)

    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])

    ax0.imshow(yw[np.newaxis, :], aspect="auto", cmap=cmap, interpolation="nearest")
    ax0.set_yticks([])
    ax0.set_xticks([])
    ax0.set_title(f"Window {window_idx}: label ribbon + feature heatmap")

    im = ax1.imshow(
        xw.T,  # [D, T]
        aspect="auto",
        interpolation="nearest",
        origin="lower",
    )
    ax1.set_xlabel("Time index within window")
    ax1.set_ylabel("Feature index")
    plt.colorbar(im, ax=ax1, fraction=0.02, pad=0.02)

    fig.tight_layout()
    return fig


# =========================
# 3) Label raster across windows [windows x time]
# =========================

def plot_label_raster_all_windows(
    y: np.ndarray,
    config: VizConfig = VizConfig(figsize=(12, 6)),
) -> plt.Figure:
    """
    Visualizes labels as an image: rows=windows, cols=time.
    """
    if y.ndim != 2:
        raise ValueError(f"y must be 2D [n_windows, window_length], got {y.shape}")
    n_states, cmap = config.resolve(y)

    fig, ax = plt.subplots(figsize=config.figsize)
    ax.imshow(y, aspect="auto", cmap=cmap, interpolation="nearest", origin="lower")
    ax.set_title("Label raster: windows × time")
    ax.set_xlabel("Time index within window")
    ax.set_ylabel("Window index")
    fig.tight_layout()
    return fig


# =========================
# 4) Label occupancy per window
# =========================

def compute_label_occupancy_per_window(
    y: np.ndarray,
    num_states: Optional[int] = None,
) -> np.ndarray:
    """
    Returns occupancy matrix shape [n_windows, num_states], fractions summing to 1 per row (if T>0).
    """
    if y.ndim != 2:
        raise ValueError(f"y must be 2D, got {y.shape}")

    W, T = y.shape
    if num_states is None:
        num_states = _infer_num_states(y)

    occ = np.zeros((W, num_states), dtype=float)
    for w in range(W):
        for s in range(num_states):
            occ[w, s] = np.mean(y[w] == s) if T > 0 else 0.0
    return occ


def plot_label_occupancy_per_window(
    y: np.ndarray,
    config: VizConfig = VizConfig(figsize=(12, 6)),
) -> plt.Figure:
    """
    Heatmap of occupancy fractions: [windows x states].
    """
    if y.ndim != 2:
        raise ValueError(f"y must be 2D, got {y.shape}")
    n_states, _ = config.resolve(y)
    occ = compute_label_occupancy_per_window(y, num_states=n_states)

    fig, ax = plt.subplots(figsize=config.figsize)
    im = ax.imshow(occ, aspect="auto", origin="lower", interpolation="nearest")
    ax.set_title("State occupancy per window")
    ax.set_xlabel("State label")
    ax.set_ylabel("Window index")
    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    fig.tight_layout()
    return fig


# =========================
# 5) Global label distribution
# =========================

def plot_global_label_distribution(
    y: np.ndarray,
    config: VizConfig = VizConfig(figsize=(10, 4)),
) -> plt.Figure:
    """
    Bar plot of label counts across all windows/timepoints.
    """
    if y.ndim != 2:
        raise ValueError(f"y must be 2D, got {y.shape}")
    n_states, cmap = config.resolve(y)

    y_flat = y.reshape(-1).astype(int)
    counts = np.array([(y_flat == s).sum() for s in range(n_states)], dtype=int)

    fig, ax = plt.subplots(figsize=config.figsize)
    ax.bar(np.arange(n_states), counts)
    ax.set_title("Global label distribution")
    ax.set_xlabel("State label")
    ax.set_ylabel("Count")
    fig.tight_layout()
    return fig


# =========================
# 6) Feature-by-label distributions (box plots)
# =========================

def plot_feature_distributions_by_label(
    X: np.ndarray,
    y: np.ndarray,
    feature_idx: int = 0,
    sample_max_points: int = 20000,
    config: VizConfig = VizConfig(figsize=(12, 5)),
) -> plt.Figure:
    """
    Box plot of a given feature conditioned on label.
    Uses sampled timepoints if dataset is large.
    """
    _check_shapes(X, y)
    n_states, _ = config.resolve(y)

    X_flat = X.reshape(-1, X.shape[-1])
    y_flat = y.reshape(-1).astype(int)

    N = len(y_flat)
    if N > sample_max_points:
        idx = np.random.choice(N, size=sample_max_points, replace=False)
        X_flat = X_flat[idx]
        y_flat = y_flat[idx]

    groups = [X_flat[y_flat == s, feature_idx] for s in range(n_states)]

    fig, ax = plt.subplots(figsize=config.figsize)
    ax.boxplot(groups, labels=[str(s) for s in range(n_states)], showfliers=False)
    ax.set_title(f"Feature {feature_idx}: distribution by label")
    ax.set_xlabel("State label")
    ax.set_ylabel("Feature value")
    fig.tight_layout()
    return fig


# =========================
# 7) PCA scatter: timepoints or window summaries
# =========================

def plot_pca_scatter(
    X: np.ndarray,
    y: np.ndarray,
    mode: str = "timepoints",  # "timepoints" or "windows"
    window_agg: str = "mean",
    max_points: int = 20000,
    config: VizConfig = VizConfig(figsize=(10, 6)),
) -> plt.Figure:
    """
    PCA scatter plot colored by labels.
    - timepoints: each timepoint is a sample
    - windows: each window is a sample (colored by dominant label)
    """
    _check_shapes(X, y)
    n_states, cmap = config.resolve(y)

    try:
        from sklearn.decomposition import PCA
    except ImportError as e:
        raise ImportError("scikit-learn required for PCA. Install: pip install scikit-learn") from e

    if mode == "timepoints":
        Xs = X.reshape(-1, X.shape[-1])
        ys = y.reshape(-1).astype(int)

        if Xs.shape[0] > max_points:
            idx = np.random.choice(Xs.shape[0], size=max_points, replace=False)
            Xs = Xs[idx]
            ys = ys[idx]

        Z = PCA(n_components=2).fit_transform(Xs)

        fig, ax = plt.subplots(figsize=config.figsize)
        sc = ax.scatter(Z[:, 0], Z[:, 1], c=ys, cmap=cmap, s=10, alpha=0.7)
        ax.set_title("PCA scatter (timepoints) colored by label")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        fig.colorbar(sc, ax=ax, fraction=0.02, pad=0.02)
        fig.tight_layout()
        return fig

    elif mode == "windows":
        Xw = _window_summary_features(X, agg=window_agg)  # [W, D]
        # dominant label per window
        yw = np.array([np.bincount(y[w].astype(int)).argmax() for w in range(y.shape[0])], dtype=int)

        Z = PCA(n_components=2).fit_transform(Xw)

        fig, ax = plt.subplots(figsize=config.figsize)
        sc = ax.scatter(Z[:, 0], Z[:, 1], c=yw, cmap=cmap, s=30, alpha=0.8)
        ax.set_title(f"PCA scatter (windows, agg={window_agg}) colored by dominant label")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        fig.colorbar(sc, ax=ax, fraction=0.02, pad=0.02)
        fig.tight_layout()
        return fig

    else:
        raise ValueError("mode must be 'timepoints' or 'windows'")


# =========================
# 8) State prototypes: average feature value per label (timepoint-level)
# =========================

def plot_state_prototypes(
    X: np.ndarray,
    y: np.ndarray,
    feature_indices: Optional[Sequence[int]] = None,
    sample_max_points: int = 200000,
    config: VizConfig = VizConfig(figsize=(12, 5)),
) -> plt.Figure:
    """
    For each label, compute mean feature values across all timepoints with that label.
    Plots mean per state for selected features.
    """
    _check_shapes(X, y)
    n_states, _ = config.resolve(y)

    W, T, D = X.shape
    if feature_indices is None:
        feature_indices = list(range(min(10, D)))
    feature_indices = list(feature_indices)

    X_flat = X.reshape(-1, D)
    y_flat = y.reshape(-1).astype(int)

    N = X_flat.shape[0]
    if N > sample_max_points:
        idx = np.random.choice(N, size=sample_max_points, replace=False)
        X_flat = X_flat[idx]
        y_flat = y_flat[idx]

    means = np.zeros((n_states, len(feature_indices)), dtype=float)
    for si, s in enumerate(range(n_states)):
        mask = (y_flat == s)
        if mask.sum() == 0:
            means[si, :] = np.nan
        else:
            means[si, :] = X_flat[mask][:, feature_indices].mean(axis=0)

    fig, ax = plt.subplots(figsize=config.figsize)
    for j, fi in enumerate(feature_indices):
        ax.plot(np.arange(n_states), means[:, j], marker="o", label=f"feat_{fi}")
    ax.set_title("State prototypes: mean feature value per label")
    ax.set_xlabel("State label")
    ax.set_ylabel("Mean feature value")
    ax.legend(loc="best", frameon=True, ncol=2)
    fig.tight_layout()
    return fig


# =========================
# 9) Transition matrix heatmap
# =========================

def compute_transition_matrix(
    y: np.ndarray,
    num_states: Optional[int] = None,
    normalize_rows: bool = True,
) -> np.ndarray:
    """
    Counts transitions y[t]->y[t+1] across all windows.
    Returns shape [num_states, num_states].
    """
    if y.ndim != 2:
        raise ValueError(f"y must be 2D, got {y.shape}")
    if num_states is None:
        num_states = _infer_num_states(y)

    Tmat = np.zeros((num_states, num_states), dtype=float)
    for w in range(y.shape[0]):
        yw = y[w].astype(int)
        for t in range(len(yw) - 1):
            i = yw[t]
            j = yw[t + 1]
            if 0 <= i < num_states and 0 <= j < num_states:
                Tmat[i, j] += 1

    if normalize_rows:
        row_sums = Tmat.sum(axis=1, keepdims=True)
        Tmat = np.divide(Tmat, np.maximum(row_sums, 1e-12))
    return Tmat


def plot_transition_matrix(
    y: np.ndarray,
    normalize_rows: bool = True,
    config: VizConfig = VizConfig(figsize=(7, 6)),
) -> plt.Figure:
    """
    Heatmap of transition probabilities or counts.
    """
    if y.ndim != 2:
        raise ValueError(f"y must be 2D, got {y.shape}")
    n_states, _ = config.resolve(y)

    Tmat = compute_transition_matrix(y, num_states=n_states, normalize_rows=normalize_rows)

    fig, ax = plt.subplots(figsize=config.figsize)
    im = ax.imshow(Tmat, interpolation="nearest", origin="lower")
    ax.set_title("Transition matrix" + (" (row-normalized)" if normalize_rows else " (counts)"))
    ax.set_xlabel("Next state")
    ax.set_ylabel("Current state")
    ax.set_xticks(np.arange(n_states))
    ax.set_yticks(np.arange(n_states))
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()
    return fig


# =========================
# 10) Dwell time distributions per state
# =========================

def compute_dwell_times(
    y: np.ndarray,
    num_states: Optional[int] = None,
) -> Dict[int, List[int]]:
    """
    Compute run lengths (dwell times) for each state across all windows.
    Returns dict: {state: [durations...]}
    """
    if y.ndim != 2:
        raise ValueError(f"y must be 2D, got {y.shape}")
    if num_states is None:
        num_states = _infer_num_states(y)

    dwell: Dict[int, List[int]] = {s: [] for s in range(num_states)}
    for w in range(y.shape[0]):
        for lab, _, length in _runs(y[w].astype(int)):
            if 0 <= lab < num_states:
                dwell[lab].append(int(length))
    return dwell


def plot_dwell_time_histograms(
    y: np.ndarray,
    bins: int = 30,
    max_states_to_plot: int = 10,
    config: VizConfig = VizConfig(figsize=(12, 6)),
) -> plt.Figure:
    """
    Plots histograms of dwell times for each state (up to max_states_to_plot).
    """
    if y.ndim != 2:
        raise ValueError(f"y must be 2D, got {y.shape}")
    n_states, cmap = config.resolve(y)
    dwell = compute_dwell_times(y, num_states=n_states)

    states = list(range(min(n_states, max_states_to_plot)))

    fig, ax = plt.subplots(figsize=config.figsize)
    for s in states:
        if len(dwell[s]) == 0:
            continue
        ax.hist(dwell[s], bins=bins, alpha=0.4, label=f"state {s}")
    ax.set_title("Dwell time (run length) histograms per state")
    ax.set_xlabel("Duration (time steps)")
    ax.set_ylabel("Count")
    ax.legend(loc="best", frameon=True)
    fig.tight_layout()
    return fig


# =========================
# 11) Boundary neighborhoods (label transitions)
# =========================

def plot_boundary_neighborhoods(
    X: np.ndarray,
    y: np.ndarray,
    feature_idx: int = 0,
    window_idx: int = 0,
    radius: int = 25,
    max_boundaries: int = 20,
    config: VizConfig = VizConfig(figsize=(12, 6)),
) -> plt.Figure:
    """
    Plot feature snippets around label change boundaries in a chosen window.
    Overlays multiple boundary-centered snippets.
    """
    _check_shapes(X, y)
    n_states, _ = config.resolve(y)

    xw = X[window_idx, :, feature_idx]
    yw = y[window_idx].astype(int)

    boundaries = _label_boundaries(yw)
    if boundaries.size > max_boundaries:
        boundaries = boundaries[:max_boundaries]

    fig, ax = plt.subplots(figsize=config.figsize)

    t_rel = np.arange(-radius, radius + 1)
    plotted = 0
    for b in boundaries:
        lo = b - radius
        hi = b + radius + 1
        if lo < 0 or hi > len(xw):
            continue
        seg = xw[lo:hi]
        if seg.shape[0] != t_rel.shape[0]:
            continue
        ax.plot(t_rel, seg, alpha=0.5)
        plotted += 1

    ax.axvline(0, linewidth=2, alpha=0.7)
    ax.set_title(f"Boundary neighborhoods: window {window_idx}, feature {feature_idx} (n={plotted})")
    ax.set_xlabel("Time relative to boundary")
    ax.set_ylabel("Feature value")
    fig.tight_layout()
    return fig


# =========================
# 12) Small multiples: several windows with label ribbon + single feature
# =========================

def plot_small_multiples_windows(
    X: np.ndarray,
    y: np.ndarray,
    feature_idx: int = 0,
    window_indices: Optional[Sequence[int]] = None,
    ncols: int = 4,
    config: VizConfig = VizConfig(figsize=(14, 8)),
) -> plt.Figure:
    """
    Grid of windows, each showing one feature trace with label ribbon behind it.
    """
    _check_shapes(X, y)
    n_states, cmap = config.resolve(y)

    W, T, D = X.shape
    if window_indices is None:
        window_indices = list(range(min(W, 12)))
    window_indices = list(window_indices)

    n = len(window_indices)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=config.figsize, squeeze=False)
    axes = axes.flatten()

    for ax_i, ax in enumerate(axes):
        if ax_i >= n:
            ax.axis("off")
            continue

        w = window_indices[ax_i]
        xw = X[w, :, feature_idx]
        yw = y[w].astype(int)
        t = np.arange(T)

        ax.imshow(
            yw[np.newaxis, :],
            aspect="auto",
            cmap=cmap,
            alpha=config.alpha_label_band,
            interpolation="nearest",
            extent=[0, T - 1, np.min(xw), np.max(xw)],
            origin="lower",
        )
        ax.plot(t, xw, linewidth=1)
        ax.set_title(f"w={w}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(f"Small multiples: feature {feature_idx} with label ribbon", y=1.02)
    fig.tight_layout()
    return fig


# =========================
# 13) t-SNE / UMAP scatter (optional)
# =========================

def plot_embedding_scatter(
    X: np.ndarray,
    y: np.ndarray,
    method: str = "tsne",  # "tsne" or "umap"
    mode: str = "timepoints",  # "timepoints" or "windows"
    window_agg: str = "mean",
    max_points: int = 15000,
    config: VizConfig = VizConfig(figsize=(10, 6)),
    random_state: int = 0,
) -> plt.Figure:
    """
    2D embedding scatter colored by labels using t-SNE or UMAP.
    """
    _check_shapes(X, y)
    n_states, cmap = config.resolve(y)

    if mode == "timepoints":
        Xs = X.reshape(-1, X.shape[-1])
        ys = y.reshape(-1).astype(int)

        if Xs.shape[0] > max_points:
            idx = np.random.choice(Xs.shape[0], size=max_points, replace=False)
            Xs = Xs[idx]
            ys = ys[idx]
    elif mode == "windows":
        Xs = _window_summary_features(X, agg=window_agg)
        ys = np.array([np.bincount(y[w].astype(int)).argmax() for w in range(y.shape[0])], dtype=int)
    else:
        raise ValueError("mode must be 'timepoints' or 'windows'")

    if method.lower() == "tsne":
        try:
            from sklearn.manifold import TSNE
        except ImportError as e:
            raise ImportError("scikit-learn required for t-SNE. Install: pip install scikit-learn") from e

        Z = TSNE(n_components=2, random_state=random_state, init="pca", learning_rate="auto").fit_transform(Xs)

        fig, ax = plt.subplots(figsize=config.figsize)
        sc = ax.scatter(Z[:, 0], Z[:, 1], c=ys, cmap=cmap, s=10, alpha=0.7)
        ax.set_title(f"t-SNE scatter ({mode}) colored by label")
        ax.set_xlabel("dim 1")
        ax.set_ylabel("dim 2")
        fig.colorbar(sc, ax=ax, fraction=0.02, pad=0.02)
        fig.tight_layout()
        return fig

    elif method.lower() == "umap":
        try:
            import umap
        except ImportError as e:
            raise ImportError("umap-learn required for UMAP. Install: pip install umap-learn") from e

        reducer = umap.UMAP(n_components=2, random_state=random_state)
        Z = reducer.fit_transform(Xs)

        fig, ax = plt.subplots(figsize=config.figsize)
        sc = ax.scatter(Z[:, 0], Z[:, 1], c=ys, cmap=cmap, s=10, alpha=0.7)
        ax.set_title(f"UMAP scatter ({mode}) colored by label")
        ax.set_xlabel("dim 1")
        ax.set_ylabel("dim 2")
        fig.colorbar(sc, ax=ax, fraction=0.02, pad=0.02)
        fig.tight_layout()
        return fig

    else:
        raise ValueError("method must be 'tsne' or 'umap'")


# =========================
# Example usage
# =========================

if __name__ == "__main__":
    # Example dummy data
    # Load configuration
    with open("configs/vqvae_wadi.json") as f:
        config = json.load(f)

    # Load dataset
    loader = WADI_Dataloader()

    out = loader.load_processed_data()
    y_hat, y = vq_vae_visualisation_setup(out, config)
    y_hat = torch.cat(y_hat).numpy()
    y = torch.cat(y).numpy()

    X = out["test_ds"].X.numpy()


    cfg = VizConfig()

    #plot_window_traces_with_labels(X, y, window_idx=100, feature_indices=[0, 1, 2], config=cfg)
    #plot_window_heatmap_with_label_ribbon(X, y, window_idx=1000, config=cfg)
    plot_multiple_windows_true_vs_learned_one_figure(
        X, y, y_hat, k=8, seed=0)
    #plot_label_raster_all_windows(y, config=cfg)
    #plot_label_occupancy_per_window(y, config=cfg)
    #plot_global_label_distribution(y, config=cfg)
    #plot_feature_distributions_by_label(X, y, feature_idx=0, config=cfg)
    #plot_pca_scatter(X, y, mode="timepoints", config=cfg)
    #plot_state_prototypes(X, y, feature_indices=[0, 1, 2, 3], config=cfg)
    #plot_transition_matrix(y, normalize_rows=True, config=cfg)
    #plot_dwell_time_histograms(y, config=cfg)
    #plot_boundary_neighborhoods(X, y, feature_idx=2, window_idx=2, config=cfg)
    #plot_small_multiples_windows(X, y, feature_idx=0, config=cfg)

    # Optional (requires extra packages)
    #plot_embedding_scatter(X, y, method="tsne", mode="timepoints", config=cfg)
    # plot_embedding_scatter(X, y, method="umap", mode="windows", config=cfg)

    plt.show()
