from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    from deeptime.markov import TransitionCountEstimator
    from deeptime.markov.msm import MaximumLikelihoodMSM
except Exception:
    TransitionCountEstimator = None
    MaximumLikelihoodMSM = None


@dataclass(frozen=True)
class ConnectedSetSummary:
    rank: int
    symbols: tuple[int, ...]
    n_states: int
    frame_count: int
    frame_fraction: float


@dataclass
class ConnectedMSMResult:
    model: Any
    safe_lag: int
    traj_lengths: list[int]
    counts_full: Any
    counts_active: Any
    connected_sets: list[ConnectedSetSummary]
    active_symbols: np.ndarray
    active_dtrajs: list[np.ndarray]
    total_frame_count: int
    active_frame_count: int


def _normalize_dtrajs(dtrajs) -> list[np.ndarray]:
    cleaned = [np.asarray(x, dtype=np.int32).ravel() for x in dtrajs if len(x) >= 2]
    if not cleaned:
        raise ValueError("没有可用于 MSM 的离散轨迹。")
    return cleaned


def _connected_set_summaries(counts_full) -> list[ConnectedSetSummary]:
    histogram = np.asarray(getattr(counts_full, "state_histogram", []), dtype=int)
    total_frames = int(histogram.sum())
    summaries: list[ConnectedSetSummary] = []
    for rank, symbols in enumerate(counts_full.connected_sets(), start=1):
        symbols_arr = np.asarray(symbols, dtype=int)
        frame_count = int(histogram[symbols_arr].sum()) if symbols_arr.size else 0
        frame_fraction = float(frame_count / max(total_frames, 1))
        summaries.append(
            ConnectedSetSummary(
                rank=rank,
                symbols=tuple(int(value) for value in symbols_arr.tolist()),
                n_states=int(symbols_arr.size),
                frame_count=frame_count,
                frame_fraction=frame_fraction,
            )
        )
    return summaries


def _select_population_dominant_set(summaries: list[ConnectedSetSummary]) -> np.ndarray:
    if not summaries:
        raise ValueError("无法从计数矩阵中识别任何连通状态集合。")
    best = sorted(summaries, key=lambda item: (-item.frame_count, -item.n_states, item.symbols))[0]
    return np.asarray(best.symbols, dtype=np.int32)


def _split_on_invalid_segments(dtrajs) -> list[np.ndarray]:
    segments: list[np.ndarray] = []
    for dtraj in dtrajs:
        arr = np.asarray(dtraj, dtype=np.int32).ravel()
        if arr.size == 0:
            continue
        valid = arr >= 0
        if not np.any(valid):
            continue
        start = None
        for idx, is_valid in enumerate(valid):
            if is_valid and start is None:
                start = idx
            elif (not is_valid) and start is not None:
                segment = arr[start:idx]
                if segment.size >= 2:
                    segments.append(segment)
                start = None
        if start is not None:
            segment = arr[start:]
            if segment.size >= 2:
                segments.append(segment)
    return segments


def fit_population_connected_msm(dtrajs, requested_lag, count_mode: str = "sliding") -> ConnectedMSMResult:
    if MaximumLikelihoodMSM is None or TransitionCountEstimator is None:
        raise ImportError("缺少 deeptime 的 MSM 估计依赖。")
    cleaned = _normalize_dtrajs(dtrajs)
    lengths = [len(x) for x in cleaned]
    safe_lag = min(int(requested_lag), max(1, min(lengths) - 1))
    counts_full = TransitionCountEstimator(lagtime=safe_lag, count_mode=count_mode).fit_fetch(cleaned)
    summaries = _connected_set_summaries(counts_full)
    active_symbols = _select_population_dominant_set(summaries)
    counts_active = counts_full.submodel(active_symbols)
    mapped_dtrajs = counts_active.transform_discrete_trajectories_to_submodel(cleaned)
    active_segments = _split_on_invalid_segments(mapped_dtrajs)
    valid_segments = [segment for segment in active_segments if len(segment) > safe_lag]
    if not valid_segments:
        raise ValueError("所选强连通分量中没有足够长的轨迹片段来拟合 MSM。")
    model = MaximumLikelihoodMSM(reversible=True, use_lcc=True).fit(valid_segments, lagtime=safe_lag).fetch_model()
    histogram = np.asarray(getattr(counts_full, "state_histogram", []), dtype=int)
    total_frame_count = int(histogram.sum())
    active_frame_count = int(histogram[active_symbols].sum()) if active_symbols.size else 0
    return ConnectedMSMResult(
        model=model,
        safe_lag=safe_lag,
        traj_lengths=lengths,
        counts_full=counts_full,
        counts_active=counts_active,
        connected_sets=summaries,
        active_symbols=np.asarray(active_symbols, dtype=np.int32),
        active_dtrajs=active_segments,
        total_frame_count=total_frame_count,
        active_frame_count=active_frame_count,
    )


def fit_msm_on_active_symbols(dtrajs, active_symbols, lagtime, count_mode: str = "sliding"):
    if MaximumLikelihoodMSM is None or TransitionCountEstimator is None:
        raise ImportError("缺少 deeptime 的 MSM 估计依赖。")
    cleaned = _normalize_dtrajs(dtrajs)
    active_symbols = np.asarray(active_symbols, dtype=np.int32).ravel()
    if active_symbols.size == 0:
        raise ValueError("active_symbols 不能为空。")
    counts_full = TransitionCountEstimator(lagtime=int(lagtime), count_mode=count_mode).fit_fetch(cleaned)
    counts_active = counts_full.submodel(active_symbols)
    mapped_dtrajs = counts_active.transform_discrete_trajectories_to_submodel(cleaned)
    active_segments = _split_on_invalid_segments(mapped_dtrajs)
    valid_segments = [segment for segment in active_segments if len(segment) > int(lagtime)]
    if not valid_segments:
        raise ValueError(f"活动状态集合 {active_symbols.tolist()} 在 lag={lagtime} 下没有足够长的轨迹片段。")
    model = MaximumLikelihoodMSM(reversible=True, use_lcc=True).fit(valid_segments, lagtime=int(lagtime)).fetch_model()
    return model, active_segments, counts_full, counts_active


def safe_msm_fit(dtrajs, requested_lag):
    result = fit_population_connected_msm(dtrajs, requested_lag)
    return result.model, result.safe_lag, result.traj_lengths


__all__ = [
    "ConnectedMSMResult",
    "ConnectedSetSummary",
    "MaximumLikelihoodMSM",
    "fit_msm_on_active_symbols",
    "fit_population_connected_msm",
    "safe_msm_fit",
]
