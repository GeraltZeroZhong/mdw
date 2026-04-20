from __future__ import annotations

import numpy as np

try:
    from deeptime.markov.msm import MaximumLikelihoodMSM
except Exception:
    MaximumLikelihoodMSM = None


def safe_msm_fit(dtrajs, requested_lag):
    if MaximumLikelihoodMSM is None:
        raise ImportError("缺少 deeptime.markov.msm.MaximumLikelihoodMSM。")
    dtrajs = [np.asarray(x, dtype=np.int32).ravel() for x in dtrajs if len(x) >= 2]
    if not dtrajs:
        raise ValueError("没有可用于 MSM 的离散轨迹。")
    lengths = [len(x) for x in dtrajs]
    safe_lag = min(requested_lag, max(1, min(lengths) - 1))
    model = MaximumLikelihoodMSM(reversible=True).fit(dtrajs, lagtime=safe_lag).fetch_model()
    return model, safe_lag, lengths


__all__ = ["MaximumLikelihoodMSM", "safe_msm_fit"]
