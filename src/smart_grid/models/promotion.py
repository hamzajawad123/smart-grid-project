"""Promote a refit only when test skill beats the current production pin."""


def should_promote(candidate: dict, production: dict, wape_tie: float) -> tuple[bool, str]:
    cand = candidate["refit_metrics"]["test"]
    prod = production["refit_metrics"]["test"]
    c_wape, p_wape = float(cand["wape"]), float(prod["wape"])
    c_peak, p_peak = float(cand["peak_mae"]), float(prod["peak_mae"])
    if c_wape < p_wape - 1e-12:
        return True, f"test WAPE {c_wape:.6f} < production {p_wape:.6f}"
    if abs(c_wape - p_wape) <= wape_tie and c_peak < p_peak - 1e-12:
        return True, (
            f"WAPE tied ({c_wape:.6f} vs {p_wape:.6f}); "
            f"Peak-MAE {c_peak:.3f} < production {p_peak:.3f}"
        )
    return False, (
        f"hold: candidate WAPE {c_wape:.6f} / Peak-MAE {c_peak:.3f} "
        f"vs production {p_wape:.6f} / {p_peak:.3f}"
    )
