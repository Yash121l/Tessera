"""Kelly criterion position sizing.

Reference: Thorp (2006), "The Kelly Criterion in Blackjack, Sports Betting,
and the Stock Market."
"""

from __future__ import annotations


def fractional_kelly(p_win: float, win_loss_ratio: float, fraction: float = 0.25) -> float:
    """Standard Kelly criterion, scaled by a fractional multiplier.

    Full Kelly: f* = (p_win * b - p_lose) / b  where b = win_loss_ratio.
    Returns fraction * f*, capped at `fraction` to prevent overlevering.

    Args:
        p_win: Probability of a winning trade, in (0, 1).
        win_loss_ratio: Expected gain per unit risked divided by expected loss
            per unit risked (b in the Kelly formula).
        fraction: Multiplier on full Kelly, e.g. 0.25 = quarter-Kelly.

    Returns:
        Position fraction of portfolio in [0, fraction].
    """
    if not (0.0 < p_win < 1.0) or win_loss_ratio <= 0.0:
        return 0.0
    p_lose = 1.0 - p_win
    f_star = (p_win * win_loss_ratio - p_lose) / win_loss_ratio
    if f_star <= 0.0:
        return 0.0
    return min(fraction * f_star, fraction)


def kelly_from_meta_prob(
    p_meta: float,
    expected_return: float,
    expected_loss: float,
    fraction: float = 0.25,
) -> float:
    """Kelly fraction from meta-model probability and expected trade magnitudes.

    Converts a meta-model's probability that a trade is profitable (p_meta)
    and the expected return/loss magnitudes into a position fraction via the
    standard Kelly formula with win_loss_ratio = expected_return / expected_loss.

    Example: p_meta=0.6, expected_return=0.02, expected_loss=0.01 →
      60% chance of +2%, 40% chance of -1%, win_loss_ratio=2.0.

    Args:
        p_meta: Meta-model probability of a winning trade, in (0, 1).
        expected_return: Expected gain magnitude (positive, e.g. 0.02 for 2%).
        expected_loss: Expected loss magnitude (positive, e.g. 0.01 for 1%).
        fraction: Fractional multiplier on full Kelly.

    Returns:
        Position fraction in [0, fraction].
    """
    if not (0.0 < p_meta < 1.0) or expected_return <= 0.0 or expected_loss <= 0.0:
        return 0.0
    win_loss_ratio = expected_return / expected_loss
    return fractional_kelly(p_meta, win_loss_ratio, fraction)
