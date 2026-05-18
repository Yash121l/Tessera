"""Unit tests for tessera.risk.kelly."""

from __future__ import annotations

from tessera.risk.kelly import fractional_kelly, kelly_from_meta_prob


class TestFractionalKelly:
    def test_positive_edge(self) -> None:
        frac = fractional_kelly(p_win=0.6, win_loss_ratio=1.0, fraction=0.25)
        assert 0.0 < frac <= 0.25

    def test_standard_formula(self) -> None:
        # f* = (0.6*2 - 0.4) / 2 = 0.4; quarter-Kelly = 0.1
        frac = fractional_kelly(p_win=0.6, win_loss_ratio=2.0, fraction=0.25)
        assert abs(frac - 0.1) < 1e-9

    def test_negative_edge_returns_zero(self) -> None:
        # p_win=0.3, b=1.0 → f* = (0.3 - 0.7) / 1.0 < 0
        assert fractional_kelly(p_win=0.3, win_loss_ratio=1.0) == 0.0

    def test_zero_p_win_returns_zero(self) -> None:
        assert fractional_kelly(p_win=0.0, win_loss_ratio=2.0) == 0.0

    def test_p_win_one_returns_zero(self) -> None:
        assert fractional_kelly(p_win=1.0, win_loss_ratio=2.0) == 0.0

    def test_zero_win_loss_ratio_returns_zero(self) -> None:
        assert fractional_kelly(p_win=0.6, win_loss_ratio=0.0) == 0.0

    def test_fraction_cap(self) -> None:
        # Very high win prob → f* > 1, but output must not exceed fraction
        frac = fractional_kelly(p_win=0.99, win_loss_ratio=10.0, fraction=0.25)
        assert frac <= 0.25

    def test_fraction_parameter_scales_result(self) -> None:
        f_half = fractional_kelly(p_win=0.6, win_loss_ratio=2.0, fraction=0.50)
        f_quarter = fractional_kelly(p_win=0.6, win_loss_ratio=2.0, fraction=0.25)
        assert abs(f_half - 2 * f_quarter) < 1e-9

    def test_negative_win_loss_ratio_returns_zero(self) -> None:
        assert fractional_kelly(p_win=0.6, win_loss_ratio=-1.0) == 0.0


class TestKellyFromMetaProb:
    def test_basic(self) -> None:
        # Same as fractional_kelly(0.6, 2.0, 0.25)
        frac = kelly_from_meta_prob(
            p_meta=0.6, expected_return=0.02, expected_loss=0.01, fraction=0.25
        )
        ref = fractional_kelly(p_win=0.6, win_loss_ratio=2.0, fraction=0.25)
        assert abs(frac - ref) < 1e-9

    def test_zero_expected_loss_returns_zero(self) -> None:
        assert kelly_from_meta_prob(0.6, 0.02, 0.0) == 0.0

    def test_zero_expected_return_returns_zero(self) -> None:
        assert kelly_from_meta_prob(0.6, 0.0, 0.01) == 0.0

    def test_p_meta_boundary_returns_zero(self) -> None:
        assert kelly_from_meta_prob(0.0, 0.02, 0.01) == 0.0
        assert kelly_from_meta_prob(1.0, 0.02, 0.01) == 0.0

    def test_result_bounded_by_fraction(self) -> None:
        frac = kelly_from_meta_prob(0.9, 0.05, 0.01, fraction=0.25)
        assert 0.0 < frac <= 0.25
