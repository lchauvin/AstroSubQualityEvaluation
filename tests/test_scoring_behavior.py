from astro_eval.metrics import EvalConfig, FrameMetrics
from astro_eval.scoring import compute_rejection_flags, compute_score, SessionStats


def _frame(mode: str = "gas", gradient: float = 60.0) -> FrameMetrics:
    return FrameMetrics(
        filename="f1.fits",
        filepath="C:/tmp/f1.fits",
        mode=mode,
        filter_name="Ha" if mode == "gas" else "L",
        exptime=300.0,
        gain=None,
        ccd_temp=None,
        pixel_scale=1.5,
        background_median=1000.0,
        background_rms=10.0,
        noise_mad=9.5,
        background_gradient=gradient,
        n_stars=200,
        fwhm_median=2.0,
        eccentricity_median=0.35,
        snr_weight=500.0,
        psf_signal_weight=600.0,
        snr_estimate=5.0,
    )


def test_high_gradient_flag_uses_session_relative_threshold():
    cfg = EvalConfig(sigma_gradient=2.0, gradient_threshold=0.0)
    frame = _frame(mode="gas", gradient=80.0)
    session_stats = {
        "background_gradient": SessionStats(
            metric_name="background_gradient",
            count=10,
            median=40.0,
            mean=42.0,
            std=10.0,
            min_val=20.0,
            max_val=80.0,
        )
    }

    flags = compute_rejection_flags(frame, session_stats, cfg)
    # 80 > 40 + 2*10 => hard reject
    assert flags.flags["high_gradient"] is True
    assert flags.rejected is True


def test_gradient_multiplier_applies_only_in_gas_mode():
    cfg = EvalConfig(gradient_knee=1.2)
    session_stats = {
        "background_gradient": SessionStats(
            metric_name="background_gradient",
            count=10,
            median=20.0,
            mean=22.0,
            std=5.0,
            min_val=8.0,
            max_val=60.0,
        ),
        "snr_estimate": SessionStats("snr_estimate", 10, 5.0, 5.1, 0.5, 3.0, 7.0),
        "background_rms": SessionStats("background_rms", 10, 10.0, 10.1, 1.0, 8.0, 12.0),
        "background_median": SessionStats("background_median", 10, 1000.0, 1002.0, 20.0, 960.0, 1040.0),
        "n_stars": SessionStats("n_stars", 10, 200.0, 205.0, 30.0, 120.0, 260.0),
        "psf_signal_weight": SessionStats("psf_signal_weight", 10, 600.0, 610.0, 80.0, 400.0, 800.0),
        "fwhm_median": SessionStats("fwhm_median", 10, 2.0, 2.1, 0.2, 1.6, 2.6),
        "eccentricity_median": SessionStats("eccentricity_median", 10, 0.35, 0.36, 0.04, 0.25, 0.48),
        "snr_weight": SessionStats("snr_weight", 10, 500.0, 510.0, 80.0, 320.0, 700.0),
    }

    gas_score = compute_score(_frame(mode="gas", gradient=60.0), session_stats, config=cfg)
    star_score = compute_score(_frame(mode="star", gradient=60.0), session_stats, config=cfg)

    # Same high gradient should penalize gas scoring, but not star scoring.
    assert gas_score < star_score
