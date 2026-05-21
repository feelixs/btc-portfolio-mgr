from datetime import datetime, timezone

from btc_portfolio_mgr.live.inference_loop import LiveInferenceResult


def test_live_inference_result_dataclass_fields():
    r = LiveInferenceResult(
        timestamp=datetime(2026, 5, 21, 0, 0, tzinfo=timezone.utc),
        mu=0.01,
        sigma=0.02,
        target_weight=0.4,
        price_usdt=77_000.0,
    )
    assert r.mu == 0.01
    assert r.target_weight == 0.4
    assert r.price_usdt == 77_000.0
