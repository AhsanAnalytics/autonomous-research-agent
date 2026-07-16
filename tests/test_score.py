from unittest.mock import patch, MagicMock

from app.fusion import reciprocal_rank_fusion
from app.api_tools import get_exchange_rate


# ---- RRF fusion: pure logic, deterministic ----

def test_rrf_merges_all_ids():
    fused = reciprocal_rank_fusion([["x", "y", "z"], ["y", "x", "w"]])
    assert set(fused) == {"x", "y", "z", "w"}


def test_rrf_rewards_items_ranked_high_in_both():
    # "top" is #1 in both lists, so it must come out first.
    fused = reciprocal_rank_fusion([["top", "mid", "low"], ["top", "other", "mid"]])
    assert fused[0] == "top"


def test_rrf_handles_empty_input():
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[]]) == []


# ---- Exchange-rate tool: mock the HTTP call, test all branches ----

def _fake_response(payload):
    r = MagicMock()
    r.raise_for_status = lambda: None
    r.json = lambda: payload
    return r


def test_exchange_rate_success():
    payload = {"result": "success", "rates": {"PKR": 278.5}}
    with patch("app.api_tools.requests.get", return_value=_fake_response(payload)):
        out = get_exchange_rate.invoke({"base": "usd", "quote": "pkr"})
    assert "278.5" in out and "USD" in out and "PKR" in out


def test_exchange_rate_missing_currency():
    payload = {"result": "success", "rates": {"EUR": 0.9}}  # no PKR
    with patch("app.api_tools.requests.get", return_value=_fake_response(payload)):
        out = get_exchange_rate.invoke({"base": "USD", "quote": "PKR"})
    assert out.startswith("error")


def test_exchange_rate_network_failure():
    with patch("app.api_tools.requests.get", side_effect=Exception("boom")):
        out = get_exchange_rate.invoke({"base": "USD", "quote": "PKR"})
    assert out.startswith("error")   # graceful failure, not a crash