import pytest


@pytest.fixture(autouse=True)
def _site_log_in_tmp(tmp_path, monkeypatch):
    """Every _DiversityState keeps a site log; under test it goes to tmp_path,
    never to the operator's ~/.aether-gate."""
    from aether_gate.adapters import diversity_state
    monkeypatch.setattr(diversity_state._DiversityState, "SITELOG_PATH",
                        str(tmp_path / "site-log.jsonl"))
