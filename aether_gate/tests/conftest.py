import pytest


@pytest.fixture(autouse=True)
def _site_log_in_tmp(tmp_path, monkeypatch):
    """Every _DiversityState keeps a site log; under test it goes to tmp_path,
    never to the operator's ~/.aether-gate."""
    from aether_gate.adapters import diversity_state
    monkeypatch.setattr(diversity_state._DiversityState, "SITELOG_PATH",
                        str(tmp_path / "site-log.jsonl"))
    # ...and so does the talker memory: G2 WRITES it (names alone were written
    # only when one was set), so a test must never be pointed at the real file.
    monkeypatch.setattr(diversity_state._DiversityState, "NAMES_PATH",
                        str(tmp_path / "diversity-names.json"))
    monkeypatch.setattr(diversity_state._DiversityState, "TALKERS_PATH",
                        str(tmp_path / "diversity-talkers.json"))
