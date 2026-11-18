import json

import numpy as np
import pandas as pd

from conftest import make_state
from superr.cricket import OUTCOMES, STATE_COLS
from superr.jev import JevCache, JevClient, prompts
from superr.jev.cache import answer_key


class FakeResponse:
    def __init__(self, payload, status=200):
        self._p = payload
        self.status_code = status
        self.headers = {}
        self.text = json.dumps(payload)

    def json(self):
        return self._p


class FakeSession:
    """Answers wp = 0.6 and a uniform next-ball distribution; counts calls."""

    def __init__(self, fail_first=0):
        self.calls = 0
        self.bodies = []
        self.fail_first = fail_first

    def post(self, url, json=None, headers=None, timeout=None):
        self.calls += 1
        self.bodies.append(json)
        if self.calls <= self.fail_first:
            return FakeResponse({"error": "busy"}, status=529)
        ans = {}
        for qid, q in json["questions"].items():
            if q["type"] == "noul":
                ans[qid] = {"type": "noul", "noul": 0.6}
            else:
                ans[qid] = {"type": "choice", "choice": "one", "confidence": 0.3,
                            "probabilities": {k: 1 / len(q["criteria"]) for k in q["criteria"]}}
        return FakeResponse({"model": "jev-1.13.0", "answers": ans, "usage": {"input_tokens": 100, "output_tokens": 0}})


def _client(tmp_path, monkeypatch, session, cfg):
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
    jcfg = dict(cfg["jev"], max_workers=2, backoff_base=0.0)
    return JevClient(jcfg, JevCache(tmp_path / "c.sqlite"), session=session)


def _all_keys(d):
    out = set()
    for k, v in d.items():
        out.add(k)
        if isinstance(v, dict):
            out |= _all_keys(v)
    return out


def test_state_payload_has_no_identities():
    S = make_state()
    payload = prompts.state_to_jev(S[STATE_COLS].iloc[0].to_dict())
    keys = json.dumps(sorted(_all_keys(payload))).lower()
    for banned in ("match_id", "team", "player_id", "player_name", "venue_id", "venue_name", "date",
                   "striker_id", "bowler", "comp_id", "comp_name", "season"):
        assert banned not in keys
    assert payload["innings"] == 2 and payload["runs_needed"] == 10


def test_client_caches_answers(tmp_path, monkeypatch, cfg):
    sess = FakeSession()
    cl = _client(tmp_path, monkeypatch, sess, cfg)
    s = prompts.state_to_jev(make_state()[STATE_COLS].iloc[0].to_dict())
    q = {"wp": prompts.WP_QUESTION, "p_next": prompts.PNEXT_QUESTION}
    a1 = cl.ask([(s, q)])
    assert sess.calls == 1 and set(sess.bodies[0]["questions"]) == {"wp", "p_next"}  # one request, both questions
    a2 = cl.ask([(s, {"wp": prompts.WP_QUESTION})])
    assert sess.calls == 1                         # served from the cache
    assert prompts.parse_wp(a2[0]["wp"]) == 0.6
    P = prompts.parse_pnext(a1[0]["p_next"])
    assert np.isclose(P.sum(), 1) and len(P) == len(OUTCOMES)
    stats = cl.cache.stats()
    assert stats["requests"] == 1 and stats["answers"] == 2


def test_client_retries_overload(tmp_path, monkeypatch, cfg):
    sess = FakeSession(fail_first=2)
    cl = _client(tmp_path, monkeypatch, sess, cfg)
    s = prompts.state_to_jev(make_state()[STATE_COLS].iloc[0].to_dict())
    out = cl.ask([(s, {"wp": prompts.WP_QUESTION})])
    assert sess.calls == 3 and prompts.parse_wp(out[0]["wp"]) == 0.6


def test_answer_key_depends_on_model_and_state():
    s = {"runs": 1}
    k1 = answer_key("jev-1.13.0", s, "wp", prompts.WP_QUESTION)
    assert k1 == answer_key("jev-1.13.0", {"runs": 1}, "wp", prompts.WP_QUESTION)
    assert k1 != answer_key("jev-1.14.0", s, "wp", prompts.WP_QUESTION)
    assert k1 != answer_key("jev-1.13.0", {"runs": 2}, "wp", prompts.WP_QUESTION)


def test_stakes_context_is_anonymous():
    ctx = dict(comp_kind="ipl", phase="league", qualify=4, this_match=(101, 202),
               standings=[dict(team_id=t, played=5, won=w, lost=5 - w, no_result=0, points=2 * w, nrr=0.1 * w)
                          for t, w in ((101, 3), (202, 2), (303, 4))],
               remaining=[(101, 303), (202, 303)])
    p = prompts.stakes_context_to_jev(ctx)
    text = json.dumps(p)
    assert "101" not in text and "202" not in text and "303" not in text
    assert p["focal_team"] in ("A", "B", "C") and p["qualifying_places"] == 4
    assert p["standings_before_match"][0]["points"] == 8      # table order, relabelled
