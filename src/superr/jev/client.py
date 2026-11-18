"""HTTP client for Jev's System One endpoint (POST /v1/systemone).

Request:  {"model": <pinned id>, "state": <object>, "questions": {qid: {type, instructions, criteria?}}}
Response: {"model": ..., "answers": {qid: {"type": "noul", "noul": p} |
                                          {"type": "choice", "choice": k, "probabilities": {...}, ...}},
           "usage": {...}}

All independent questions about one state go in one request (TypeSafe evaluates them in parallel).
429 / 529 / 5xx are retried with exponential backoff + jitter.
"""
from __future__ import annotations

import logging
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

from .cache import JevCache, answer_key, request_key

log = logging.getLogger("superr.jev")

RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504, 529}


class JevError(RuntimeError):
    pass


class JevClient:
    def __init__(self, jcfg: dict, cache: JevCache, session: requests.Session | None = None):
        self.cfg = jcfg
        self.model = jcfg["model"]
        self.url = jcfg["base_url"]
        self.key = os.environ.get(jcfg.get("api_key_env", "TYPESAFE_API_KEY"), "")
        self.cache = cache
        self.session = session or requests.Session()
        self.timeout = jcfg.get("timeout", 30)
        self.max_retries = jcfg.get("max_retries", 8)
        self.backoff = jcfg.get("backoff_base", 1.0)
        self.workers = jcfg.get("max_workers", 32)
        self.flush_every = jcfg.get("flush_every", 500)
        self.usage = {"requests": 0, "input_tokens": 0, "output_tokens": 0, "cache_hits": 0}

    @property
    def available(self) -> bool:
        return bool(self.key)

    # ------------------------------------------------------------------ low level
    def post(self, body: dict) -> dict:
        if not self.key:
            raise JevError(f"Jev API key missing: set ${self.cfg.get('api_key_env', 'TYPESAFE_API_KEY')}")
        headers = {"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"}
        for attempt in range(self.max_retries + 1):
            try:
                r = self.session.post(self.url, json=body, headers=headers, timeout=self.timeout)
            except requests.RequestException as e:
                if attempt == self.max_retries:
                    raise JevError(f"network error: {e}") from e
                self._sleep(attempt)
                continue
            if r.status_code == 200:
                return r.json()
            if r.status_code in RETRY_STATUS and attempt < self.max_retries:
                ra = r.headers.get("Retry-After")
                self._sleep(attempt, float(ra) if ra and ra.replace(".", "", 1).isdigit() else None)
                continue
            raise JevError(f"Jev HTTP {r.status_code}: {r.text[:500]}")
        raise JevError("retries exhausted")

    def _sleep(self, attempt: int, retry_after: float | None = None) -> None:
        t = retry_after if retry_after is not None else self.backoff * (2 ** attempt)
        time.sleep(min(t, 60.0) * (0.5 + random.random()))

    # ------------------------------------------------------------------ batched, cached
    def ask(self, items: list[tuple[dict, dict]], force: bool = False, desc: str = "jev") -> list[dict]:
        """Ask questions about many states.

        items: list of (state, questions) where questions = {qid: question_def}.
        Returns, per item, {qid: answer}. Answers are served from the cache when present;
        only the missing questions of each state are sent (one request per state).
        """
        keys = [{qid: answer_key(self.model, s, qid, q) for qid, q in qs.items()} for s, qs in items]
        cached = {} if force else self.cache.get_many([k for d in keys for k in d.values()])
        results: list[dict] = [dict() for _ in items]
        todo = []
        for i, ((state, qs), kd) in enumerate(zip(items, keys)):
            missing = {}
            for qid, k in kd.items():
                if k in cached:
                    results[i][qid] = cached[k]
                    self.usage["cache_hits"] += 1
                else:
                    missing[qid] = qs[qid]
            if missing:
                todo.append((i, state, missing))
        if not todo:
            return results
        log.info("%s: %d cached answers, %d requests to send (model %s)", desc,
                 sum(len(r) for r in results), len(todo), self.model)
        done = 0
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=self.workers) as ex:
            futs = {ex.submit(self._one, state, qs): (i, state, qs) for i, state, qs in todo}
            for fut in as_completed(futs):
                i, state, qs = futs[fut]
                body, resp = fut.result()
                ans = resp.get("answers", {})
                rows = []
                for qid in qs:
                    if qid not in ans:
                        raise JevError(f"answer '{qid}' missing in response: {resp}")
                    results[i][qid] = ans[qid]
                    rows.append((keys[i][qid], qid, ans[qid]))
                self.cache.put(request_key(body), self.model, body, resp, rows)
                u = resp.get("usage", {})
                self.usage["requests"] += 1
                self.usage["input_tokens"] += int(u.get("input_tokens", 0))
                self.usage["output_tokens"] += int(u.get("output_tokens", 0))
                done += 1
                if done % self.flush_every == 0:
                    self.cache.commit()
                    rate = done / max(time.time() - t0, 1e-9)
                    log.info("%s: %d/%d requests (%.0f/s, eta %.0f min)", desc, done, len(todo), rate,
                             (len(todo) - done) / max(rate, 1e-9) / 60)
        self.cache.commit()
        return results

    def _one(self, state: dict, questions: dict) -> tuple[dict, dict]:
        body = {"model": self.model, "state": state, "questions": questions}
        return body, self.post(body)

    # ------------------------------------------------------------------ planning
    def estimate(self, items: list[tuple[dict, dict]]) -> dict:
        """Requests still needed and a rough token/cost estimate (~4 characters per token)."""
        from .cache import canonical_json
        keys = [answer_key(self.model, s, qid, q) for s, qs in items for qid, q in qs.items()]
        cached = self.cache.get_many(keys)
        need, chars = 0, 0
        for s, qs in items:
            miss = {qid: q for qid, q in qs.items() if answer_key(self.model, s, qid, q) not in cached}
            if miss:
                need += 1
                chars += len(canonical_json({"state": s, "questions": miss}))
        tokens = chars / 4
        return {"requests": need, "approx_input_tokens": int(tokens),
                "approx_usd": tokens / 1e6 * self.cfg.get("usd_per_million_input_tokens", 0.042),
                "cached_answers": len(cached)}
