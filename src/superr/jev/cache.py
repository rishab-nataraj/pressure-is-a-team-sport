"""SQLite cache for Jev: every raw request/response is kept, and every answer is keyed by a
hash of (pinned model, state, question). Re-running the pipeline never re-bills a call."""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from pathlib import Path


def canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def answer_key(model: str, state, qid: str, question: dict) -> str:
    """Hash of one question asked about one state on one pinned model version."""
    payload = canonical_json({"model": model, "state": state, "qid": qid, "question": question})
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def request_key(body: dict) -> str:
    return hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest()


class JevCache:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("""CREATE TABLE IF NOT EXISTS requests (
            req_key TEXT PRIMARY KEY, model TEXT, request TEXT, response TEXT, created REAL)""")
        self.conn.execute("""CREATE TABLE IF NOT EXISTS answers (
            key TEXT PRIMARY KEY, qid TEXT, model TEXT, served_model TEXT, answer TEXT, req_key TEXT)""")
        self.conn.commit()

    def get_many(self, keys: list[str]) -> dict[str, dict]:
        out: dict[str, dict] = {}
        uniq = list(dict.fromkeys(keys))
        with self._lock:
            for i in range(0, len(uniq), 900):
                chunk = uniq[i:i + 900]
                q = f"SELECT key, answer FROM answers WHERE key IN ({','.join('?' * len(chunk))})"
                for k, a in self.conn.execute(q, chunk):
                    out[k] = json.loads(a)
        return out

    def put(self, req_key: str, model: str, request: dict, response: dict,
            answers: list[tuple[str, str, dict]]) -> None:
        """answers: list of (answer_key, qid, answer_json)."""
        served = response.get("model", model)
        with self._lock:
            self.conn.execute("INSERT OR REPLACE INTO requests VALUES (?,?,?,?,?)",
                              (req_key, model, canonical_json(request), canonical_json(response), time.time()))
            self.conn.executemany("INSERT OR REPLACE INTO answers VALUES (?,?,?,?,?,?)",
                                  [(k, qid, model, served, canonical_json(a), req_key) for k, qid, a in answers])

    def commit(self) -> None:
        with self._lock:
            self.conn.commit()

    def stats(self) -> dict:
        with self._lock:
            n_req = self.conn.execute("SELECT COUNT(*) FROM requests").fetchone()[0]
            n_ans = self.conn.execute("SELECT COUNT(*) FROM answers").fetchone()[0]
            served = dict(self.conn.execute("SELECT served_model, COUNT(*) FROM answers GROUP BY served_model"))
        return {"requests": n_req, "answers": n_ans, "served_models": served}

    def sample_requests(self, frac: float, seed: int = 0) -> list[tuple[str, dict, dict]]:
        """Random sample of cached (req_key, request, response) for the reproducibility check."""
        with self._lock:
            rows = self.conn.execute("SELECT req_key, request, response FROM requests").fetchall()
        import random
        rnd = random.Random(seed)
        k = max(1, int(len(rows) * frac)) if rows else 0
        return [(r[0], json.loads(r[1]), json.loads(r[2])) for r in rnd.sample(rows, k)]

    def close(self) -> None:
        self.commit()
        self.conn.close()
