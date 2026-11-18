"""Command line: python -m superr run all | <stage> [<stage> ...]

Stages: canon states jev balls stints fit rank validate report  (or 0..8)
"""
from __future__ import annotations

import argparse
import logging
import time

import yaml

from .config import load_config, setup_logging

log = logging.getLogger("superr")

STAGES = ["canon", "states", "jev", "balls", "stints", "fit", "rank", "validate", "report"]


def _set_overrides(pairs: list[str]) -> dict:
    """--set pressure.k=0.5 --set model.use_tau=false  ->  nested dict."""
    out: dict = {}
    for p in pairs or []:
        key, _, val = p.partition("=")
        cur = out
        parts = key.split(".")
        for part in parts[:-1]:
            cur = cur.setdefault(part, {})
        cur[parts[-1]] = yaml.safe_load(val)
    return out


def run_stage(name: str, cfg: dict, args) -> None:
    t0 = time.time()
    log.info("=== stage %s ===", name)
    if name == "canon":
        from . import canon; canon.run(cfg)
    elif name == "states":
        from . import states; states.run(cfg)
    elif name == "jev":
        from . import jev_stage; jev_stage.run(cfg, plan_only=getattr(args, "plan", False))
    elif name == "balls":
        from . import balls; balls.run(cfg)
    elif name == "stints":
        from . import stints; stints.run(cfg)
    elif name == "fit":
        from . import fit_stage
        fit_stage.run(cfg, skip_cv=getattr(args, "skip_cv", False), skip_bootstrap=getattr(args, "skip_bootstrap", False))
    elif name == "rank":
        from . import rank; rank.run(cfg)
    elif name == "validate":
        from . import validate_data, validate_model
        validate_data.run(cfg)
        validate_model.run(cfg)
    elif name == "report":
        from . import report; report.run(cfg)
    else:
        raise SystemExit(f"unknown stage {name}; choose from {STAGES}")
    log.info("stage %s done in %.1fs", name, time.time() - t0)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="superr", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run pipeline stages")
    r.add_argument("stages", nargs="+", help="'all' or stage names / numbers")
    r.add_argument("--config", default=None)
    r.add_argument("--set", action="append", default=[], help="override, e.g. --set pressure.k=0.5")
    r.add_argument("--skip-cv", action="store_true", help="use model.lambda_default instead of CV")
    r.add_argument("--skip-bootstrap", action="store_true")
    r.add_argument("--plan", action="store_true", help="jev stage: only estimate Jev requests and cost")
    r.add_argument("--log", default="INFO")
    p = sub.add_parser("plan-jev", help="estimate Jev requests / cost without calling it")
    p.add_argument("--config", default=None)
    p.add_argument("--set", action="append", default=[])
    args = ap.parse_args(argv)
    setup_logging(getattr(args, "log", "INFO"))
    cfg = load_config(args.config, _set_overrides(args.set))
    if args.cmd == "plan-jev":
        args.plan = True
        cfg["provider"] = {"wp": "jev", "p_next": "jev", "stakes": "jev"}
        run_stage("jev", cfg, args)
        return
    names = []
    for s in args.stages:
        if s == "all":
            names.extend(STAGES)
        elif s.isdigit():
            names.append(STAGES[int(s)])
        else:
            names.append(s)
    t0 = time.time()
    for n in names:
        run_stage(n, cfg, args)
    log.info("finished %s in %.1fs", names, time.time() - t0)


if __name__ == "__main__":
    main()
