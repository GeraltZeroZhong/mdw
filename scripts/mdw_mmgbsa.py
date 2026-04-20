#!/usr/bin/env python3
from pathlib import Path
import sys
import argparse
import json

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from md_workbench.config import load_workflow_config
from md_workbench.core import start_run_log
from md_workbench.postprocess.mmgbsa import run_mmgbsa_postprocess

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_positional", nargs="?", default=None)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    default_cfg = Path(__file__).resolve().parents[1] / "default_config.json"
    config_path = args.config or args.config_positional or str(default_cfg)
    cfg = load_workflow_config(config_path)
    log_session = start_run_log(cfg.workspace_root, "mmgbsa_postprocess")
    log_session.log(f"Config path: {Path(config_path).resolve()}")
    log_session.log_json("Workflow configuration", cfg.to_dict())
    try:
        outputs = run_mmgbsa_postprocess(cfg.mmgbsa, cfg.plot_style)
        log_session.log_json("Workflow outputs", outputs)
        log_session.finalize("completed", payload=outputs)
        print(json.dumps(outputs, ensure_ascii=False, indent=2, default=str))
    except Exception as exc:
        log_session.log_exception(exc)
        log_session.finalize("failed", error=str(exc))
        print(f"Run log: {log_session.log_path}", file=sys.stderr)
        raise
