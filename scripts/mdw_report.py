#!/usr/bin/env python3
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from md_workbench.config import load_workflow_config
from md_workbench.core import start_run_log
from md_workbench.report import build_workflow_report, build_workflow_report_for_config


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_positional", nargs="?", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--bundle-root", default=None)
    parser.add_argument("--figures-dir-name", default="figures_combined")
    parser.add_argument("--data-dir-name", default="process_data")
    parser.add_argument("--out", default="workflow_report.docx")
    args = parser.parse_args()

    if args.bundle_root:
        bundle_root = Path(args.bundle_root).expanduser().resolve()
        log_session = start_run_log(bundle_root.parent, "report_workflow")
        log_session.log(f"Bundle root: {bundle_root}")
        cfg = None
    else:
        default_cfg = Path(__file__).resolve().parents[1] / "default_config.json"
        config_path = args.config or args.config_positional or str(default_cfg)
        cfg = load_workflow_config(config_path)
        log_session = start_run_log(cfg.workspace_root, "report_workflow")
        log_session.log(f"Config path: {Path(config_path).resolve()}")
        log_session.log_json("Workflow configuration", cfg.to_dict())
    try:
        if args.bundle_root:
            outputs = build_workflow_report(
                bundle_root,
                figures_dir_name=args.figures_dir_name,
                data_dir_name=args.data_dir_name,
                report_docx_name=args.out,
            )
        else:
            outputs = build_workflow_report_for_config(cfg)
        log_session.log_json("Workflow outputs", outputs)
        log_session.finalize("completed", payload=outputs)
        print(json.dumps(outputs, ensure_ascii=False, indent=2, default=str))
    except Exception as exc:
        log_session.log_exception(exc)
        log_session.finalize("failed", error=str(exc))
        print(f"Run log: {log_session.log_path}", file=sys.stderr)
        raise
