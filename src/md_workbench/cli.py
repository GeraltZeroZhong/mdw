from __future__ import annotations

import argparse

from .config import load_workflow_config
from .core import start_run_log
from .gui import main as gui_main
from .self_check import run_self_check
from .workflows import (
    prepare_existing_results_workflow_config,
    prepare_next_replica_workflow_config,
    run_existing_results_workflow,
    run_full_md_workflow,
    run_next_replica_workflow,
    run_plot_workflow,
)


def main():
    parser = argparse.ArgumentParser(description="MD Workbench CLI")
    parser.add_argument("mode", choices=["run", "next-replica", "existing-results", "plot", "gui", "self-check"])
    parser.add_argument("config_positional", nargs="?", default=None)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    if args.mode == "gui":
        gui_main()
        return

    config_path = args.config or args.config_positional or "default_config.json"
    cfg = load_workflow_config(config_path)
    if args.mode == "existing-results":
        effective_cfg = prepare_existing_results_workflow_config(cfg)
    elif args.mode == "next-replica":
        effective_cfg = prepare_next_replica_workflow_config(cfg)
    else:
        effective_cfg = cfg
    run_type = {
        "run": "full_workflow",
        "next-replica": "next_replica_workflow",
        "existing-results": "existing_results_workflow",
        "plot": "plot_workflow",
        "self-check": "self_check",
    }[args.mode]
    log_session = start_run_log(effective_cfg.workspace_root, run_type)
    log_session.log(f"Config path: {config_path}")
    log_session.log_json("Workflow configuration", effective_cfg.to_dict())
    try:
        if args.mode == "run":
            outputs = run_full_md_workflow(effective_cfg)
        elif args.mode == "next-replica":
            outputs = run_next_replica_workflow(effective_cfg)
        elif args.mode == "existing-results":
            outputs = run_existing_results_workflow(effective_cfg)
        elif args.mode == "plot":
            outputs = run_plot_workflow(effective_cfg)
        else:
            outputs = run_self_check(effective_cfg)
        log_session.log_json("Workflow outputs", outputs)
        log_session.finalize("completed", payload=outputs)
        print(outputs)
    except Exception as exc:
        log_session.log_exception(exc)
        log_session.finalize("failed", error=str(exc))
        raise


if __name__ == "__main__":
    main()
