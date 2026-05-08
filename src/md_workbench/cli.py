from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_workflow_config


def _config_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("config_positional", nargs="?", default=None)
    parser.add_argument("--config", default=None)
    return parser


def _print_json(payload) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))


def _load_config(args) -> tuple[object, str]:
    config_path = args.config or args.config_positional or "default_config.json"
    return load_workflow_config(config_path), config_path


def _run_logged(cfg, config_path: str, run_type: str, runner):
    from .core.runlog import start_run_log

    log_session = start_run_log(cfg.workspace_root, run_type)
    log_session.log(f"Config path: {Path(config_path).resolve()}")
    log_session.log_json("Workflow configuration", cfg.to_dict())
    try:
        outputs = runner(cfg)
        log_session.log_json("Workflow outputs", outputs)
        log_session.finalize("completed", payload=outputs)
        _print_json(outputs)
        return 0
    except Exception as exc:
        log_session.log_exception(exc)
        log_session.finalize("failed", error=str(exc))
        print(f"Run log: {log_session.log_path}")
        raise


def run_entry(argv: list[str] | None = None):
    parser = _config_parser("Run the full MD Workbench workflow")
    args = parser.parse_args(argv)
    cfg, config_path = _load_config(args)
    from .workflows import run_full_md_workflow

    return _run_logged(cfg, config_path, "full_workflow", run_full_md_workflow)


def next_replica_entry(argv: list[str] | None = None):
    parser = _config_parser("Prepare and run the next MD replica")
    args = parser.parse_args(argv)
    cfg, config_path = _load_config(args)
    from .workflows import prepare_next_replica_workflow_config, run_next_replica_workflow

    cfg = prepare_next_replica_workflow_config(cfg)
    return _run_logged(cfg, config_path, "next_replica_workflow", run_next_replica_workflow)


def existing_results_entry(argv: list[str] | None = None):
    parser = _config_parser("Process existing MD Workbench results")
    args = parser.parse_args(argv)
    cfg, config_path = _load_config(args)
    from .workflows import prepare_existing_results_workflow_config, run_existing_results_workflow

    cfg = prepare_existing_results_workflow_config(cfg)
    return _run_logged(cfg, config_path, "existing_results_workflow", run_existing_results_workflow)


def plot_entry(argv: list[str] | None = None):
    parser = _config_parser("Run MD Workbench plotting workflow")
    args = parser.parse_args(argv)
    cfg, config_path = _load_config(args)
    from .workflows import run_plot_workflow

    return _run_logged(cfg, config_path, "plot_workflow", run_plot_workflow)


def report_entry(argv: list[str] | None = None):
    parser = _config_parser("Build an MD Workbench report")
    parser.add_argument("--bundle-root", default=None)
    parser.add_argument("--figures-dir-name", default="figures_combined")
    parser.add_argument("--data-dir-name", default="process_data")
    parser.add_argument("--out", default="workflow_report.docx")
    args = parser.parse_args(argv)

    from .core.runlog import start_run_log
    from .report import build_workflow_report, build_workflow_report_for_config

    if args.bundle_root:
        bundle_root = Path(args.bundle_root).expanduser().resolve()
        log_session = start_run_log(bundle_root.parent, "report_workflow")
        log_session.log(f"Bundle root: {bundle_root}")
        try:
            outputs = build_workflow_report(
                bundle_root,
                figures_dir_name=args.figures_dir_name,
                data_dir_name=args.data_dir_name,
                report_docx_name=args.out,
            )
            log_session.log_json("Workflow outputs", outputs)
            log_session.finalize("completed", payload=outputs)
            _print_json(outputs)
            return 0
        except Exception as exc:
            log_session.log_exception(exc)
            log_session.finalize("failed", error=str(exc))
            print(f"Run log: {log_session.log_path}")
            raise

    cfg, config_path = _load_config(args)
    return _run_logged(cfg, config_path, "report_workflow", build_workflow_report_for_config)


def mmgbsa_entry(argv: list[str] | None = None):
    parser = _config_parser("Run MM/GBSA postprocess workflow")
    args = parser.parse_args(argv)
    cfg, config_path = _load_config(args)
    from .core.pathing import normalize_workflow_paths
    from .postprocess.mmgbsa import run_mmgbsa_postprocess

    cfg = normalize_workflow_paths(cfg)

    def runner(effective_cfg):
        return run_mmgbsa_postprocess(effective_cfg.mmgbsa, effective_cfg.plot_style)

    return _run_logged(cfg, config_path, "mmgbsa_postprocess", runner)


def parse_mmgbsa_entry(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Parse AmberTools MMPBSA.py final results")
    parser.add_argument("input_path", nargs="?", default="mmpbsa_FINAL_RESULTS.dat")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    from .postprocess.mmgbsa import parse_mmpbsa_results

    output_path = parse_mmpbsa_results(args.input_path, args.out)
    print(output_path)
    return 0


def self_check_entry(argv: list[str] | None = None):
    parser = _config_parser("Run MD Workbench self-check")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    cfg, config_path = _load_config(args)
    from .self_check import run_self_check, save_self_check_report

    def runner(effective_cfg):
        report = run_self_check(effective_cfg)
        if args.out:
            save_self_check_report(report, args.out)
        return report

    return _run_logged(cfg, config_path, "self_check", runner)


def gui_entry(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Launch the MD Workbench GUI")
    parser.parse_args(argv)
    from .gui import main as gui_main

    return gui_main()


def init_config_entry(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Write a default MD Workbench config JSON")
    parser.add_argument("output", nargs="?", default="default_config.json")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    output_path = Path(args.output)
    if output_path.exists() and not args.force:
        raise FileExistsError(f"Refusing to overwrite existing config: {output_path}. Use --force to overwrite.")

    from .config import WorkflowConfig

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(WorkflowConfig().to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(output_path.resolve())
    return 0


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="MD Workbench CLI")
    parser.add_argument(
        "mode",
        choices=[
            "run",
            "next-replica",
            "existing-results",
            "plot",
            "report",
            "mmgbsa",
            "parse-mmgbsa",
            "self-check",
            "gui",
            "init-config",
        ],
    )
    parser.add_argument("mode_args", nargs=argparse.REMAINDER, metavar="args")
    args = parser.parse_args(argv)

    handlers = {
        "run": run_entry,
        "next-replica": next_replica_entry,
        "existing-results": existing_results_entry,
        "plot": plot_entry,
        "report": report_entry,
        "mmgbsa": mmgbsa_entry,
        "parse-mmgbsa": parse_mmgbsa_entry,
        "self-check": self_check_entry,
        "gui": gui_entry,
        "init-config": init_config_entry,
    }
    return handlers[args.mode](args.mode_args)


if __name__ == "__main__":
    main()
