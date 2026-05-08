__version__ = "0.1.0"

__all__ = ["__version__", "run_self_check", "save_self_check_report"]


def __getattr__(name: str):
    if name in {"run_self_check", "save_self_check_report"}:
        from .self_check import run_self_check, save_self_check_report

        return {
            "run_self_check": run_self_check,
            "save_self_check_report": save_self_check_report,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
