from __future__ import annotations

import json
import queue
import threading
import traceback
from copy import deepcopy
from dataclasses import fields as dataclass_fields
from datetime import datetime
from pathlib import Path
import tkinter as tk
import tkinter.font as tkfont
from tkinter import filedialog, messagebox, simpledialog, ttk

from ..config import WorkflowConfig, load_workflow_config, save_workflow_config
from ..core import ProgressEvent, ensure_project_layout, preflight_validate, preflight_validate_existing_results, project_config_path, start_run_log
from ..workflows import (
    prepare_docking_only_workflow_config,
    prepare_existing_results_workflow_config,
    prepare_next_replica_workflow_config,
    run_existing_results_workflow,
    run_full_md_workflow,
    run_next_replica_workflow,
)
from .forms import DataclassFrame, _visible_fields
from .i18n import safe_text, tr


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.cfg = WorkflowConfig()
        self._outer: ttk.Frame | None = None
        self._ui_queue: queue.Queue[tuple[str, tuple]] = queue.Queue()
        self.status_var = tk.StringVar(value="")
        self.progress_percent_var = tk.DoubleVar(value=0.0)
        self.progress_text_var = tk.StringVar(value="0/1 · 0%")
        self.progress_detail_var = tk.StringVar(value="")
        self.subprogress_percent_var = tk.DoubleVar(value=0.0)
        self.subprogress_text_var = tk.StringVar(value="0/1 · 0%")
        self.subprogress_detail_var = tk.StringVar(value="")
        self._subprogress_eta_seconds: int | None = None
        self._workspace_root_log_value = ""
        self.display_mode = tk.StringVar(value="basic")
        self._run_active = False
        self.title(tr("app_title"))
        self._target_window_size = (1360, 940)
        self.minsize(1120, 760)
        self._set_default_font()
        self._setup_style()
        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close_requested)
        self.after_idle(self._place_window_on_screen)
        self.after(80, self._drain_ui_queue)

    @property
    def lang(self) -> str:
        return "en_US"

    def _set_default_font(self):
        try:
            default_font = tkfont.nametofont("TkDefaultFont")
            families = set(tkfont.families())
            for family in [
                "Arial",
                "Microsoft YaHei UI",
                "Microsoft YaHei",
                "PingFang SC",
                "Noto Sans CJK SC",
                "WenQuanYi Micro Hei",
                "SimHei",
                "Arial Unicode MS",
            ]:
                if family in families:
                    default_font.configure(family=family, size=10)
                    break
            text_font = tkfont.nametofont("TkTextFont")
            text_font.configure(**default_font.actual())
            heading_font = tkfont.nametofont("TkHeadingFont")
            heading_font.configure(**default_font.actual())
        except Exception:
            pass

    def _setup_style(self):
        style = ttk.Style(self)
        if "clam" in style.theme_names():
            style.theme_use("clam")
        bg = "#F6F7FB"
        card = "#FFFFFF"
        border = "#D7DCE5"
        text = "#172033"
        muted = "#5E6B82"
        accent = "#2F6FED"
        accent_dark = "#2253B5"
        success = "#1C8C5E"
        self.configure(background=bg)
        style.configure("TFrame", background=bg)
        style.configure("Card.TFrame", background=card)
        style.configure("TLabel", background=bg, foreground=text)
        style.configure("Muted.TLabel", background=card, foreground=muted)
        style.configure("Header.TLabel", background=bg, foreground=text, font=(None, 16, "bold"))
        style.configure("SubHeader.TLabel", background=bg, foreground=muted)
        style.configure("FieldLabel.TLabel", background=card, foreground=text, font=(None, 10, "bold"))
        style.configure("GroupHeader.TLabel", background=bg, foreground=text, font=(None, 11, "bold"))
        style.configure("GroupHint.TLabel", background=bg, foreground=muted)
        style.configure("TLabelFrame", background=card, bordercolor=border, borderwidth=1, relief="solid")
        style.configure("TLabelFrame.Label", background=card, foreground=text, font=(None, 10, "bold"))
        style.configure("TEntry", padding=7)
        style.configure("TCombobox", padding=4)
        style.configure("Toolbar.TButton", padding=(12, 8), background=card)
        style.map("Toolbar.TButton", background=[("active", "#EEF3FF")])
        style.configure("Accent.TButton", padding=(14, 9), background=accent, foreground="#FFFFFF", borderwidth=0)
        style.map("Accent.TButton", background=[("active", accent_dark), ("pressed", accent_dark)], foreground=[("disabled", "#DDE5F6")])
        style.configure("Success.TButton", padding=(14, 9), background=success, foreground="#FFFFFF", borderwidth=0)
        style.map("Success.TButton", background=[("active", "#146947"), ("pressed", "#146947")])
        style.configure("Small.TButton", padding=(8, 5))
        style.configure("TNotebook", background=bg, borderwidth=0)
        style.configure("TNotebook.Tab", padding=(16, 8), background="#E9EDF5", foreground=text)
        style.map("TNotebook.Tab", background=[("selected", card), ("active", "#F1F4FA")])
        style.configure("TCheckbutton", background=card, foreground=text)
        style.configure("Switch.TCheckbutton", background=card)
        style.configure("Status.TLabel", background="#EEF2F8", foreground=muted, padding=(10, 6))
        style.configure(
            "Accent.Horizontal.TProgressbar",
            troughcolor="#E4EAF5",
            bordercolor=border,
            background=accent,
            lightcolor=accent,
            darkcolor=accent,
            thickness=14,
        )
        style.configure("ProgressValue.TLabel", background=card, foreground=text, font=(None, 10, "bold"))

    def _clear_ui(self):
        if self._outer is not None:
            self._outer.destroy()
            self._outer = None

    def _place_window_on_screen(self):
        """Work around X11/Wayland sessions that place new Tk windows off-screen."""
        self.update_idletasks()
        target_width, target_height = self._target_window_size
        screen_width = max(self.winfo_screenwidth(), 1)
        screen_height = max(self.winfo_screenheight(), 1)
        width = min(target_width, screen_width)
        height = min(target_height, screen_height)
        x = max((screen_width - width) // 2, 0)
        y = max((screen_height - height) // 2, 0)
        self.geometry(f"{width}x{height}+{x}+{y}")
        self.deiconify()
        self.lift()

    def _build_ui(self, preserve_log: str = ""):
        self._clear_ui()
        self.title(tr("app_title", self.lang))
        outer = ttk.Frame(self, padding=12)
        outer.pack(fill="both", expand=True)
        self._outer = outer

        header = ttk.Frame(outer, style="Card.TFrame")
        header.pack(fill="x", pady=(0, 10))
        left = ttk.Frame(header, style="Card.TFrame")
        left.pack(side="left", fill="x", expand=True, padx=14, pady=12)
        ttk.Label(left, text=tr("app_title", self.lang), style="Header.TLabel").pack(anchor="w")
        ttk.Label(left, text=tr("app_subtitle", self.lang), style="SubHeader.TLabel").pack(anchor="w", pady=(4, 0))

        toolbar = ttk.Frame(outer, style="Card.TFrame")
        toolbar.pack(fill="x", pady=(0, 10))
        ttk.Button(toolbar, text=tr("open_project", self.lang), command=self.open_project, style="Toolbar.TButton").pack(side="left", padx=(12, 6), pady=10)
        ttk.Button(toolbar, text=tr("new_project", self.lang), command=self.create_project, style="Toolbar.TButton").pack(side="left", padx=6, pady=10)
        ttk.Button(toolbar, text=tr("save_project", self.lang), command=self.save_project, style="Toolbar.TButton").pack(side="left", padx=6, pady=10)
        self.run_md_button = ttk.Button(toolbar, text=tr("run_md", self.lang), command=self.run_md, style="Accent.TButton")
        self.run_md_button.pack(side="left", padx=(18, 6), pady=10)
        self.run_docking_only_button = ttk.Button(
            toolbar,
            text=tr("run_docking_only", self.lang),
            command=self.run_docking_only,
            style="Toolbar.TButton",
        )
        self.run_docking_only_button.pack(side="left", padx=6, pady=10)
        self.run_next_replica_button = ttk.Button(
            toolbar,
            text=tr("run_next_replica", self.lang),
            command=self.run_next_replica,
            style="Toolbar.TButton",
        )
        self.run_next_replica_button.pack(side="left", padx=6, pady=10)
        self.run_existing_results_button = ttk.Button(
            toolbar,
            text=tr("run_existing_results", self.lang),
            command=self.run_existing_results,
            style="Success.TButton",
        )
        self.run_existing_results_button.pack(side="left", padx=6, pady=10)
        self._set_run_controls_enabled(not self._run_active)

        mode_box = ttk.Frame(toolbar, style="Card.TFrame")
        mode_box.pack(side="right", padx=12, pady=8)
        ttk.Label(mode_box, text=tr("display_mode", self.lang), style="FieldLabel.TLabel").pack(side="left", padx=(0, 10))
        ttk.Button(
            mode_box,
            text=tr("basic_mode", self.lang),
            command=lambda: self.set_display_mode("basic"),
            style="Accent.TButton" if self.display_mode.get() == "basic" else "Toolbar.TButton",
        ).pack(side="left", padx=4)
        ttk.Button(
            mode_box,
            text=tr("advanced_mode", self.lang),
            command=lambda: self.set_display_mode("advanced"),
            style="Accent.TButton" if self.display_mode.get() == "advanced" else "Toolbar.TButton",
        ).pack(side="left", padx=4)

        root_row = ttk.LabelFrame(outer, text=tr("project_group", self.lang), padding=12)
        root_row.pack(fill="x", pady=(0, 10))
        self.workspace_root_var = tk.StringVar(value=self.cfg.workspace_root)
        ttk.Label(root_row, text=tr("project_root", self.lang), style="FieldLabel.TLabel").grid(row=0, column=0, sticky="w", padx=(4, 10), pady=2)
        self.workspace_root_entry = ttk.Entry(root_row, textvariable=self.workspace_root_var, width=96)
        self.workspace_root_entry.grid(row=0, column=1, sticky="ew", padx=4)
        self.workspace_root_entry.bind("<FocusOut>", self._on_workspace_root_committed)
        self.workspace_root_entry.bind("<Return>", self._on_workspace_root_committed)
        ttk.Button(root_row, text=tr("choose_directory", self.lang), command=self.choose_workspace_root, style="Small.TButton").grid(row=0, column=2, sticky="e", padx=(6, 4))
        ttk.Button(root_row, text=tr("open_location", self.lang), command=self._open_workspace_root, style="Small.TButton").grid(row=0, column=3, sticky="e", padx=(2, 4))
        ttk.Label(root_row, text=tr("project_hint", self.lang), style="Muted.TLabel").grid(row=1, column=1, columnspan=3, sticky="w", padx=4, pady=(6, 0))
        root_row.columnconfigure(1, weight=1)
        self._workspace_root_log_value = self.workspace_root_var.get().strip()

        self.flag_vars = {}
        switch_names = [
            "do_prep",
            "do_run_md",
            "do_basic_analysis",
            "do_waterbridge_analysis",
            "do_advanced_analysis",
            "do_mmgbsa_postprocess",
        ]
        if self.display_mode.get() == "advanced":
            flags = ttk.LabelFrame(outer, text=tr("workflow_flags", self.lang), padding=12)
            flags.pack(fill="x", pady=(0, 10))
            for idx, name in enumerate(switch_names):
                var = tk.BooleanVar(value=getattr(self.cfg, name))
                card = ttk.Frame(flags, style="Card.TFrame")
                card.grid(row=0, column=idx, sticky="ew", padx=5, pady=4)
                ttk.Checkbutton(
                    card,
                    text=tr(name, self.lang),
                    variable=var,
                    command=lambda flag_name=name, flag_var=var: self._log_workflow_flag_toggle(flag_name, bool(flag_var.get())),
                ).pack(anchor="w", padx=8, pady=6)
                self.flag_vars[name] = var
                flags.columnconfigure(idx, weight=1)
        else:
            defaults_card = ttk.Frame(outer, style="Card.TFrame")
            defaults_card.pack(fill="x", pady=(0, 10))
            ttk.Label(
                defaults_card,
                text=tr("basic_defaults_hint", self.lang),
                style="Muted.TLabel",
                wraplength=1120,
                justify="left",
            ).pack(anchor="w", padx=14, pady=10)
            for name in switch_names:
                self.flag_vars[name] = tk.BooleanVar(value=getattr(self.cfg, name))

        paned = ttk.Panedwindow(outer, orient="vertical")
        paned.pack(fill="both", expand=True)

        upper = ttk.Frame(paned)
        lower = ttk.Frame(paned)
        paned.add(upper, weight=5)
        paned.add(lower, weight=2)

        notebook = ttk.Notebook(upper)
        notebook.pack(fill="both", expand=True)
        self.frames = {}
        pages = [
            ("page_prep", "prep"),
            ("page_docking", "docking"),
            ("page_run", "run"),
            ("page_basic", "basic"),
            ("page_waterbridge", "waterbridge"),
            ("page_advanced", "advanced"),
            ("page_plot_selection", "plot_selection"),
            ("page_plot_style", "plot_style"),
            ("page_output_bundle", "output_bundle"),
            ("page_mmgbsa", "mmgbsa"),
        ]
        for title_key, section_key in pages:
            obj = getattr(self.cfg, section_key)
            visible = _visible_fields(section_key, list(dataclass_fields(obj)), self.display_mode.get(), obj)
            if self.display_mode.get() == "basic" and len(visible) == 0:
                continue
            page = ttk.Frame(notebook, padding=0)
            notebook.add(page, text=tr(title_key, self.lang))
            frame = DataclassFrame(
                page,
                section_key,
                obj,
                self.lang,
                workspace_root_getter=lambda: self.workspace_root_var.get().strip() or ".",
                display_mode_getter=lambda: self.display_mode.get(),
                action_logger=self.log_user_action,
            )
            frame.pack(fill="both", expand=True)
            self.frames[section_key] = frame

        log_box = ttk.LabelFrame(lower, text=tr("log", self.lang), padding=10)
        log_box.pack(fill="both", expand=True)
        self.log = tk.Text(log_box, height=11, wrap="word", relief="flat", bd=0)
        self.log.pack(fill="both", expand=True)
        if preserve_log:
            self.log.insert("1.0", preserve_log)
            self.log.see("end")

        status_row = ttk.Frame(outer, style="Card.TFrame")
        status_row.pack(fill="x", pady=(10, 0))
        self.status_var.set(tr("status_ready", self.lang))
        ttk.Label(status_row, textvariable=self.status_var, style="Status.TLabel").pack(fill="x", padx=10, pady=(10, 6))

        progress_row = ttk.Frame(status_row, style="Card.TFrame")
        progress_row.pack(fill="x", padx=14, pady=(0, 6))
        ttk.Label(progress_row, text=tr("progress", self.lang), style="FieldLabel.TLabel").pack(side="left", padx=(0, 12))
        ttk.Progressbar(
            progress_row,
            orient="horizontal",
            mode="determinate",
            maximum=100,
            variable=self.progress_percent_var,
            style="Accent.Horizontal.TProgressbar",
        ).pack(side="left", fill="x", expand=True)
        ttk.Label(progress_row, textvariable=self.progress_text_var, style="ProgressValue.TLabel").pack(side="left", padx=(12, 0))
        ttk.Label(
            status_row,
            textvariable=self.progress_detail_var,
            style="Muted.TLabel",
            wraplength=1120,
            justify="left",
        ).pack(fill="x", padx=14, pady=(0, 10))

        subprogress_row = ttk.Frame(status_row, style="Card.TFrame")
        subprogress_row.pack(fill="x", padx=14, pady=(0, 6))
        ttk.Label(subprogress_row, text=tr("subprogress", self.lang), style="FieldLabel.TLabel").pack(side="left", padx=(0, 12))
        ttk.Progressbar(
            subprogress_row,
            orient="horizontal",
            mode="determinate",
            maximum=100,
            variable=self.subprogress_percent_var,
            style="Accent.Horizontal.TProgressbar",
        ).pack(side="left", fill="x", expand=True)
        ttk.Label(subprogress_row, textvariable=self.subprogress_text_var, style="ProgressValue.TLabel").pack(side="left", padx=(12, 0))
        ttk.Label(
            status_row,
            textvariable=self.subprogress_detail_var,
            style="Muted.TLabel",
            wraplength=1120,
            justify="left",
        ).pack(fill="x", padx=14, pady=(0, 10))
        self._set_progress_widgets(0, 1, tr("progress_idle", self.lang))
        self._set_subprogress_widgets(0, 0, tr("subprogress_idle", self.lang))

    def _project_config_path(self) -> Path:
        root = Path(self.workspace_root_var.get().strip() or self.cfg.workspace_root or ".").expanduser().resolve()
        return project_config_path(root)

    def open_project(self):
        current = self.workspace_root_var.get().strip() or "."
        initial = current if Path(current).exists() else "."
        selected = filedialog.askdirectory(initialdir=initial)
        if not selected:
            return
        project_root = ensure_project_layout(selected)
        cfg_path = project_config_path(project_root)
        if cfg_path.exists():
            self.cfg = load_workflow_config(cfg_path)
        else:
            self.cfg = WorkflowConfig()
            self.cfg.workspace_root = str(project_root)
        self.cfg.workspace_root = str(project_root)
        preserve_log = self.log.get("1.0", "end-1c") if hasattr(self, "log") else ""
        self._build_ui(preserve_log=preserve_log)
        self.log_user_action(f"{tr('project_loaded', self.lang)}: {project_root}")

    def create_project(self):
        base_dir = filedialog.askdirectory(initialdir=Path(self.workspace_root_var.get().strip() or '.').expanduser().resolve().parent)
        if not base_dir:
            return
        name = simpledialog.askstring(tr('new_project_title', self.lang), tr('new_project_name_prompt', self.lang), parent=self)
        if not name:
            return
        safe_name = ''.join(ch if ch.isalnum() or ch in {'-', '_'} else '_' for ch in name).strip('._')
        if not safe_name:
            messagebox.showerror(tr('run_failed', self.lang), 'Project name is empty after sanitization.')
            return
        project_root = Path(base_dir) / safe_name
        if project_root.exists() and any(project_root.iterdir()):
            messagebox.showerror(tr('run_failed', self.lang), f'Project folder already exists and is not empty: {project_root}')
            return
        ensure_project_layout(project_root)
        self.cfg = WorkflowConfig()
        self.cfg.workspace_root = str(project_root.resolve())
        cfg_to_save = deepcopy(self.cfg)
        cfg_to_save.workspace_root = "."
        save_workflow_config(cfg_to_save, project_config_path(project_root))
        preserve_log = self.log.get("1.0", "end-1c") if hasattr(self, "log") else ""
        self._build_ui(preserve_log=preserve_log)
        self.log_user_action(f"{tr('project_created', self.lang)}: {project_root}")

    def save_project(self):
        self.sync_to_cfg()
        project_root = ensure_project_layout(self.cfg.workspace_root)
        cfg_path = project_config_path(project_root)
        cfg_to_save = deepcopy(self.cfg)
        cfg_to_save.workspace_root = "."
        save_workflow_config(cfg_to_save, cfg_path)
        self.log_user_action(f"{tr('project_saved', self.lang)}: {cfg_path}")

    def _open_workspace_root(self):
        from .forms import _open_location
        self.log_user_action(f"Opened project folder location: {self.workspace_root_var.get().strip() or '.'}")
        _open_location(self.workspace_root_var.get(), self.workspace_root_var.get() or ".")

    def choose_workspace_root(self):
        current = self.workspace_root_var.get().strip() or "."
        initial = current if Path(current).exists() else "."
        selected = filedialog.askdirectory(initialdir=initial)
        if selected:
            self.workspace_root_var.set(selected)
            self.status_var.set(tr("status_config_synced", self.lang))
            self._workspace_root_log_value = selected.strip()
            self.log_user_action(f"Changed project folder: {selected}")

    def append_log(self, text: str):
        self.log.insert("end", safe_text(text) + "\n")
        self.log.see("end")
        self.update_idletasks()

    def _timestamp(self) -> str:
        return datetime.now().strftime("%H:%M:%S")

    def _format_log_entry(self, kind: str, text: str) -> str:
        return f"[{self._timestamp()}] [{kind}] {safe_text(text)}"

    def log_user_action(self, text: str):
        self.append_log(self._format_log_entry("USER", text))

    def log_runtime_event(self, text: str):
        self.append_log(self._format_log_entry("RUN", text))

    def _set_run_controls_enabled(self, enabled: bool):
        if hasattr(self, "run_md_button"):
            self.run_md_button.state(["!disabled"] if enabled else ["disabled"])
        if hasattr(self, "run_docking_only_button"):
            self.run_docking_only_button.state(["!disabled"] if enabled else ["disabled"])
        if hasattr(self, "run_next_replica_button"):
            self.run_next_replica_button.state(["!disabled"] if enabled else ["disabled"])
        if hasattr(self, "run_existing_results_button"):
            self.run_existing_results_button.state(["!disabled"] if enabled else ["disabled"])

    def _set_run_state(self, is_running: bool):
        self._run_active = is_running
        self._set_run_controls_enabled(not is_running)

    def _set_progress_widgets(self, current: int, total: int, detail: str):
        safe_total = max(int(total), 1)
        safe_current = min(max(int(current), 0), safe_total)
        percent = 100.0 * safe_current / safe_total
        self.progress_percent_var.set(percent)
        self.progress_text_var.set(f"{safe_current}/{safe_total} · {percent:.0f}%")
        self.progress_detail_var.set(safe_text(detail))

    def _format_eta(self, eta_seconds: int) -> str:
        remaining = max(int(eta_seconds), 0)
        if remaining < 60:
            return f"{remaining}s"
        minutes, seconds = divmod(remaining, 60)
        if minutes < 60:
            return f"{minutes}m {seconds:02d}s"
        hours, minutes = divmod(minutes, 60)
        if hours < 24:
            return f"{hours}h {minutes:02d}m"
        days, hours = divmod(hours, 24)
        return f"{days}d {hours:02d}h"

    def _set_subprogress_widgets(self, current: int, total: int, detail: str, eta_seconds: int | None = None):
        safe_total = max(int(total), 1) if int(total) > 0 else 1
        safe_current = min(max(int(current), 0), safe_total)
        percent = 0.0 if int(total) <= 0 else 100.0 * safe_current / safe_total
        self.subprogress_percent_var.set(percent)
        text = "0/0 · 0%" if int(total) <= 0 else f"{safe_current}/{safe_total} · {percent:.0f}%"
        if eta_seconds is not None:
            text += f" · ETA {self._format_eta(eta_seconds)}"
        self.subprogress_text_var.set(text)
        self.subprogress_detail_var.set(safe_text(detail))

    def _log_workflow_flag_toggle(self, name: str, value: bool):
        state = "Enabled" if value else "Disabled"
        self.log_user_action(f"Toggled workflow flag {tr(name, self.lang)}: {state}")

    def _on_workspace_root_committed(self, _event=None):
        value = self.workspace_root_var.get().strip()
        if value and value != self._workspace_root_log_value:
            self._workspace_root_log_value = value
            self.log_user_action(f"Updated project folder: {value}")

    def set_display_mode(self, mode: str):
        if mode not in {"basic", "advanced"}:
            return
        for frame in self.frames.values():
            frame.write_back()
        self.display_mode.set(mode)
        self.status_var.set(f"{tr('status_config_synced', self.lang)} · {tr(mode + '_mode', self.lang)}")
        preserve = self.log.get("1.0", "end-1c") if hasattr(self, "log") else ""
        self._build_ui(preserve_log=preserve)
        self.log_user_action(f"Switched display mode to {tr(mode + '_mode', self.lang)}")

    def sync_to_cfg(self):
        self.cfg.workspace_root = self.workspace_root_var.get().strip() or "."
        for key, frame in self.frames.items():
            frame.write_back()
            setattr(self.cfg, key, frame.obj)
        for name, var in self.flag_vars.items():
            setattr(self.cfg, name, bool(var.get()))
        self.status_var.set(tr("status_config_synced", self.lang))

    def load_config(self):
        path = filedialog.askopenfilename(initialdir=self.workspace_root_var.get().strip() or ".", filetypes=[("JSON", "*.json")])
        if not path:
            return
        self.cfg = load_workflow_config(path)
        if str(self.cfg.workspace_root).strip() in {"", "."}:
            self.cfg.workspace_root = str(Path(path).resolve().parent)
        preserve_log = self.log.get("1.0", "end-1c") if hasattr(self, "log") else ""
        self._build_ui(preserve_log=preserve_log)
        self.log_user_action(f"{tr('config_loaded', self.lang)}: {path}")

    def save_config(self):
        self.sync_to_cfg()
        path = filedialog.asksaveasfilename(initialdir=self.workspace_root_var.get().strip() or ".", initialfile="project_config.json", defaultextension=".json", filetypes=[("JSON", "*.json")])
        if not path:
            return
        save_workflow_config(self.cfg, path)
        self.log_user_action(f"{tr('config_saved', self.lang)}: {path}")

    def _basic_mode_validation_errors(self, cfg: WorkflowConfig) -> list[str]:
        if self.display_mode.get() != "basic":
            return []
        errors: list[str] = []
        if cfg.do_prep and str(cfg.docking.docking_mode).strip().lower() == "auto":
            box_center = [
                cfg.docking.search_center_x,
                cfg.docking.search_center_y,
                cfg.docking.search_center_z,
            ]
            if any(value is None for value in box_center):
                errors.append(
                    "Basic mode requires docking box center X, Y, and Z. Box sizes are optional and will use the default values."
                )
        if not cfg.plot_style.formats:
            errors.append("Basic mode requires at least one image output format, for example png or svg.")
        return errors

    def _run_in_thread(self, func, label_key: str, *, cfg_transform=None, validator=None):
        if self._run_active:
            self.log_runtime_event(f"{tr('status_running', self.lang)}: another workflow is already in progress.")
            return
        log_session = None
        effective_cfg = None
        try:
            self.sync_to_cfg()
            effective_cfg = cfg_transform(self.cfg) if cfg_transform is not None else deepcopy(self.cfg)
            self._set_run_state(True)
            log_session = start_run_log(effective_cfg.workspace_root, label_key)
            log_session.log_json("Workflow configuration", effective_cfg.to_dict())
            self.log_user_action(f"Requested workflow run: {tr(label_key, self.lang)}")
            self.log_runtime_event(f"Run log: {log_session.log_path}")
            self._set_progress_widgets(0, 1, tr("progress_queued", self.lang))
            self._subprogress_eta_seconds = None
            self._set_subprogress_widgets(0, 0, tr("subprogress_idle", self.lang))
            validate = validator or preflight_validate
            preflight = validate(effective_cfg)
            basic_mode_errors = self._basic_mode_validation_errors(effective_cfg)
            for message in basic_mode_errors:
                if message not in preflight.errors:
                    preflight.errors.append(message)
            if preflight.warnings:
                log_session.log_json("Preflight warnings", preflight.warnings, level="WARNING")
            for warning in preflight.warnings:
                self.log_runtime_event(f"Warning: {warning}")
            if preflight.errors:
                message = "\n".join(f"- {item}" for item in preflight.errors)
                self.log_runtime_event("Preflight validation failed:\n" + message)
                if log_session is not None:
                    log_session.log_json("Preflight errors", preflight.errors, level="ERROR")
                    log_session.finalize("preflight_failed", payload={"warnings": preflight.warnings, "errors": preflight.errors}, error=message)
                self.status_var.set(tr("run_failed", self.lang))
                self._set_progress_widgets(0, 1, "Preflight validation failed")
                self._subprogress_eta_seconds = None
                self._set_subprogress_widgets(0, 0, tr("subprogress_idle", self.lang))
                self._set_run_state(False)
                messagebox.showerror(tr("run_failed", self.lang), safe_text(message))
                return
        except Exception as exc:
            tb = traceback.format_exc()
            self.append_log(tb)
            if log_session is None:
                try:
                    workspace_root = effective_cfg.workspace_root if effective_cfg is not None else self.cfg.workspace_root
                    log_session = start_run_log(workspace_root, label_key)
                except Exception:
                    log_session = None
            if log_session is not None:
                log_session.log_exception(exc)
                log_session.finalize("failed", error=str(exc))
            self.status_var.set(tr("run_failed", self.lang))
            self._set_progress_widgets(0, 1, tr("run_failed", self.lang))
            self._subprogress_eta_seconds = None
            self._set_subprogress_widgets(0, 0, tr("subprogress_idle", self.lang))
            self._set_run_state(False)
            messagebox.showerror(tr("run_failed", self.lang), safe_text(str(exc)))
            return

        def target():
            progress_state = {"current": 0, "total": 1, "detail": tr("progress_queued", self.lang)}
            last_logged_progress: tuple[int, int, str, str, int, int, str, int | None] | None = None

            def progress_callback(event: ProgressEvent) -> None:
                nonlocal last_logged_progress
                progress_state["current"] = event.current
                progress_state["total"] = event.total
                progress_state["detail"] = event.detail
                self._set_progress_async(event)
                signature = (
                    event.current,
                    event.total,
                    event.stage,
                    event.detail,
                    event.subcurrent,
                    event.subtotal,
                    event.subdetail,
                    event.subeta_seconds,
                )
                if signature != last_logged_progress and log_session is not None:
                    log_session.log_progress(event)
                    last_logged_progress = signature

            try:
                assert log_session is not None
                self._set_status_async(f"{tr('status_running', self.lang)} · {tr(label_key, self.lang)}")
                self._append_log_async(self._format_log_entry("RUN", f"{tr('start_running', self.lang)}: {tr(label_key, self.lang)}"))
                log_session.log(f"Executing workflow callable for {label_key}")
                outputs = func(effective_cfg, progress_callback=progress_callback)
                log_session.log_json("Workflow outputs", outputs)
                log_session.finalize("completed", payload=outputs)
                self._append_log_async(json.dumps(outputs, ensure_ascii=False, indent=2, default=str))
                self._append_log_async(self._format_log_entry("RUN", f"{tr('completed', self.lang)}: {tr(label_key, self.lang)}"))
                self._set_status_async(f"{tr('completed', self.lang)} · {tr(label_key, self.lang)}")
                self._set_progress_async(
                    ProgressEvent(
                        current=progress_state["total"],
                        total=progress_state["total"],
                        stage=label_key,
                        detail=f"{tr('completed', self.lang)}: {tr(label_key, self.lang)}",
                    )
                )
            except Exception as exc:
                tb = traceback.format_exc()
                if log_session is not None:
                    log_session.log_exception(exc)
                    log_session.finalize("failed", error=str(exc))
                self._append_log_async(tb)
                self._append_log_async(self._format_log_entry("RUN", f"Run log: {log_session.log_path if log_session is not None else 'unavailable'}"))
                self._set_status_async(tr("run_failed", self.lang))
                self._set_progress_async(
                    ProgressEvent(
                        current=progress_state["current"],
                        total=progress_state["total"],
                        stage=label_key,
                        detail=f"{tr('run_failed', self.lang)}: {progress_state['detail']}",
                    )
                )
                self._show_error_async(tr("run_failed", self.lang), str(exc))
            finally:
                self._set_run_state_async(False)

        threading.Thread(target=target, daemon=False).start()

    def _append_log_async(self, text: str):
        self._ui_queue.put(("append_log", (text,)))

    def _set_status_async(self, text: str):
        self._ui_queue.put(("set_status", (safe_text(text),)))

    def _set_run_state_async(self, is_running: bool):
        self._ui_queue.put(("set_run_state", (is_running,)))

    def _set_progress_async(self, event: ProgressEvent):
        self._ui_queue.put(("set_progress", (event,)))

    def _show_error_async(self, title: str, message: str):
        self._ui_queue.put(("show_error", (safe_text(title), safe_text(message))))

    def _drain_ui_queue(self):
        while True:
            try:
                action, args = self._ui_queue.get_nowait()
            except queue.Empty:
                break
            if action == "append_log":
                self.append_log(*args)
            elif action == "set_status":
                self.status_var.set(*args)
            elif action == "set_run_state":
                self._set_run_state(*args)
            elif action == "set_progress":
                self._handle_progress_event(*args)
            elif action == "show_error":
                messagebox.showerror(*args)
        self.after(80, self._drain_ui_queue)

    def _handle_progress_event(self, event: ProgressEvent):
        self._set_progress_widgets(event.current, event.total, event.detail)
        if event.subtotal > 0:
            detail = event.subdetail or event.detail
            show_eta = event.stage == "md" and event.subdetail.startswith("Production:") and event.subeta_seconds is not None
            self._subprogress_eta_seconds = event.subeta_seconds if show_eta else None
            self._set_subprogress_widgets(event.subcurrent, event.subtotal, detail, self._subprogress_eta_seconds)
        else:
            self._subprogress_eta_seconds = None
            self._set_subprogress_widgets(0, 0, tr("subprogress_idle", self.lang))
        self.log_runtime_event(
            f"{event.stage}: {event.detail}"
            + (
                f" | {event.subdetail or event.detail} ({event.subcurrent}/{event.subtotal})"
                + (
                    f", ETA {self._format_eta(event.subeta_seconds)}"
                    if event.stage == "md" and event.subdetail.startswith("Production:") and event.subeta_seconds is not None
                    else ""
                )
                if event.subtotal > 0
                else ""
            )
        )

    def run_md(self):
        self._run_in_thread(run_full_md_workflow, "full_workflow")

    def run_docking_only(self):
        self._run_in_thread(
            run_full_md_workflow,
            "docking_only_workflow",
            cfg_transform=prepare_docking_only_workflow_config,
        )

    def run_next_replica(self):
        self._run_in_thread(
            run_next_replica_workflow,
            "next_replica_workflow",
            cfg_transform=prepare_next_replica_workflow_config,
        )

    def run_existing_results(self):
        self._run_in_thread(
            run_existing_results_workflow,
            "existing_results_workflow",
            cfg_transform=prepare_existing_results_workflow_config,
            validator=preflight_validate_existing_results,
        )

    def _on_close_requested(self):
        if self._run_active and not messagebox.askyesno(
            "Workflow running",
            "A workflow is still running. Closing the window will keep the Python process alive until the run finishes so the log can be finalized. Close the window anyway?",
            parent=self,
        ):
            return
        self.destroy()


def main():
    app = App()
    app.mainloop()
