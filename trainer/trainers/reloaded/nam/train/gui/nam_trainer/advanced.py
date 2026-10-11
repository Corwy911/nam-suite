"""Advanced Options dialog — a Toplevel with grouped form fields."""
import tkinter as tk
from tkinter import ttk

from .style import (
    FT_DISPLAY, FT_EYEBROW, FT_BODY, FT_LABEL, FT_MONO,
)
from .widgets import eyebrow, divider


def open_advanced(parent, P):
    """Open (or focus) the Advanced Options window."""
    # Single-instance behaviour
    if getattr(parent, "_adv_win", None) is not None:
        try:
            parent._adv_win.lift()
            parent._adv_win.focus_force()
            return
        except tk.TclError:
            parent._adv_win = None

    adv = tk.Toplevel(parent)
    adv.title("Advanced Options")
    adv.configure(bg=P["BG"])
    adv.geometry("460x780")
    adv.minsize(420, 640)
    adv.transient(parent)
    parent._adv_win = adv

    def _on_close():
        parent._adv_win = None
        adv.destroy()
    adv.protocol("WM_DELETE_WINDOW", _on_close)

    body = ttk.Frame(adv, style="TFrame")
    body.pack(fill="both", expand=True, padx=24, pady=(20, 18))

    # ---- Title ----
    ttk.Label(body, text="Advanced options",
              style="Display.TLabel").pack(anchor="w")
    eyebrow(body, "Architecture · Schedule · Stages", P, prefix=""
        ).pack(anchor="w", pady=(2, 14))

    # ---- Helpers ----
    def field_row(parent_frame, r, label, widget, full_width=True):
        """One form row: label | widget | 1-px divider underneath.

        full_width=True  → widget stretches across column (use for combos)
        full_width=False → widget keeps its declared width (use for entries)
        """
        ttk.Label(parent_frame, text=label, style="Muted.TLabel"
            ).grid(row=r * 2, column=0, sticky="w", padx=(0, 14), pady=6)
        sticky = "ew" if full_width else "w"
        widget.grid(row=r * 2, column=1, sticky=sticky, pady=6)
        divider(parent_frame, P).grid(
            row=r * 2 + 1, column=0, columnspan=2, sticky="ew")

    def entry(parent_frame, var, width=14):
        # Entries are fixed-width — numbers don't need 250 px of space.
        return ttk.Entry(parent_frame, textvariable=var,
                         width=width, font=FT_MONO, justify="left")

    def combo(parent_frame, var, values, width=22):
        # Combos stretch — the label text varies in length.
        c = ttk.Combobox(parent_frame, textvariable=var, values=values,
                         state="readonly", width=width, font=FT_BODY)
        return c

    # Convenience wrappers so the section blocks below stay readable:
    def row_combo(parent_frame, r, label, var, values):
        field_row(parent_frame, r, label,
                  combo(parent_frame, var, values), full_width=True)

    def row_entry(parent_frame, r, label, var):
        field_row(parent_frame, r, label,
                  entry(parent_frame, var), full_width=False)

    # ---- Variables (would be wired up to your real model in a full app) ----
    v_arch    = tk.StringVar(value="ComplexRF300Lite")
    v_epochs  = tk.StringVar(value="600")
    v_sched   = tk.StringVar(value="exponential")
    v_lr      = tk.StringVar(value="0.005")
    v_decay   = tk.StringVar(value="0.0028")
    v_batch   = tk.StringVar(value="32")
    v_ny      = tk.StringVar(value="8192")
    v_latency = tk.StringVar(value="")
    v_thresh  = tk.StringVar(value="")

    v_mode    = tk.StringVar(value="two_stage")
    v_s2_ep   = tk.StringVar(value="100")
    v_s2_sch  = tk.StringVar(value="reduce_on_plateau")
    v_s2_lr   = tk.StringVar(value="0.00067")
    v_s2_dec  = tk.StringVar(value="0.77")
    v_s2_focus = tk.StringVar(value="Lows")
    v_ckpt    = tk.StringVar(value="Minimal")

    # ---- Core section ----
    eyebrow(body, "Core", P).pack(anchor="w", pady=(0, 4))
    core = ttk.Frame(body, style="TFrame")
    core.pack(fill="x", pady=(0, 14))
    core.columnconfigure(1, weight=1)

    field_row(core, 0, "Architecture",
              combo(core, v_arch, ["ComplexRF300Lite", "Standard", "Lite"]),
              full_width=True)
    row_entry(core, 1, "Epochs",        v_epochs)
    row_combo(core, 2, "LR scheduler",  v_sched,
              ["exponential", "step", "cosine"])
    row_entry(core, 3, "Learning rate", v_lr)
    row_entry(core, 4, "LR decay",      v_decay)
    row_entry(core, 5, "Batch size",    v_batch)
    row_entry(core, 6, "NY",            v_ny)
    row_entry(core, 7, "Reamp latency", v_latency)
    row_entry(core, 8, "Threshold ESR", v_thresh)

    # ---- Stage 2 section ----
    eyebrow(body, "Stage 2", P).pack(anchor="w", pady=(0, 4))
    stg = ttk.Frame(body, style="TFrame")
    stg.pack(fill="x", pady=(0, 10))
    stg.columnconfigure(1, weight=1)

    row_combo(stg, 0, "Training mode",     v_mode,
              ["one_stage", "two_stage"])
    row_entry(stg, 1, "Stage 2 epochs",    v_s2_ep)
    row_combo(stg, 2, "Stage 2 scheduler", v_s2_sch,
              ["reduce_on_plateau",
               "cosine_annealing_warm_restarts",
               "exponential"])
    row_entry(stg, 3, "Stage 2 LR",        v_s2_lr)
    row_entry(stg, 4, "Stage 2 LR decay",  v_s2_dec)
    row_combo(stg, 5, "Stage 2 focus",     v_s2_focus,
              ["Lows", "Mids", "Highs", "Full"])
    row_combo(stg, 6, "Checkpoint saving", v_ckpt,
              ["None", "Minimal", "Every epoch"])

    # ---- Footer ----
    divider(body, P).pack(fill="x", pady=(10, 12))
    foot = ttk.Frame(body, style="TFrame")
    foot.pack(fill="x")
    # All three footer buttons share the same character width so they
    # render at identical pixel widths regardless of label length.
    BTN_W = 14
    ttk.Button(foot, text="Save Preset…", style="Ghost.TButton", width=BTN_W
        ).pack(side="left", padx=(0, 8))
    ttk.Button(foot, text="Load Preset…", style="Ghost.TButton", width=BTN_W
        ).pack(side="left")
    ttk.Button(foot, text="OK", style="Primary.TButton", width=BTN_W,
               command=_on_close
        ).pack(side="right")
