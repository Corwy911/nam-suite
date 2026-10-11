"""NAM Trainer — minimalist redesign mockup.

Run with:
    python -m nam_trainer
or:
    python nam_trainer/main.py

A working UI mockup, fully interactive: switch themes with the swatch
buttons at the top, open Advanced Options, type into fields, etc.
"""
import tkinter as tk
from tkinter import ttk

from .themes import THEMES, DEFAULT
from .style  import (
    apply_style, FT_DISPLAY, FT_EYEBROW, FT_BODY, FT_LABEL,
    FT_MONO, FT_MONO_S, FT_BTN,
)
from .widgets import (
    eyebrow, divider, vrule, status_dot,
    make_metrics_widget, render_metrics,
)
from .advanced import open_advanced


# Demo state for the live metrics line — replace with real training state.
DEMO_STATE = dict(
    phase="stage 2 lows",
    ep=40, total=100, pct=40.0,
    cur_loss="0.01096", cur_esr="0.01096", cur_mse="5.121e-05",
    bst_loss="0.010801", bst_esr="0.010801",
    bst_ep=37, bst_back=3,
    eta="00:08:09", around="14:15:35",
)


def run():
    # Module-scoped state held inside `app` so the theme-switch closure works.
    app = {"theme": DEFAULT, "P": dict(THEMES[DEFAULT])}

    root = tk.Tk()
    root.title("NAM Trainer · v0.2")
    root.geometry("1060x880")
    root.minsize(940, 780)

    def set_theme(name):
        app["theme"] = name
        app["P"].clear()
        app["P"].update(THEMES[name])
        root.configure(bg=app["P"]["BG"])
        apply_style(root, app["P"])
        build_ui()
        # Park focus on root so no widget (esp. the readonly Combobox)
        # retains a selection highlight from the previous theme.
        root.focus_set()
        root.update_idletasks()

    def build_ui():
        P = app["P"]
        for w in root.winfo_children():
            w.destroy()

        outer = ttk.Frame(root, style="TFrame", padding=(28, 16, 28, 22))
        outer.pack(fill="both", expand=True)

        # ---------- Header (title left, status + theme dropdown right) ----------
        hdr = tk.Frame(outer, bg=P["BG"], highlightthickness=0)
        hdr.pack(fill="x", pady=(0, 18))

        left = tk.Frame(hdr, bg=P["BG"]); left.pack(side="left")
        ttk.Label(left, text="NAM Trainer", style="Display.TLabel"
            ).pack(side="left")
        vrule(left, P, height=18).pack(side="left", padx=12)
        eyebrow(left, "Neural amp modeler", P).pack(side="left")

        right = tk.Frame(hdr, bg=P["BG"]); right.pack(side="right")

        # Status: "● Running · stage 2"
        status_dot(right, P, "M_BEST").pack(side="left", padx=(0, 6))
        eyebrow(right, "Running · stage 2", P).pack(side="left", padx=(0, 14))

        # Separator
        vrule(right, P, height=18).pack(side="left", padx=(0, 14))

        # Theme switcher
        eyebrow(right, "Theme", P).pack(side="left", padx=(0, 10))

        # Color chip — Canvas is the only widget that guarantees its
        # declared size even with no children, so it survives theme
        # rebuilds reliably (a tk.Frame would collapse to 0×0).
        chip_wrap = tk.Frame(right, bg=P["BORDER"], bd=0, highlightthickness=0)
        chip_wrap.pack(side="left", padx=(0, 8))
        chip = tk.Canvas(chip_wrap, width=12, height=12,
                         bg=P["ACCENT"], highlightthickness=0, bd=0)
        chip.pack(padx=1, pady=1)

        theme_var = tk.StringVar(value=app["theme"])
        theme_dd = ttk.Combobox(
            right, textvariable=theme_var,
            values=list(THEMES.keys()), state="readonly",
            width=12, font=FT_BTN, style="TCombobox")
        theme_dd.pack(side="left")

        def _on_theme_change(event):
            new_name = theme_var.get()
            # Drop selection highlight + blur the dropdown before
            # rebuilding, so the readonly text doesn't stay highlighted
            # in the previous theme's selection color.
            event.widget.selection_clear()
            root.focus_set()
            set_theme(new_name)

        theme_dd.bind("<<ComboboxSelected>>", _on_theme_change)

        # ---------- Files ----------
        eyebrow(outer, "Files", P).pack(anchor="w", pady=(0, 6))
        files = tk.Frame(outer, bg=P["BG"], highlightthickness=0)
        files.pack(fill="x", pady=(0, 4))
        files.columnconfigure(0, minsize=170)
        files.columnconfigure(1, weight=1)
        files.columnconfigure(2, minsize=200)

        def files_row(r, label, path, action,
                      primary=False, path_is_hint=False,
                      action_cmd=None):
            ttk.Button(files, text=label, style="Ink.TButton"
                ).grid(row=r, column=0, sticky="ew",
                       padx=(0, 18), pady=4)

            inner = tk.Frame(files, bg=P["BG"])
            inner.grid(row=r, column=1, sticky="w")
            ttk.Label(inner, text="›", background=P["BG"],
                      foreground=P["FAINT"], font=FT_LABEL
                ).pack(side="left", padx=(0, 8))
            ttk.Label(inner, text=path, background=P["BG"],
                      foreground=P["FAINT"] if path_is_hint else P["MUTED"],
                      font=FT_MONO_S
                ).pack(side="left")

            ttk.Button(files,
                       text=action,
                       style="Primary.TButton" if primary else "Ghost.TButton",
                       command=action_cmd or (lambda: None)
                ).grid(row=r, column=2, sticky="ew",
                       padx=(18, 0), pady=4)

        files_row(0, "Input Audio",
                  "G:/NAM MXR X100/TTS input v10.wav",
                  "Download input file")
        files_row(1, "Output Audio",
                  "G:/NAM BRAY 4550 DLX/CLIPS/SLAMMIN_RAY_455DLX_VI8…",
                  "Analyze Lightning Folders")
        files_row(2, "Train Destination",
                  "G:/NAM BRAY 4550 DLX/RECEPTIVE FIELD MODELS",
                  "Advanced options…",
                  action_cmd=lambda: open_advanced(root, app["P"]))
        files_row(3, "Metadata",
                  "Optional model and gear metadata",
                  "TRAIN",
                  primary=True, path_is_hint=True)

        # ---------- Options strip ----------
        divider(outer, P).pack(fill="x", pady=(14, 0))
        opts = tk.Frame(outer, bg=P["BG"])
        opts.pack(fill="x", pady=(10, 10))
        eyebrow(opts, "Options", P).pack(side="left", padx=(0, 22))
        for label, default in (
            ("Silent run  (suggested for batch training)", True),
            ("Save ESR plot automatically",                True),
            ("Show advanced initialization info",          False),
        ):
            v = tk.BooleanVar(value=default)
            ttk.Checkbutton(opts, text=label, variable=v,
                            style="Flat.TCheckbutton"
                ).pack(side="left", padx=(0, 22))
        divider(outer, P).pack(fill="x", pady=(0, 16))

        # ---------- Scheduler ----------
        sch_hdr = tk.Frame(outer, bg=P["BG"])
        sch_hdr.pack(fill="x", pady=(0, 8))
        eyebrow(sch_hdr, "Scheduler", P).pack(side="left", padx=(0, 14))
        ttk.Label(sch_hdr, text="●  ", background=P["BG"],
                  foreground=P["M_BEST"], font=FT_MONO_S
            ).pack(side="left")
        ttk.Label(sch_hdr, text="Running: ", background=P["BG"],
                  foreground=P["MUTED"], font=FT_MONO_S
            ).pack(side="left")
        ttk.Label(sch_hdr,
                  text="ComplexRF300Lite  |  600e  |  b32 ny8192  |  "
                       "lr0.005  |  SLAMMIN_RAY_455DLX… +6 more",
                  background=P["BG"], foreground=P["TEXT"], font=FT_MONO_S
            ).pack(side="left")
        ttk.Label(sch_hdr, text="Queued  0", background=P["BG"],
                  foreground=P["MUTED"], font=FT_MONO_S
            ).pack(side="right")

        # Queue listbox (use tk, not ttk, for full color control)
        qframe = tk.Frame(outer, bg=P["BORDER"])
        qframe.pack(fill="both", expand=True)
        queue_inner = tk.Frame(qframe, bg=P["SURFACE"])
        queue_inner.pack(fill="both", expand=True, padx=1, pady=1)
        queue = tk.Listbox(queue_inner,
            bg=P["SURFACE"], fg=P["TEXT"],
            selectbackground=P["ACCENT"], selectforeground=P["ACC_TEXT"],
            relief="flat", highlightthickness=0, borderwidth=0,
            font=FT_MONO, activestyle="none", height=5)
        queue.pack(fill="both", expand=True, padx=10, pady=8)
        queue.insert("end", "— queue empty —")
        queue.itemconfig(0, foreground=P["FAINT"])

        # 2x2 button grid
        btns = tk.Frame(outer, bg=P["BG"])
        btns.pack(fill="x", pady=(10, 0))
        btns.columnconfigure(0, weight=1)
        btns.columnconfigure(1, weight=1)
        ttk.Button(btns, text="Add to Schedule", style="Subtle.TButton"
            ).grid(row=0, column=0, sticky="ew", padx=(0, 5), pady=(0, 5))
        ttk.Button(btns, text="Start Schedule", style="Primary.TButton"
            ).grid(row=0, column=1, sticky="ew", padx=(5, 0), pady=(0, 5))
        ttk.Button(btns, text="Remove Selected", style="Subtle.TButton"
            ).grid(row=1, column=0, sticky="ew", padx=(0, 5), pady=(5, 0))
        ttk.Button(btns, text="Clear Schedule", style="Ghost.TButton"
            ).grid(row=1, column=1, sticky="ew", padx=(5, 0), pady=(5, 0))

        # Progress bar
        pbar = ttk.Progressbar(outer,
            style="Flat.Horizontal.TProgressbar",
            maximum=100, value=40)
        pbar.pack(fill="x", pady=(14, 8))

        # Metrics line
        metrics = make_metrics_widget(outer, P)
        metrics.pack(fill="x")
        render_metrics(metrics, P, DEMO_STATE)

    # Boot
    apply_style(root, app["P"])
    root.configure(bg=app["P"]["BG"])
    build_ui()
    root.mainloop()


if __name__ == "__main__":
    run()
