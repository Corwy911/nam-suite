"""Small composable widgets used by both main.py and advanced.py."""
import tkinter as tk
from tkinter import ttk

from .style import (
    FT_EYEBROW, FT_BODY, FT_LABEL, FT_MONO, FT_MONO_S, FT_METRIC,
)


# ---------------------------------------------------------------------------
# Typographic atoms
# ---------------------------------------------------------------------------

def eyebrow(parent, text, P, prefix="\u25c6"):
    """Uppercase, letter-tracked section header. Tk has no letter-spacing
    property, so we approximate by inserting double-spaces between glyphs."""
    spaced = "  ".join(list(text.upper()))
    if prefix:
        frame = tk.Frame(parent, bg=P["BG"], bd=0, highlightthickness=0)
        tk.Label(
            frame,
            text=prefix,
            bg=P["BG"],
            fg=P["FAINT"],
            font=("Segoe UI", 6),
            padx=0,
            pady=0,
        ).pack(side="left", anchor="center", padx=(0, 8))
        ttk.Label(
            frame,
            text=spaced,
            background=P["BG"],
            foreground=P["FAINT"],
            font=FT_EYEBROW,
            style="Faint.TLabel",
        ).pack(side="left", anchor="center")
        return frame
    return ttk.Label(
        parent,
        text=spaced,
        background=P["BG"],
        foreground=P["FAINT"],
        font=FT_EYEBROW,
    )


def divider(parent, P):
    return tk.Frame(parent, height=1, bg=P["DIVIDER"], bd=0, highlightthickness=0)


def vrule(parent, P, height=16):
    return tk.Frame(parent, width=1, height=height, bg=P["BORDER"], bd=0,
                    highlightthickness=0)


def status_dot(parent, P, color_key="M_BEST"):
    dot = tk.Canvas(
        parent,
        width=8,
        height=14,
        bd=0,
        highlightthickness=0,
        bg=P["BG"],
    )
    dot.create_oval(2, 5, 5, 8, fill=P[color_key], outline=P[color_key], tags=("dot",))
    return dot


# ---------------------------------------------------------------------------
# Theme switcher button row
# ---------------------------------------------------------------------------

def swatch_button(parent, theme_name, theme_dict, P, is_active, on_click):
    """A flat button that shows the theme's own accent color as a chip,
    next to its name. The active theme button has solid accent background."""
    if is_active:
        bg, fg = P["ACCENT"], P["ACC_TEXT"]
        chip_bg = P["ACC_TEXT"]
    else:
        bg, fg = P["SURFACE_ALT"], P["TEXT"]
        chip_bg = theme_dict["ACCENT"]

    box = tk.Frame(parent, bg=bg, bd=0, highlightthickness=0, cursor="hand2")
    chip = tk.Frame(box, bg=chip_bg, width=8, height=8,
                    bd=0, highlightthickness=0)
    chip.pack(side="left", padx=(10, 8), pady=8)
    chip.pack_propagate(False)
    lbl = tk.Label(box, text=theme_name, bg=bg, fg=fg,
                   font=("Segoe UI Semibold", 9), padx=0)
    lbl.pack(side="left", padx=(0, 12))

    for w in (box, chip, lbl):
        w.bind("<Button-1>", lambda e: on_click())
    return box


# ---------------------------------------------------------------------------
# Colored live-metrics line (tk.Text with named tags)
# ---------------------------------------------------------------------------

def make_metrics_widget(parent, P):
    t = tk.Text(parent, height=2, bg=P["BG"], fg=P["MUTED"],
                bd=0, relief="flat", font=FT_MONO_S, wrap="word",
                highlightthickness=0, cursor="arrow",
                padx=0, pady=0)
    return t


def render_metrics(widget, P, state):
    """Re-render the metrics line. Pass a dict like:
        dict(ep=40, total=100, pct=40.0,
             cur_loss=..., cur_esr=..., cur_mse=...,
             bst_loss=..., bst_esr=...,
             bst_ep=37, bst_back=3,
             eta="00:08:09", around="14:15:35", phase="stage 2 lows")
    """
    widget.config(state="normal", bg=P["BG"], fg=P["MUTED"])
    widget.delete("1.0", "end")

    # (Re)configure tags every render — cheap and survives theme swaps.
    widget.tag_configure("strong",  foreground=P["TEXT"])
    widget.tag_configure("faint",   foreground=P["FAINT"])
    widget.tag_configure("current", foreground=P["M_CURRENT"], font=FT_METRIC)
    widget.tag_configure("best",    foreground=P["M_BEST"],    font=FT_METRIC)
    widget.tag_configure("loss",    foreground=P["M_LOSS"],    font=FT_METRIC)
    widget.tag_configure("esr",     foreground=P["M_ESR"],     font=FT_METRIC)
    widget.tag_configure("mse",     foreground=P["M_MSE"])

    def w(text, *tags):
        widget.insert("end", text, tags)

    w(state.get("phase", "stage 2 lows"), "strong")
    w("  ·  ", "faint")
    w(f"epoch {state['ep']}/{state['total']}", "strong")
    w("  ·  ", "faint")
    w(f"{state['pct']:.1f}%", "strong")

    w("    |    ", "faint")
    w("current ", "current")
    w("loss "); w(state["cur_loss"], "strong")
    w("  ·  ", "faint"); w("ESR ");  w(state["cur_esr"], "strong")
    w("  ·  ", "faint"); w("MSE ");  w(state["cur_mse"], "mse")

    w("    |    ", "faint")
    w("best ", "best")
    w("loss "); w(state["bst_loss"], "strong")
    w("  ·  ", "faint"); w("ESR "); w(state["bst_esr"], "strong")

    w("\n")
    w(f"at epoch {state['bst_ep']}", "esr")
    w(f"  ({state['bst_back']} epochs back)", "faint")
    w("     |     ", "faint")
    w(f"ETA {state['eta']}", "loss")
    w(f"  ·  around {state['around']}", "faint")

    widget.config(state="disabled")
