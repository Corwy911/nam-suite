"""Centralised ttk.Style configuration.

Call apply_style(root, P) once at startup and whenever the theme changes.
All button, entry, combo, checkbutton and progressbar styles are derived
from the palette dict P.
"""
import tkinter as tk
from tkinter import ttk

# ----- Fonts (Windows-first; falls back gracefully on macOS/Linux) -----
FT_DISPLAY  = ("Segoe UI Light", 16)
FT_EYEBROW  = ("Segoe UI", 8)
FT_BODY     = ("Segoe UI", 10)
FT_LABEL    = ("Segoe UI", 9)
FT_MONO     = ("Consolas", 10)
FT_MONO_S   = ("Consolas", 9)
FT_BTN      = ("Segoe UI Semibold", 9)
FT_BTN_BIG  = ("Segoe UI Semibold", 10)
FT_METRIC   = ("Segoe UI Semibold", 9)


def apply_style(root, P):
    """Configure every ttk style from the palette dict P."""
    s = ttk.Style(root)
    try:
        s.theme_use("clam")  # only clam allows full recolouring
    except tk.TclError:
        pass

    # ---- Base ----
    s.configure(".", background=P["BG"], foreground=P["TEXT"], font=FT_BODY)

    s.configure("TFrame",      background=P["BG"])
    s.configure("Card.TFrame", background=P["SURFACE"])

    s.configure("TLabel",        background=P["BG"], foreground=P["TEXT"], font=FT_BODY)
    s.configure("Muted.TLabel",  background=P["BG"], foreground=P["MUTED"], font=FT_LABEL)
    s.configure("Faint.TLabel",  background=P["BG"], foreground=P["FAINT"], font=FT_EYEBROW)
    s.configure("Display.TLabel",background=P["BG"], foreground=P["TEXT"], font=FT_DISPLAY)
    s.configure("Mono.TLabel",   background=P["BG"], foreground=P["MUTED"], font=FT_MONO_S)

    # ---- Entry ----
    s.configure("TEntry",
        fieldbackground=P["SURFACE"], foreground=P["TEXT"],
        bordercolor=P["DIVIDER"], lightcolor=P["DIVIDER"], darkcolor=P["DIVIDER"],
        insertcolor=P["ACCENT"], borderwidth=1, relief="flat", padding=4)
    s.map("TEntry",
        bordercolor=[("focus", P["ACCENT"])],
        lightcolor=[("focus", P["ACCENT"])],
        darkcolor =[("focus", P["ACCENT"])])

    # ---- Combobox ----
    s.configure("TCombobox",
        fieldbackground=P["SURFACE"], background=P["SURFACE"],
        foreground=P["TEXT"], arrowcolor=P["MUTED"],
        bordercolor=P["DIVIDER"], lightcolor=P["DIVIDER"], darkcolor=P["DIVIDER"],
        selectbackground=P["SURFACE"], selectforeground=P["TEXT"],
        arrowsize=8, borderwidth=1, padding=4, relief="flat")
    s.map("TCombobox",
        fieldbackground=[("readonly", P["SURFACE"]),
                         ("readonly", "focus", P["SURFACE"])],
        foreground=[("readonly", P["TEXT"]),
                    ("readonly", "focus", P["TEXT"])],
        # Suppress the readonly text-selection highlight entirely.
        selectbackground=[("readonly", P["SURFACE"]),
                          ("readonly", "focus", P["SURFACE"]),
                          ("!focus", P["SURFACE"]),
                          ("focus", P["SURFACE"])],
        selectforeground=[("readonly", P["TEXT"]),
                          ("readonly", "focus", P["TEXT"]),
                          ("!focus", P["TEXT"]),
                          ("focus", P["TEXT"])],
        bordercolor=[("focus", P["ACCENT"])],
        lightcolor=[("focus", P["ACCENT"])],
        darkcolor =[("focus", P["ACCENT"])],
        arrowcolor=[("active", P["MUTED"]), ("readonly", P["MUTED"])])
    # Match the popup list:
    root.option_add("*TCombobox*Listbox.background",       P["SURFACE"])
    root.option_add("*TCombobox*Listbox.foreground",       P["TEXT"])
    root.option_add("*TCombobox*Listbox.selectBackground", P["ACCENT"])
    root.option_add("*TCombobox*Listbox.selectForeground", P["ACC_TEXT"])
    root.option_add("*TCombobox*Listbox.font",             FT_BODY)
    root.option_add("*TCombobox*Listbox.borderWidth",      "0")
    root.option_add("*TCombobox*Listbox.relief",           "flat")

    # ---- Buttons ----
    def _btn(name, bg, fg, border=None, pad=(14, 9), font=FT_BTN):
        b = border or bg
        s.configure(name,
            background=bg, foreground=fg,
            bordercolor=b, lightcolor=b, darkcolor=b,
            focusthickness=0, focuscolor=bg,
            font=font, padding=pad, relief="flat")

    # All button styles share the same vertical metric (pad_y = 9, FT_BTN 9pt)
    # so they line up perfectly when placed in the same row.
    _btn("Primary.TButton", P["ACCENT"],      P["ACC_TEXT"], pad=(14, 9))
    _btn("Ink.TButton",     P["TEXT"],        P["ACC_TEXT"], pad=(14, 9))
    _btn("Subtle.TButton",  P["SURFACE_ALT"], P["TEXT"], P["BORDER"], pad=(14, 9))

    # Keep Ghost visually identical in height to filled buttons in the file grid.
    s.configure("Ghost.TButton",
        background=P["BG"], foreground=P["TEXT"],
        bordercolor=P["TEXT"], lightcolor=P["TEXT"], darkcolor=P["TEXT"],
        focusthickness=0, focuscolor=P["BG"],
        font=FT_BTN, padding=(14, 9),
        relief="solid", borderwidth=1)

    s.map("Primary.TButton", background=[("active", P["ACCENT_HOVER"])],
                             bordercolor=[("active", P["ACCENT_HOVER"])])
    s.map("Ink.TButton",     background=[("active", P["MUTED"])],
                             bordercolor=[("active", P["MUTED"])])
    s.map("Subtle.TButton",  background=[("active", P["BORDER"])])
    s.map("Ghost.TButton",   background=[("active", P["SURFACE_ALT"])])

    # ---- Checkbutton ----
    s.configure("Flat.TCheckbutton",
        background=P["BG"], foreground=P["TEXT"],
        focuscolor=P["BG"], indicatorrelief="flat",
        indicatormargin=3, padding=4, font=FT_BODY)
    s.map("Flat.TCheckbutton",
        background=[("active", P["BG"])],
        foreground=[("active", P["TEXT"])],
        indicatorcolor=[("selected",  P["ACCENT"]),
                        ("!selected", P["SURFACE"])])

    # ---- Progressbar ----
    s.configure("Flat.Horizontal.TProgressbar",
        troughcolor=P["PROG_TRACK"], background=P["PROG_FILL"],
        bordercolor=P["PROG_TRACK"],
        lightcolor=P["PROG_FILL"], darkcolor=P["PROG_FILL"],
        thickness=6, borderwidth=0)

    # ---- Scrollbar ----
    s.configure("Flat.Vertical.TScrollbar",
        background=P["SURFACE_ALT"], troughcolor=P["BG"],
        bordercolor=P["BG"], lightcolor=P["SURFACE_ALT"],
        darkcolor=P["SURFACE_ALT"], arrowcolor=P["MUTED"],
        relief="flat")
