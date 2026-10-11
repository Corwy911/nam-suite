"""Color palettes for the NAM Trainer UI.

Each theme is a dict with the same keys, so swapping themes is just:
    P.clear(); P.update(THEMES[name])
followed by re-running apply_style(root, P) and rebuilding the UI.
"""

THEMES = {
    "Paper": dict(
        BG="#F1ECE2", SURFACE="#FBF8F2", SURFACE_ALT="#F6F1E5",
        BORDER="#E2DACB", DIVIDER="#EAE3D5",
        TEXT="#1F1B16", MUTED="#7A7368", FAINT="#A89F90",
        ACCENT="#B85C36", ACCENT_HOVER="#9E4D2C", ACC_TEXT="#FBF8F2",
        M_CURRENT="#2E6FA8", M_BEST="#2F7A3C",
        M_LOSS="#B85C36", M_ESR="#A85C7A", M_MSE="#C68A1E",
        PROG_TRACK="#E2DACB", PROG_FILL="#B85C36",
    ),
    "Graphite": dict(
        BG="#FFFFFF", SURFACE="#FFFFFF", SURFACE_ALT="#F5F5F5",
        BORDER="#D8D8D8", DIVIDER="#EFEFEF",
        TEXT="#0F0F0F", MUTED="#5C5C5C", FAINT="#9A9A9A",
        ACCENT="#1F3FA8", ACCENT_HOVER="#162F82", ACC_TEXT="#FFFFFF",
        M_CURRENT="#1F3FA8", M_BEST="#1E6B3C",
        M_LOSS="#B4441A", M_ESR="#8E2E63", M_MSE="#A77410",
        PROG_TRACK="#EFEFEF", PROG_FILL="#1F3FA8",
    ),
    "Slate": dict(
        BG="#15171C", SURFACE="#1B1E25", SURFACE_ALT="#20242C",
        BORDER="#2A2E37", DIVIDER="#252932",
        TEXT="#E7E9ED", MUTED="#8B919E", FAINT="#565B65",
        ACCENT="#7FB3D5", ACCENT_HOVER="#A2C8E0", ACC_TEXT="#0F1115",
        M_CURRENT="#7FB3D5", M_BEST="#7AB893",
        M_LOSS="#E6A06A", M_ESR="#D08AA8", M_MSE="#D8C079",
        PROG_TRACK="#2A2E37", PROG_FILL="#7FB3D5",
    ),
    "Cyan": dict(
        BG="#EEF6F8", SURFACE="#FFFFFF", SURFACE_ALT="#E1EEF1",
        BORDER="#C8DDE2", DIVIDER="#DEEAEE",
        TEXT="#062731", MUTED="#456E78", FAINT="#88A8B0",
        ACCENT="#0AA5B8", ACCENT_HOVER="#088594", ACC_TEXT="#FFFFFF",
        M_CURRENT="#0E7C8E", M_BEST="#1F8A5B",
        M_LOSS="#C24E2F", M_ESR="#8E3B7E", M_MSE="#B5821B",
        PROG_TRACK="#C8DDE2", PROG_FILL="#0AA5B8",
    ),
    "Pink": dict(
        BG="#FAF1F4", SURFACE="#FFFFFF", SURFACE_ALT="#F5E5EA",
        BORDER="#E8CDD6", DIVIDER="#F0DAE0",
        TEXT="#2D1722", MUTED="#7E5C68", FAINT="#B89AA3",
        ACCENT="#C2185B", ACCENT_HOVER="#9D124A", ACC_TEXT="#FFFFFF",
        M_CURRENT="#7A3E97", M_BEST="#2E7A4B",
        M_LOSS="#C2185B", M_ESR="#A8377A", M_MSE="#B47718",
        PROG_TRACK="#E8CDD6", PROG_FILL="#C2185B",
    ),
    "Amber": dict(
        BG="#1A140D", SURFACE="#231C13", SURFACE_ALT="#2A2218",
        BORDER="#3A2F22", DIVIDER="#2E251A",
        TEXT="#F2E2C4", MUTED="#9A8970", FAINT="#6A5C48",
        ACCENT="#E5A140", ACCENT_HOVER="#F1B968", ACC_TEXT="#1A140D",
        M_CURRENT="#E5A140", M_BEST="#8FBF6A",
        M_LOSS="#E58A65", M_ESR="#D88FB5", M_MSE="#E5C77A",
        PROG_TRACK="#3A2F22", PROG_FILL="#E5A140",
    ),
    "Cyan Dark": dict(
        BG="#0E1A1E", SURFACE="#152428", SURFACE_ALT="#1B2D32",
        BORDER="#26393F", DIVIDER="#1F3338",
        TEXT="#D8ECF0", MUTED="#7BA2AC", FAINT="#4E6C73",
        ACCENT="#22D3DC", ACCENT_HOVER="#4DE0E8", ACC_TEXT="#08161A",
        M_CURRENT="#22D3DC", M_BEST="#7DD89A",
        M_LOSS="#E58A65", M_ESR="#E091B8", M_MSE="#E5C77A",
        PROG_TRACK="#26393F", PROG_FILL="#22D3DC",
    ),
    "Pink Dark": dict(
        BG="#1A1014", SURFACE="#22161B", SURFACE_ALT="#2A1B22",
        BORDER="#3D2A33", DIVIDER="#2E2028",
        TEXT="#F0DCE4", MUTED="#A38898", FAINT="#6B5260",
        ACCENT="#EC5A91", ACCENT_HOVER="#F37AA8", ACC_TEXT="#1A0911",
        M_CURRENT="#9D8FE5", M_BEST="#7DD89A",
        M_LOSS="#EC5A91", M_ESR="#E091B8", M_MSE="#E5C77A",
        PROG_TRACK="#3D2A33", PROG_FILL="#EC5A91",
    ),
}

DEFAULT = "Slate"
