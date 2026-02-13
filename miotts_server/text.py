from __future__ import annotations

import re

REPLACE_MAP: dict[str, str] = {
    r"\t": "",
    r"\[n\]": "",
    r" ": "",
    r"　": "",
    r"[;▼♀♂《》≪≫①②③④⑤⑥]": "",
    r"[\u02d7\u2010-\u2015\u2043\u2212\u23af\u23e4\u2500\u2501\u2e3a\u2e3b]": "",
    r"[\uff5e\u301C]": "ー",
    r"？": "?",
    r"！": "!",
    r"[●◯〇]": "○",
    r"♥": "♡",
}

FULLWIDTH_ALPHA_TO_HALFWIDTH = str.maketrans(
    {
        chr(full): chr(half)
        for full, half in zip(
            list(range(0xFF21, 0xFF3B)) + list(range(0xFF41, 0xFF5B)),
            list(range(0x41, 0x5B)) + list(range(0x61, 0x7B)),
            strict=True,
        )
    }
)
_HALFWIDTH_KATAKANA_CHARS = "ｦｧｨｩｪｫｬｭｮｯｰｱｲｳｴｵｶｷｸｹｺｻｼｽｾｿﾀﾁﾂﾃﾄﾅﾆﾇﾈﾉﾊﾋﾌﾍﾎﾏﾐﾑﾒﾓﾔﾕﾖﾗﾘﾙﾚﾛﾜﾝ"
_FULLWIDTH_KATAKANA_CHARS = "ヲァィゥェォャュョッーアイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワン"
HALFWIDTH_KATAKANA_TO_FULLWIDTH = str.maketrans(
    _HALFWIDTH_KATAKANA_CHARS, _FULLWIDTH_KATAKANA_CHARS
)
FULLWIDTH_DIGITS_TO_HALFWIDTH = str.maketrans(
    {
        chr(full): chr(half)
        for full, half in zip(range(0xFF10, 0xFF1A), range(0x30, 0x3A), strict=True)
    }
)


def normalize_text(text: str) -> str:
    """Normalize text for TTS."""
    for pattern, replacement in REPLACE_MAP.items():
        text = re.sub(pattern, replacement, text)

    text = text.translate(FULLWIDTH_ALPHA_TO_HALFWIDTH)
    text = text.translate(FULLWIDTH_DIGITS_TO_HALFWIDTH)
    text = text.translate(HALFWIDTH_KATAKANA_TO_FULLWIDTH)

    text = re.sub(r"…{3,}", "……", text)

    if text.startswith("「") and text.endswith("」"):
        text = text[1:-1]
    if text.startswith("『") and text.endswith("』"):
        text = text[1:-1]
    if text.startswith("（") and text.endswith("）"):
        text = text[1:-1]
    if text.startswith("【") and text.endswith("】"):
        text = text[1:-1]
    if text.startswith("(") and text.endswith(")"):
        text = text[1:-1]

    if text.endswith(("。", "、")):
        text = text.rstrip("。、")

    return text
