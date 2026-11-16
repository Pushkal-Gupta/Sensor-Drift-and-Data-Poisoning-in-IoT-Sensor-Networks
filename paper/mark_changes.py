"""Produce the 'revised manuscript with changes highlighted' file.

    python paper/mark_changes.py OLD.docx NEW.docx OUT.docx

Every paragraph of NEW whose text does not appear in OLD (or matches its closest
OLD paragraph with similarity below THRESH) is highlighted in turquoise, as is
every table cell whose text is not found in any OLD table.  Paragraph-level
marking, the convention most journals accept for a regenerated manuscript.
"""

from __future__ import annotations

import difflib
import re
import sys

from docx import Document
from docx.enum.text import WD_COLOR_INDEX

THRESH = 0.93


def norm(t: str) -> str:
    return re.sub(r"\s+", " ", t.replace("‑", "-")).strip()


def main(old_p, new_p, out_p):
    old, new = Document(old_p), Document(new_p)
    old_pars = [norm(p.text) for p in old.paragraphs if norm(p.text)]
    old_set = set(old_pars)
    old_cells = {norm(c.text) for t in old.tables for r in t.rows for c in r.cells}
    n_mark = n_tot = 0
    for p in new.paragraphs:
        t = norm(p.text)
        if not t or not p.runs:
            continue
        n_tot += 1
        if t in old_set:
            continue
        best = max((difflib.SequenceMatcher(None, t, o, autojunk=False).quick_ratio(), o)
                   for o in old_pars)[1] if old_pars else ""
        if difflib.SequenceMatcher(None, t, best, autojunk=False).ratio() >= THRESH:
            continue
        for r in p.runs:
            if r.font.highlight_color is None:
                r.font.highlight_color = WD_COLOR_INDEX.TURQUOISE
        n_mark += 1
    c_mark = 0
    for tb in new.tables:
        for row in tb.rows:
            for c in row.cells:
                if norm(c.text) and norm(c.text) not in old_cells:
                    for p in c.paragraphs:
                        for r in p.runs:
                            r.font.highlight_color = WD_COLOR_INDEX.TURQUOISE
                    c_mark += 1
    new.save(out_p)
    print(f"marked {n_mark}/{n_tot} paragraphs and {c_mark} table cells -> {out_p}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
