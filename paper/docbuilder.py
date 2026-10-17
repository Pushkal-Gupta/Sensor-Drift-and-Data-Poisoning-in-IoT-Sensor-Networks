"""Minimal manuscript builder on python-docx.

Text may contain citation markers {cite:key1,key2}.  They are resolved in
document order to Elsevier numbered style ([1], [2,3], [4-6]) and the reference
list is emitted in that order, so numbering always matches first appearance.

Layout follows the authors' original draft: paper/template.docx is that draft
with its body removed, so page size, margins, styles, list numbering and the
page-number footer come from it, and the block formats below copy its
explicit paragraph formatting (sizes in pt, spacing in pt).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from docx import Document
from docx.enum.section import WD_ORIENT  # noqa: F401
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor  # noqa: F401
from docx.enum.text import WD_TAB_ALIGNMENT

TEMPLATE = Path(__file__).resolve().parent / "template.docx"
TNR = "Times New Roman"

CITE = re.compile(r"\{cite:([^}]+)\}")
FMT = re.compile(r"(\*\*[^*]+\*\*|\*[^*]+\*|_\{[^}]+\}|\^\{[^}]+\})")


class Manuscript:
    def __init__(self, refs: Dict[str, str], template: Optional[Path] = TEMPLATE):
        self.doc = Document(str(template)) if template else Document()
        self.refs = refs
        self.order: List[str] = []
        self.n_fig = 0
        self.n_tab = 0
        self.n_eq = 0
        self._style()

    # ------------------------------------------------------------------ style
    def _style(self):
        d = self.doc
        for name in ("Normal", "Heading 1", "Heading 2", "Heading 3"):
            st = d.styles[name]
            st.font.name = TNR
            rf = st.element.get_or_add_rPr().get_or_add_rFonts()
            for att in ("w:asciiTheme", "w:hAnsiTheme", "w:eastAsiaTheme", "w:cstheme"):
                rf.attrib.pop(qn(att), None)
            rf.set(qn("w:eastAsia"), TNR)

    @staticmethod
    def _fmt(p, before=None, after=None, single=False, align=None, keep_next=False):
        pf = p.paragraph_format
        if before is not None:
            pf.space_before = Pt(before)
        if after is not None:
            pf.space_after = Pt(after)
        if single:
            pf.line_spacing = 1.0
        if align is not None:
            p.alignment = align
        if keep_next:
            pf.keep_with_next = True
        return p

    # -------------------------------------------------------------- citations
    def _resolve(self, text: str) -> str:
        def rep(m):
            keys = [k.strip() for k in m.group(1).split(",")]
            nums = []
            for k in keys:
                if k not in self.refs:
                    raise KeyError(f"unknown reference key {k!r}")
                if k not in self.order:
                    self.order.append(k)
                nums.append(self.order.index(k) + 1)
            return "[" + _compress(sorted(set(nums))) + "]"
        return CITE.sub(rep, text)

    # ------------------------------------------------------------------- runs
    def _runs(self, p, text: str, size: Optional[float] = None, italic=False):
        text = self._resolve(text)
        for part in FMT.split(text):
            if not part:
                continue
            if part.startswith("**"):
                r = p.add_run(part[2:-2]); r.bold = True
            elif part.startswith("*") and part.endswith("*") and len(part) > 2:
                r = p.add_run(part[1:-1]); r.italic = True
            elif part.startswith("_{"):
                r = p.add_run(part[2:-1]); r.font.subscript = True
            elif part.startswith("^{"):
                r = p.add_run(part[2:-1]); r.font.superscript = True
            else:
                r = p.add_run(part)
            if italic:
                r.italic = True
            if size:
                r.font.size = Pt(size)
        return p

    # ----------------------------------------------------------------- blocks
    def title(self, text):
        p = self._fmt(self.doc.add_paragraph(), after=8, align=WD_ALIGN_PARAGRAPH.CENTER)
        r = p.add_run(text); r.bold = True; r.font.size = Pt(16)

    def centered(self, text, size=11, italic=False, after=2):
        p = self._fmt(self.doc.add_paragraph(), after=after, align=WD_ALIGN_PARAGRAPH.CENTER)
        self._runs(p, text, size, italic)

    def h1(self, t): self.doc.add_heading(t, level=1)
    def h2(self, t): self.doc.add_heading(t, level=2)
    def h3(self, t): self.doc.add_heading(t, level=3)

    def p(self, text, lead: Optional[str] = None, size: Optional[float] = None,
          before: Optional[float] = None):
        para = self._fmt(self.doc.add_paragraph(), before=before,
                         align=WD_ALIGN_PARAGRAPH.JUSTIFY)
        if lead:
            r = para.add_run(lead + " "); r.bold = True
            if size:
                r.font.size = Pt(size)
        self._runs(para, text, size)
        return para

    def bullets(self, items: Sequence[str], numbered=False):
        for it in items:
            para = self.doc.add_paragraph(style="List Number" if numbered else "List Bullet")
            self._fmt(para, after=3, align=WD_ALIGN_PARAGRAPH.JUSTIFY)
            self._runs(para, it)

    def eq(self, text: str, label: Optional[str] = None) -> int:
        self.n_eq += 1
        p = self._fmt(self.doc.add_paragraph(), before=4, after=4, single=True,
                      align=WD_ALIGN_PARAGRAPH.LEFT)
        p.paragraph_format.keep_together = True
        ts = p.paragraph_format.tab_stops
        ts.add_tab_stop(Cm(8.0), WD_TAB_ALIGNMENT.CENTER)
        ts.add_tab_stop(Cm(15.9), WD_TAB_ALIGNMENT.RIGHT)
        p.add_run("\t")
        self._runs(p, text)
        p.add_run(f"\t({self.n_eq})")
        return self.n_eq

    def figure(self, path: Path, caption: str, width_cm: float = 15.5) -> int:
        self.n_fig += 1
        p = self._fmt(self.doc.add_paragraph(), before=10, after=3, single=True,
                      align=WD_ALIGN_PARAGRAPH.CENTER, keep_next=True)
        p.add_run().add_picture(str(path), width=Cm(min(width_cm, 15.9)))
        c = self._fmt(self.doc.add_paragraph(), before=3, after=13, single=True,
                      align=WD_ALIGN_PARAGRAPH.JUSTIFY)
        r = c.add_run(f"Fig. {self.n_fig}."); r.bold = True; r.font.size = Pt(9.5)
        self._runs(c, " " + caption, size=9.5)
        return self.n_fig

    def table(self, caption: str, header: Sequence[str], rows: Sequence[Sequence[str]],
              note: Optional[str] = None, widths: Optional[Sequence[float]] = None,
              font: float = 9) -> int:
        self.n_tab += 1
        c = self._fmt(self.doc.add_paragraph(), before=6, after=5, single=True,
                      align=WD_ALIGN_PARAGRAPH.CENTER, keep_next=True)
        r = c.add_run(f"Table {self.n_tab}."); r.bold = True; r.font.size = Pt(9.5)
        self._runs(c, " " + caption, size=9.5)
        ncol = len(header)
        t = self.doc.add_table(rows=1 + len(rows), cols=ncol)
        t.style = "Table Grid"
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        t.autofit = False
        total = 15.9
        if widths:
            k = total / sum(widths)
            widths = [w * k for w in widths]
        else:
            widths = [total / ncol] * ncol
        _table_props(t)
        for i, row in enumerate([list(header)] + [list(r) for r in rows]):
            tr = t.rows[i]
            _row_props(tr, header=i == 0)
            for j, v in enumerate(row):
                cell = tr.cells[j]
                cell.width = Cm(widths[j])
                cell.vertical_alignment = 1  # centre
                para = cell.paragraphs[0]
                self._fmt(para, before=2, after=2, single=True,
                          align=WD_ALIGN_PARAGRAPH.LEFT if j == 0 else WD_ALIGN_PARAGRAPH.CENTER)
                if i == 0:
                    rr = para.add_run(self._resolve(str(v))); rr.bold = True
                    rr.font.size = Pt(font)
                    _shade(cell, "DCE6F1")
                else:
                    self._runs(para, str(v), size=font)
        n = self._fmt(self.doc.add_paragraph(), after=6, single=True)
        if note:
            self._runs(n, note, size=9)
        return self.n_tab

    def page_break(self):
        self.doc.add_page_break()

    def references(self):
        self.h1("References")
        for i, k in enumerate(self.order, start=1):
            para = self._fmt(self.doc.add_paragraph(), after=4, align=WD_ALIGN_PARAGRAPH.LEFT)
            para.paragraph_format.left_indent = Cm(1.016)
            para.paragraph_format.first_line_indent = Cm(-1.016)
            r = para.add_run(f"[{i}]\t"); r.font.size = Pt(9.5)
            r2 = para.add_run(self.refs[k]); r2.font.size = Pt(9.5)
        unused = [k for k in self.refs if k not in self.order]
        return unused

    def save(self, path: Path):
        self.doc.save(str(path))


def _compress(nums: List[int]) -> str:
    out, i = [], 0
    while i < len(nums):
        j = i
        while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
            j += 1
        out.append(f"{nums[i]}–{nums[j]}" if j - i >= 2 else
                   ",".join(str(n) for n in nums[i:j + 1]))
        i = j + 1
    return ",".join(out)


def _shade(cell, hex_fill: str):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear"); shd.set(qn("w:color"), "auto"); shd.set(qn("w:fill"), hex_fill)
    tcPr.append(shd)


def _table_props(t):
    tblPr = t._tbl.tblPr
    lay = OxmlElement("w:tblLayout"); lay.set(qn("w:type"), "fixed"); tblPr.append(lay)
    mar = OxmlElement("w:tblCellMar")
    for side in ("left", "right"):
        e = OxmlElement(f"w:{side}"); e.set(qn("w:w"), "85"); e.set(qn("w:type"), "dxa")
        mar.append(e)
    tblPr.append(mar)


def _row_props(tr, header: bool):
    trPr = tr._tr.get_or_add_trPr()
    trPr.append(OxmlElement("w:cantSplit"))
    if header:
        trPr.append(OxmlElement("w:tblHeader"))
