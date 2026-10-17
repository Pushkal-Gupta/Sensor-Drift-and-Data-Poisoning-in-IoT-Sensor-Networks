"""Build the journal manuscript (.docx) from prose + experiment artifacts.

    python paper/build_manuscript.py

Writes paper/PIRCAD_MLWA_manuscript.docx.  All numbers in Sections 6-7 and in
the abstract are read from artifacts/*.json via results_tables.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from docbuilder import Manuscript  # noqa: E402
from references import REFS  # noqa: E402
import sections_front as front  # noqa: E402
import sections_results as results  # noqa: E402


def main() -> None:
    R = results.collect()
    m = Manuscript(REFS)
    m.title(front.TITLE)
    m.centered(front.AUTHORS)
    for line in front.AFFIL:
        m.centered(line, 10, italic=True)
    m.centered(front.CORR, 9)

    m.h1("Highlights")
    m.bullets(results.highlights(R))
    m.h1("Abstract")
    m.p(results.abstract(R))
    m.p(front.KEYWORDS, lead="Keywords:")

    front.methods(m, R)
    results.results_section(m, R)
    results.discussion_and_conclusion(m, R)
    results.declarations(m)
    unused = m.references()
    out = HERE / (sys.argv[1] if len(sys.argv) > 1 else "PIRCAD_MLWA_manuscript.docx")
    m.save(out)
    print(f"wrote {out}  ({m.n_fig} figures, {m.n_tab} tables, {len(m.order)} references)")
    if unused:
        print("uncited references:", ", ".join(unused))


if __name__ == "__main__":
    main()
