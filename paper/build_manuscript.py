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


def verdicts(R):
    """Narrative conclusions about the revision experiments (see revision_verdicts.py)."""
    from revision_verdicts import variant_verdict, variant_summary, aug_verdict
    return variant_verdict(R), variant_summary(R), aug_verdict(R)


def main() -> None:
    R = results.collect()
    m = Manuscript(REFS)
    R["variant_verdict"], R["variant_summary"], R["aug_verdict"] = verdicts(R)
    m.title(front.TITLE)
    m.centered(front.AUTHORS, 12, after=4)
    m.centered(front.AFFIL, 10)
    m.centered(front.EMAILS, 10)
    m.centered(front.CORR, 10, after=12)

    m.h1("Abstract")
    m.p(results.abstract(R), size=10.5)
    m.p(front.KEYWORDS, lead="Keywords:", size=10.5, before=6)

    front.methods(m, R)
    results.results_section(m, R)
    results.discussion_and_conclusion(m, R)
    results.declarations(m)
    results.appendix(m, R)
    unused = m.references()
    out = HERE / (sys.argv[1] if len(sys.argv) > 1 else "Ageing_or_Attack_Array_R1.docx")
    m.save(out)
    # Elsevier takes highlights as a separate file
    h = Manuscript(REFS)
    h.h1("Highlights")
    h.bullets(results.highlights(R))
    h.save(HERE / "Highlights_R1.docx")
    print(f"wrote {out}  ({m.n_fig} figures, {m.n_tab} tables, {len(m.order)} references)")
    if unused:
        print("uncited references:", ", ".join(unused))


if __name__ == "__main__":
    main()
