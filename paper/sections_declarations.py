"""End matter required by Elsevier.

The CRediT statement uses the taxonomy's optional degree of contribution
(lead / equal / supporting).  Every author must agree with it before submission.
"""

REPO = "https://github.com/Pushkal-Gupta/Sensor-Drift-and-Data-Poisoning-in-IoT-Sensor-Networks"


def declarations(m):
    m.h1("CRediT authorship contribution statement")
    m.p("**Pushkal Gupta:** Conceptualization (lead), Methodology (lead), Software "
        "(lead), Formal analysis (lead), Investigation (lead), Data curation (lead), "
        "Validation (lead), Visualization (lead), Writing – original draft (lead), "
        "Writing – review & editing (lead), Project administration (lead). "
        "**Shridhar M:** Software (supporting), Validation (supporting), Investigation "
        "(supporting), Writing – review & editing (supporting). **Vanka Kaushik:** "
        "Software (supporting), Validation (supporting), Data curation (supporting), "
        "Writing – review & editing (supporting). **Syamasudha Veeragandham:** "
        "Supervision (lead), Resources (lead), Conceptualization (supporting), "
        "Writing – review & editing (supporting).")
    m.p("P. Gupta conceived the drift-versus-poisoning attribution problem and the "
        "physics-anchored formulation, designed the threat model and the two "
        "propositions, implemented the fog pipeline, the attack and scenario "
        "generators, all comparison systems and the edge kernel, designed and ran every "
        "experiment, carried out the statistical analysis and the Intel Lab study, "
        "identified and fixed the three defects reported in Section 7.3, produced all "
        "figures and wrote the manuscript. [[AUTHORS: state here the specific "
        "contributions of Shridhar M, V. Kaushik and S. Veeragandham (for example: who "
        "built the Wokwi/ESP32 node, who reviewed or re-ran which experiments, who "
        "checked the statistics), since Reviewer 1 asked how the work was divided.]]")
    m.h1("Declaration of competing interest")
    m.p("The authors declare that they have no known competing financial interests or "
        "personal relationships that could have appeared to influence the work reported "
        "in this paper.")
    m.h1("Funding")
    m.p("This research did not receive any specific grant from funding agencies in the "
        "public, commercial, or not-for-profit sectors.")
    m.h1("Data availability")
    m.p("The Intel Berkeley Research Lab data are publicly available "
        "{cite:intel2004}. All code — the fog pipeline, the scenario and attack "
        "generators, every comparison system, the experiment scripts that produce each "
        f"table and figure, and the regression tests — is available at {REPO}.")
    m.h1("Declaration of generative AI and AI-assisted technologies in the "
         "manuscript preparation process")
    m.p("During the preparation of this work the authors used Claude (Anthropic), a "
        "large language model, as a programming and writing assistant. "
        "[[AUTHORS: confirm or correct this division of labour before submission. The "
        "research question, the threat model, the propositions, the design of every "
        "experiment and the decision of which results to report were the authors'. Under "
        "the authors' direction, Claude wrote and refactored parts of the code (the fog "
        "pipeline, the comparison systems and the experiment scripts), executed the "
        "experiment scripts, and drafted and edited passages of the original manuscript "
        "and of this revision, including the new analyses requested by the reviewers. "
        "The authors reviewed the code, checked every reported number against the "
        "released artifacts, and revised the text.]] Every number in the paper is "
        "produced by released scripts from released per-episode artifacts, so the "
        "analysis can be audited independently. After using this tool, the authors "
        "reviewed and edited the content as needed and take full responsibility for the "
        "content of the published article.")
