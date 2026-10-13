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
        "identified and fixed the three defects reported in Section 7.2, produced all "
        "figures and wrote the manuscript.")
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
    m.p("During the preparation of this work the authors used Claude (Anthropic) to "
        "assist with software development, the execution of the experiments, and "
        "drafting and editing of the manuscript. After using this tool, the authors "
        "reviewed and edited the content as needed and take full responsibility for the "
        "content of the published article.")
