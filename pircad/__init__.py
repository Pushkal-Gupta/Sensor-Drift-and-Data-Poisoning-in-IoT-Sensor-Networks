"""
PIRCAD -- Physics-Informed Root-Cause Disambiguation and Adaptive-Response
framework for trusted IoT sensing at the fog layer.

Generation 2 of the FHAD work.  Where FHAD (Gen 1) answered "is this sample
anomalous?", PIRCAD answers "*why* is it anomalous, and what should the system
therefore do differently?".

Module -> claim-element map (see docs/PATENT_PACKAGE.md):

    simulator.py    experimental plant + sensor + attack models (evidence only)
    twin.py         E1  expected-response generation (physics / process model)
    residual.py     E2  residual generation + residual-characteristic features
    triggers.py     E3  lightweight statistical trigger on the residual
    consistency.py  E4  cross-sensor physical-consistency localisation
    rootcause.py    E5  root-cause disambiguation + confidence
    escalation.py   E6  confidence-gated escalation (incl. quantum backend)
    governor.py     E7  adaptation / retraining inhibition
    actions.py      E8  differential response control
    fognode.py      the claimed system, assembled
"""

from .types import RootCause, Action, Decision, Frame
from .fognode import PIRCADFogNode

__all__ = ["RootCause", "Action", "Decision", "Frame", "PIRCADFogNode"]
__version__ = "2.0.0"
