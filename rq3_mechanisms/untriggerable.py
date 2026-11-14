from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class HypothesisTestResult:
    score_clean: float
    score_twin: float
    score_untriggerable: float
    score_unreached: float
    score_reached: float
    favored_hypothesis: str


def evaluate_untriggerable_contrast(
    scores_clean: np.ndarray,
    scores_twin: np.ndarray,
    scores_untriggerable: np.ndarray,
    scores_unreached: np.ndarray,
    scores_reached: np.ndarray,
) -> HypothesisTestResult:
    m_clean = float(np.mean(scores_clean))
    m_twin = float(np.mean(scores_twin))
    m_untrig = float(np.mean(scores_untriggerable))
    m_unreached = float(np.mean(scores_unreached))
    m_reached = float(np.mean(scores_reached))

    gap_unreached_untrig = m_unreached - m_untrig
    gap_untrig_clean = m_untrig - m_clean

    if m_twin > m_unreached * 0.8:
        favored = "H0: Surface Text Artifact (Rejected by Benign Twins)"
    elif gap_unreached_untrig > gap_untrig_clean:
        favored = "H1: Pre-Firing Readiness (Triggerable Context Required)"
    else:
        favored = "H2: Payload Presence (Passive Syntax Feature)"

    return HypothesisTestResult(
        score_clean=m_clean,
        score_twin=m_twin,
        score_untriggerable=m_untrig,
        score_unreached=m_unreached,
        score_reached=m_reached,
        favored_hypothesis=favored,
    )
