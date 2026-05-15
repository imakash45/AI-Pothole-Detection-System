"""
Pothole Prahari — Detection & Decision System
app/detection.py

Converts raw YOLO pothole detections into:
  - Road condition  : Good / Moderate / Poor
  - Speed limit     : 60 / 40 / 20 km/h
  - Warning message : context-aware hazard text

Design principles:
  - Fully rule-based: all decisions use explicit deterministic rules.
  - No scoring, density metrics, or normalisation in decision logic.
  - Decisions depend ONLY on pothole counts and severity classification.
  - display_score is computed after decisions for UI use only.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

RoadCondition = Literal["Good", "Moderate", "Poor"]
Severity = Literal["Minor", "Moderate", "Severe"]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Detection:
    """Single pothole detection from the YOLO model."""
    bbox: tuple[float, float, float, float]   # (x1, y1, x2, y2) — absolute pixels
    confidence: float                          # 0.0 – 1.0
    frame_width: int
    frame_height: int

    @property
    def relative_area(self) -> float:
        """Bounding-box area as a fraction of the total frame area."""
        x1, y1, x2, y2 = self.bbox
        box_area = abs((x2 - x1) * (y2 - y1))
        frame_area = self.frame_width * self.frame_height
        if frame_area == 0:
            return 0.0
        return box_area / frame_area


@dataclass
class DetectionSummary:
    """Aggregated counts produced by summarise_detections()."""
    total: int = 0
    minor: int = 0
    moderate: int = 0
    severe: int = 0
    detections: list[Detection] = field(default_factory=list)


@dataclass
class RoadAnalysis:
    """Final output of analyze_road_condition()."""
    condition: RoadCondition
    speed_kmh: int
    warning: str
    summary: DetectionSummary
    display_score: float   # 0–100, for UI only — never used in decision logic


# ---------------------------------------------------------------------------
# Severity thresholds (relative bbox area)
# ---------------------------------------------------------------------------
#
#   < MINOR_THRESHOLD              → Minor   (small surface crack)
#   MINOR_THRESHOLD – SEV_THRESHOLD → Moderate (visible pothole)
#   ≥ SEV_THRESHOLD               → Severe  (large / deep pothole)
#
MINOR_THRESHOLD = 0.005   # 0.5 % of frame
SEVERE_THRESHOLD = 0.025  # 2.5 % of frame


def classify_severity(detection: Detection) -> Severity:
    """
    Classify a single detection as Minor, Moderate, or Severe
    based on its bounding-box area relative to the frame.
    """
    area = detection.relative_area
    if area >= SEVERE_THRESHOLD:
        return "Severe"
    elif area >= MINOR_THRESHOLD:
        return "Moderate"
    else:
        return "Minor"


# ---------------------------------------------------------------------------
# Summarisation
# ---------------------------------------------------------------------------

def summarise_detections(detections: list[Detection]) -> DetectionSummary:
    """
    Aggregate a list of Detection objects into counts per severity class.

    Args:
        detections: Raw YOLO outputs for one frame / image.

    Returns:
        DetectionSummary with total + per-severity counts.
    """
    summary = DetectionSummary(detections=detections)

    for det in detections:
        severity = classify_severity(det)
        summary.total += 1
        if severity == "Severe":
            summary.severe += 1
        elif severity == "Moderate":
            summary.moderate += 1
        else:
            summary.minor += 1

    return summary


# ---------------------------------------------------------------------------
# Core logic: road condition
# ---------------------------------------------------------------------------

def analyze_road_condition(summary: DetectionSummary) -> RoadCondition:
    """
    Determine road condition using strict, ordered rules.

    Rules are evaluated top-to-bottom; the FIRST matching rule wins.
    No scoring, density metrics, or normalisation is used — decisions
    depend only on pothole counts and severity classification.

      Rule 1 — Severe ≥ 1        → Poor
               Critical hazard; immediate risk to vehicle and driver.

      Rule 2 — Moderate ≥ 2      → Moderate
               Multiple clearly visible potholes affecting ride quality.

      Rule 3 — Total ≥ 3         → Moderate
               Scattered road damage regardless of individual severity.

      Rule 4 — Otherwise         → Good
               Minimal or no pothole presence; safe driving conditions.

    Properties:
      - No overlapping rules: each rule covers a distinct scenario.
      - No redundant rules: every rule contributes uniquely.
      - Fully deterministic: identical inputs always produce identical output.
      - No hidden dependencies: display_score is computed after this function
        and must never be fed back into condition logic.
    """
    # Rule 1 — any severe pothole → immediate Poor
    if summary.severe >= 1:
        return "Poor"

    # Rule 2 — two or more moderate potholes → Moderate
    if summary.moderate >= 2:
        return "Moderate"

    # Rule 3 — three or more potholes of any severity → Moderate
    if summary.total >= 3:
        return "Moderate"

    # Rule 4 — minimal or no damage → Good
    return "Good"


# ---------------------------------------------------------------------------
# Speed recommendation
# ---------------------------------------------------------------------------

def get_speed_recommendation(condition: RoadCondition) -> int:
    """
    Return the safe driving speed in km/h for a given road condition.

    Speed is always derived from condition — never from raw counts directly.

        Good     → 60 km/h
        Moderate → 40 km/h
        Poor     → 20 km/h
    """
    speed_map: dict[RoadCondition, int] = {
        "Good": 60,
        "Moderate": 40,
        "Poor": 20,
    }
    return speed_map[condition]


# ---------------------------------------------------------------------------
# Warning message
# ---------------------------------------------------------------------------

def get_warning_message(condition: RoadCondition, summary: DetectionSummary) -> str:
    """
    Generate a hazard-aware warning message.

    Warning reflects actual risk, not just condition:
      - Severe pothole present → explicit severe-hazard alert
      - Moderate condition     → general caution for uneven surface
      - Good condition         → all-clear message
    """
    if summary.severe >= 1:
        return "⚠️  Severe pothole detected ahead — reduce speed immediately."
    elif condition == "Moderate":
        return "⚠️  Drive carefully — uneven road surface detected."
    else:
        return "✅  Road is safe for normal driving."


# ---------------------------------------------------------------------------
# Display score (UI only — not used in decision logic)
# ---------------------------------------------------------------------------

def _compute_display_score(summary: DetectionSummary) -> float:
    """
    Compute a 0–100 road health score for UI visualisation only.

    This value is calculated AFTER road condition is determined and must
    never influence condition, speed, or warning decisions. It exists solely
    to populate dashboard indicators and progress bars.

    Formula: each pothole type contributes a hazard weight; the total is
    capped at 100 and inverted so that 100 = perfect road, 0 = worst road.
    No density factors or normalisation ratios are used.
    """
    if summary.total == 0:
        return 100.0

    # Weighted hazard index — higher = worse road
    hazard = (
        summary.severe * 40 +
        summary.moderate * 15 +
        summary.minor * 5
    )
    # Cap and invert so 100 = perfect road, 0 = worst road
    score = max(0.0, 100.0 - min(hazard, 100.0))
    return round(score, 1)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_analysis(detections: list[Detection]) -> RoadAnalysis:
    """
    Full pipeline: raw detections → RoadAnalysis.

    Usage example:
        from app.detection import Detection, run_analysis

        dets = [
            Detection(bbox=(100, 200, 180, 260), confidence=0.91,
                      frame_width=1280, frame_height=720),
        ]
        result = run_analysis(dets)
        print(result.condition)    # "Good" / "Moderate" / "Poor"
        print(result.speed_kmh)    # 60 / 40 / 20
        print(result.warning)      # human-readable message

    Args:
        detections: List of Detection objects from the YOLO output layer.

    Returns:
        RoadAnalysis dataclass containing condition, speed, warning, summary,
        and a UI-only display score.
    """
    summary = summarise_detections(detections)
    condition = analyze_road_condition(summary)
    speed = get_speed_recommendation(condition)
    warning = get_warning_message(condition, summary)
    score = _compute_display_score(summary)

    return RoadAnalysis(
        condition=condition,
        speed_kmh=speed,
        warning=warning,
        summary=summary,
        display_score=score,
    )


# ---------------------------------------------------------------------------
# Convenience: build a Detection from a YOLO result dict
# ---------------------------------------------------------------------------

def detection_from_yolo(
    yolo_result: dict,
    frame_width: int,
    frame_height: int,
) -> Detection:
    """
    Construct a Detection from a typical YOLO result dictionary.

    Expected keys in yolo_result:
        "bbox"        : [x1, y1, x2, y2]
        "confidence"  : float

    Adapt this helper to match your actual YOLO output format.
    """
    return Detection(
        bbox=tuple(yolo_result["bbox"]),
        confidence=float(yolo_result["confidence"]),
        frame_width=frame_width,
        frame_height=frame_height,
    )