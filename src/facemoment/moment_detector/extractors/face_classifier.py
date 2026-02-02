"""Face classifier extractor - classifies detected faces by role.

Depends on face_detect to classify faces as:
- main: Primary subject (driver/main person)
- passenger: Secondary subject (co-passenger)
- transient: Temporarily detected face (passing by)
- noise: False detection or low-quality face
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List
from collections import defaultdict
import logging
import time

from visualbase import Frame

from facemoment.moment_detector.extractors.base import (
    BaseExtractor,
    Observation,
    FaceObservation,
)
from facemoment.moment_detector.extractors.outputs import FaceDetectOutput

logger = logging.getLogger(__name__)


@dataclass
class ClassifiedFace:
    """Face with classification info."""
    face: FaceObservation
    role: str  # "main", "passenger", "transient", "noise"
    confidence: float  # Classification confidence
    track_length: int  # Number of consecutive frames tracked
    avg_area: float  # Average area ratio over track


@dataclass
class FaceClassifierOutput:
    """Output from FaceClassifierExtractor."""
    faces: List[ClassifiedFace] = field(default_factory=list)
    main_face: Optional[ClassifiedFace] = None
    passenger_faces: List[ClassifiedFace] = field(default_factory=list)
    transient_count: int = 0
    noise_count: int = 0


class FaceClassifierExtractor(BaseExtractor):
    """Classifies detected faces by their role in the scene.

    Uses temporal tracking and spatial analysis to classify faces:
    - main: Largest, most central, consistently present
    - passenger: Secondary position, consistently present
    - transient: Appears for only a few frames
    - noise: Too small, edge of frame, or low confidence

    depends: ["face_detect"]

    Args:
        min_track_frames: Minimum frames to be considered non-transient (default: 5)
        min_area_ratio: Minimum face area ratio to not be noise (default: 0.01)
        min_confidence: Minimum detection confidence (default: 0.5)
        main_zone: Normalized x-range for main subject (default: (0.3, 0.7))
        edge_margin: Margin from edge to be considered valid (default: 0.05)

    Example:
        >>> classifier = FaceClassifierExtractor()
        >>> # Use with FlowGraph
        >>> graph = (FlowGraphBuilder()
        ...     .source()
        ...     .path("detect", extractors=[FaceDetectionExtractor()])
        ...     .path("classify", extractors=[FaceClassifierExtractor()])
        ...     .build())
    """

    depends = ["face_detect"]

    def __init__(
        self,
        min_track_frames: int = 5,
        min_area_ratio: float = 0.01,
        min_confidence: float = 0.5,
        main_zone: tuple[float, float] = (0.3, 0.7),
        edge_margin: float = 0.05,
    ):
        self._min_track_frames = min_track_frames
        self._min_area_ratio = min_area_ratio
        self._min_confidence = min_confidence
        self._main_zone = main_zone
        self._edge_margin = edge_margin

        # Tracking state: face_id -> history
        self._track_history: Dict[int, List[FaceObservation]] = defaultdict(list)
        self._track_stats: Dict[int, Dict] = {}  # Aggregated stats per face_id

    @property
    def name(self) -> str:
        return "face_classifier"

    def initialize(self) -> None:
        self._track_history.clear()
        self._track_stats.clear()
        logger.info("FaceClassifierExtractor initialized")

    def cleanup(self) -> None:
        self._track_history.clear()
        self._track_stats.clear()
        logger.info("FaceClassifierExtractor cleaned up")

    def extract(
        self,
        frame: Frame,
        deps: Optional[Dict[str, Observation]] = None,
    ) -> Optional[Observation]:
        # Support both face_detect (split) and face (composite) extractors
        face_obs = None
        faces = []

        if deps:
            if "face_detect" in deps:
                # Split extractor: FaceDetectionExtractor
                face_obs = deps["face_detect"]
                if face_obs.data and hasattr(face_obs.data, 'faces'):
                    faces = face_obs.data.faces
            elif "face" in deps:
                # Composite extractor: FaceExtractor
                face_obs = deps["face"]
                faces = face_obs.faces if face_obs.faces else []

        if face_obs is None:
            logger.warning("FaceClassifierExtractor: no face_detect or face dependency")
            return None

        if not faces:
            return Observation(
                source=self.name,
                frame_id=frame.frame_id,
                t_ns=frame.t_src_ns,
                signals={
                    "main_detected": 0,
                    "passenger_count": 0,
                    "transient_count": 0,
                    "noise_count": 0,
                },
                data=FaceClassifierOutput(),
            )

        start_ns = time.perf_counter_ns()

        # Update tracking history
        current_ids = set()
        for face in faces:
            self._track_history[face.face_id].append(face)
            current_ids.add(face.face_id)
            self._update_stats(face.face_id, face)

        # Classify each face
        classified_faces = []
        for face in faces:
            role, confidence = self._classify_face(face)
            track_length = len(self._track_history[face.face_id])
            avg_area = self._track_stats[face.face_id].get("avg_area", face.area_ratio)

            classified = ClassifiedFace(
                face=face,
                role=role,
                confidence=confidence,
                track_length=track_length,
                avg_area=avg_area,
            )
            classified_faces.append(classified)

        # Separate by initial classification
        # Main: exactly 1, Passenger: at most 1
        candidates = []  # (score, classified_face) for main/passenger selection
        transient_faces = []  # Original transient faces
        noise_faces = []  # Original noise faces

        for cf in classified_faces:
            if cf.role in ("main", "passenger"):
                # Score based on avg_area and track_length
                score = cf.avg_area * 100 + cf.track_length * 0.1
                candidates.append((score, cf))
            elif cf.role == "transient":
                transient_faces.append(cf)
            else:  # noise
                noise_faces.append(cf)

        # Sort candidates by score (descending)
        candidates.sort(key=lambda x: x[0], reverse=True)

        # Assign roles: best -> main, second best -> passenger (if exists)
        main_face = None
        passenger_faces = []
        demoted_to_transient = []  # Candidates beyond top 2

        if candidates:
            # Best candidate becomes main
            best = candidates[0][1]
            main_face = ClassifiedFace(
                face=best.face,
                role="main",
                confidence=best.confidence,
                track_length=best.track_length,
                avg_area=best.avg_area,
            )

            # Second best becomes passenger (at most 1)
            if len(candidates) > 1:
                second = candidates[1][1]
                passenger = ClassifiedFace(
                    face=second.face,
                    role="passenger",
                    confidence=second.confidence,
                    track_length=second.track_length,
                    avg_area=second.avg_area,
                )
                passenger_faces = [passenger]

            # Remaining candidates (3rd+) become transient
            for _, cf in candidates[2:]:
                demoted_to_transient.append(ClassifiedFace(
                    face=cf.face,
                    role="transient",
                    confidence=cf.confidence,
                    track_length=cf.track_length,
                    avg_area=cf.avg_area,
                ))

        # Rebuild classified_faces with corrected roles (for visualization)
        classified_faces = []
        if main_face:
            classified_faces.append(main_face)
        classified_faces.extend(passenger_faces)
        classified_faces.extend(demoted_to_transient)
        classified_faces.extend(transient_faces)
        classified_faces.extend(noise_faces)

        # Count
        transient_count = len(transient_faces) + len(demoted_to_transient)
        noise_count = len(noise_faces)

        # Clean up old tracks (not seen for a while)
        self._cleanup_old_tracks(current_ids)

        timing = {"total_ms": (time.perf_counter_ns() - start_ns) / 1_000_000}

        return Observation(
            source=self.name,
            frame_id=frame.frame_id,
            t_ns=frame.t_src_ns,
            signals={
                "main_detected": 1 if main_face else 0,
                "passenger_count": len(passenger_faces),
                "transient_count": transient_count,
                "noise_count": noise_count,
                "total_faces": len(classified_faces),
            },
            data=FaceClassifierOutput(
                faces=classified_faces,
                main_face=main_face,
                passenger_faces=passenger_faces,
                transient_count=transient_count,
                noise_count=noise_count,
            ),
            timing=timing,
        )

    def _classify_face(self, face: FaceObservation) -> tuple[str, float]:
        """Classify a face based on various criteria.

        Returns:
            Tuple of (role, confidence)
        """
        track_length = len(self._track_history[face.face_id])
        stats = self._track_stats.get(face.face_id, {})

        # Check for noise first
        if self._is_noise(face, stats):
            return ("noise", 0.9)

        # Check for transient (not enough tracking history)
        if track_length < self._min_track_frames:
            return ("transient", 0.7)

        # Classify as main or passenger based on position and size
        center_x = face.bbox[0] + face.bbox[2] / 2
        main_zone_left, main_zone_right = self._main_zone

        # Score for being "main"
        main_score = 0.0

        # Position stability is the most important factor (camera is fixed)
        # Passengers have stable positions, transients move around
        position_stability = stats.get("position_stability", 0.5)
        max_drift = stats.get("max_position_drift", 0.0)

        # High drift indicates transient (passing by)
        if max_drift > 0.15:  # 15% of frame movement
            return ("transient", 0.8)

        # Position stability score (most important - 40% weight)
        main_score += position_stability * 0.4

        # Size score: larger faces are more likely main (30% weight)
        avg_area = stats.get("avg_area", face.area_ratio)
        if avg_area > 0.05:  # Large face
            main_score += 0.3
        elif avg_area > 0.02:  # Medium face
            main_score += 0.2
        else:
            main_score += 0.1

        # Position in frame: center vs edge (20% weight)
        if main_zone_left <= center_x <= main_zone_right:
            main_score += 0.2
        else:
            main_score += 0.05

        # Inside frame score (10% weight)
        if face.inside_frame:
            main_score += 0.1

        # Classify based on score
        # High score with good stability = main/passenger candidate
        if main_score >= 0.5 and position_stability >= 0.5:
            return ("main", main_score)
        elif main_score >= 0.3 and position_stability >= 0.3:
            return ("passenger", main_score)
        else:
            return ("transient", main_score)

    def _is_noise(self, face: FaceObservation, stats: Dict) -> bool:
        """Check if face is likely noise/false detection."""
        # Too small
        if face.area_ratio < self._min_area_ratio:
            return True

        # Low confidence
        if face.confidence < self._min_confidence:
            return True

        # At edge of frame
        x, y, w, h = face.bbox
        if (x < self._edge_margin or
            y < self._edge_margin or
            x + w > 1 - self._edge_margin or
            y + h > 1 - self._edge_margin):
            # Edge faces with small area are likely noise
            if face.area_ratio < 0.02:
                return True

        return False

    def _update_stats(self, face_id: int, face: FaceObservation) -> None:
        """Update aggregated stats for a face track.

        Tracks position stability which is key for passenger detection:
        - Main/passenger have fixed camera positions, so their positions are stable
        - Transient faces (passers-by) have varying positions
        """
        center_x = face.bbox[0] + face.bbox[2] / 2
        center_y = face.bbox[1] + face.bbox[3] / 2

        if face_id not in self._track_stats:
            self._track_stats[face_id] = {
                "avg_area": face.area_ratio,
                "avg_center_x": center_x,
                "avg_center_y": center_y,
                "position_stability": 1.0,  # Position stability score [0-1]
                "max_position_drift": 0.0,  # Maximum position change observed
                "frame_count": 1,
            }
        else:
            stats = self._track_stats[face_id]
            n = stats["frame_count"]

            # Exponential moving average for area
            alpha = 0.3
            stats["avg_area"] = (1 - alpha) * stats["avg_area"] + alpha * face.area_ratio

            # Calculate position drift (distance from average position)
            drift_x = abs(center_x - stats["avg_center_x"])
            drift_y = abs(center_y - stats["avg_center_y"])
            position_drift = (drift_x ** 2 + drift_y ** 2) ** 0.5

            # Update max drift seen
            stats["max_position_drift"] = max(stats["max_position_drift"], position_drift)

            # Update average position (slow update to maintain reference point)
            alpha_pos = 0.1  # Slower update for position
            stats["avg_center_x"] = (1 - alpha_pos) * stats["avg_center_x"] + alpha_pos * center_x
            stats["avg_center_y"] = (1 - alpha_pos) * stats["avg_center_y"] + alpha_pos * center_y

            # Position stability: penalize large movements
            # Passengers have drift < 0.05 (5% of frame), transients have drift > 0.1
            if position_drift > 0.1:
                # Large movement - likely transient
                stats["position_stability"] = max(0, stats["position_stability"] - 0.3)
            elif position_drift > 0.05:
                # Medium movement - might be head movement
                stats["position_stability"] = max(0, stats["position_stability"] - 0.1)
            else:
                # Small movement - stable, recover slowly
                stats["position_stability"] = min(1, stats["position_stability"] + 0.02)

            stats["frame_count"] = n + 1

    def _cleanup_old_tracks(self, current_ids: set, max_history: int = 30) -> None:
        """Remove old tracks and limit history length."""
        # Remove tracks not seen recently
        to_remove = []
        for face_id in self._track_history:
            if face_id not in current_ids:
                # Mark for removal if not seen for a while
                history = self._track_history[face_id]
                if len(history) > 0:
                    # Keep some history for re-identification
                    if len(history) > max_history:
                        to_remove.append(face_id)

        for face_id in to_remove:
            del self._track_history[face_id]
            if face_id in self._track_stats:
                del self._track_stats[face_id]

        # Limit history length for active tracks
        for face_id in self._track_history:
            if len(self._track_history[face_id]) > max_history:
                self._track_history[face_id] = self._track_history[face_id][-max_history:]
