from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum, auto

class ErrorType(Enum):
    CORRECT = auto()
    OVER_DETECTION = auto()
    UNDER_DETECTION = auto()
    ILLEGAL_DISAPPEARANCE = auto()
    ILLEGAL_REAPPEARANCE = auto()
    DUPLICATION = auto()
    OTHER = auto()

@dataclass
class IllegalChange:
    ball_type: str
    change_type: ErrorType
    previous_count: int
    current_count: int
    timestamp: float
    description: str

@dataclass
class DuplicationError:
    ball_type: str
    positions: List[Tuple[float, float]]
    confidences: List[float]
    distance: float
    timestamp: float

@dataclass
class CountError:
    ball_type: str
    error_type: ErrorType
    expected_count: int
    detected_count: int
    magnitude: int

@dataclass
class MomentEvaluation:
    moment_idx: int
    timestamp: float
    expected_state: Dict[str, Any]
    detected_counts: Dict[str, int]
    active_events: List[Any]  # Replace Any with actual event type if known
    detected_positions: List[Dict[str, Any]]
    count_errors: List[CountError]
    illegal_changes: List[IllegalChange]
    duplication_errors: List[DuplicationError]
    suppressed: bool

class MomentEvaluator:
    def __init__(self, config: Optional[Dict[str, Any]] = None): ...
    def evaluate_moment(
        self, 
        expected_state: Dict[str, Any], 
        detected_objects: List[Any],  # Replace Any with actual detected object type
        timestamp: float,
        frame_idx: int
    ) -> MomentEvaluation: ...
