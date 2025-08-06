from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

@dataclass
class ProcessedEvent:
    event_type: str
    timestamp: float
    frame_idx: int
    event_data: Dict[str, Any]
    confidence: float
    is_active: bool

class EventProcessor:
    def __init__(self, config: Optional[Dict[str, Any]] = None): ...
    def process_frame(
        self, 
        frame: Any,  # Replace with actual frame type (e.g., numpy.ndarray)
        timestamp: float,
        frame_idx: int
    ) -> List[ProcessedEvent]: ...
    def get_active_events(self) -> List[ProcessedEvent]: ...
