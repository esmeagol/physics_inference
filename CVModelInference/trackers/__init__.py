"""
Trackers Package for CVModelInference

This package contains implementations of various tracking algorithms.
"""

from .deepsort_tracker import DeepSORTTracker
from .sv_bytetrack_tracker import SVByteTrackTracker

__all__ = ['DeepSORTTracker', 'SVByteTrackTracker']
