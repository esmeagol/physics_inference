"""
Configuration classes and constants for MOLT tracker.

This module defines the configuration schema and default parameters
for the MOLT tracker implementation.
"""

from typing import Dict, List
from dataclasses import dataclass, field
from .types import DEFAULT_SIMILARITY_WEIGHTS, DEFAULT_DIVERSITY_DISTRIBUTION


@dataclass
class MOLTTrackerConfig:
    """Configuration schema for MOLT tracker with default parameters."""
    
    # Default population sizes for different ball types
    population_sizes: Dict[str, int] = field(default_factory=lambda: {
        'white': 1500,      # Larger population for fast-moving white ball
        'red': 300,         # Medium population for red balls
        'yellow': 200,      # Smaller population for colored balls
        'green': 200,
        'brown': 200,
        'blue': 200,
        'pink': 200,
        'black': 200
    })
    
    # Default exploration radii for different ball types
    exploration_radii: Dict[str, int] = field(default_factory=lambda: {
        'white': 30,        # Larger search radius for white ball
        'red': 20,          # Medium radius for red balls
        'default': 15       # Default radius for other balls
    })
    
    # Expected ball counts for snooker (configurable based on game state)
    expected_ball_counts: Dict[str, int] = field(default_factory=lambda: {
        'white': 1,
        'red': 15,          # Configurable based on game state
        'yellow': 1,
        'green': 1,
        'brown': 1,
        'blue': 1,
        'pink': 1,
        'black': 1
    })
    
    # Histogram configuration
    histogram_bins: int = 16
    color_space: str = 'HSV'  # HSV or RGB
    
    # Similarity weights
    similarity_weights: Dict[str, float] = field(
        default_factory=lambda: DEFAULT_SIMILARITY_WEIGHTS.copy()
    )
    
    # Diversity distribution for population generation
    diversity_distribution: List[float] = field(
        default_factory=lambda: DEFAULT_DIVERSITY_DISTRIBUTION.copy()
    )
    
    # Tracking parameters
    min_confidence: float = 0.1
    max_frames_without_detection: int = 30
    
    @classmethod
    def create_default(cls) -> 'MOLTTrackerConfig':
        """Create a configuration with all default values."""
        return cls()
    
    @classmethod
    def create_for_snooker(cls) -> 'MOLTTrackerConfig':
        """Create a configuration optimized for snooker tracking."""
        return cls()  # Default is already optimized for snooker
    
    @classmethod
    def create_for_pool(cls) -> 'MOLTTrackerConfig':
        """Create a configuration optimized for pool/billiards tracking."""
        config = cls()
        # Adjust for pool (fewer balls, different dynamics)
        config.expected_ball_counts = {
            'white': 1,
            'yellow': 7,  # Stripes or solids
            'red': 7,     # Stripes or solids
            'black': 1    # 8-ball
        }
        config.population_sizes = {
            'white': 1000,
            'yellow': 250,
            'red': 250,
            'black': 300
        }
        return config
    
    def validate(self) -> None:
        """
        Validate the configuration parameters.
        
        Raises:
            ValueError: If any configuration parameter is invalid
        """
        # Validate population sizes
        for ball_type, size in self.population_sizes.items():
            if size <= 0:
                raise ValueError(f"Population size for {ball_type} must be positive, got {size}")
        
        # Validate exploration radii
        for ball_type, radius in self.exploration_radii.items():
            if radius <= 0:
                raise ValueError(f"Exploration radius for {ball_type} must be positive, got {radius}")
        
        # Validate expected ball counts
        for ball_type, count in self.expected_ball_counts.items():
            if count < 0:
                raise ValueError(f"Expected count for {ball_type} must be non-negative, got {count}")
        
        # Validate histogram bins
        if self.histogram_bins <= 0:
            raise ValueError(f"Histogram bins must be positive, got {self.histogram_bins}")
        
        # Validate color space
        if self.color_space not in ['HSV', 'RGB', 'LAB']:
            raise ValueError(f"Color space must be HSV, RGB, or LAB, got {self.color_space}")
        
        # Validate similarity weights
        if not (0 <= self.similarity_weights.get('histogram', 0) <= 1):
            raise ValueError("Histogram weight must be between 0 and 1")
        if not (0 <= self.similarity_weights.get('spatial', 0) <= 1):
            raise ValueError("Spatial weight must be between 0 and 1")
        
        # Validate diversity distribution
        if len(self.diversity_distribution) != 3:
            raise ValueError("Diversity distribution must have exactly 3 values")
        if abs(sum(self.diversity_distribution) - 1.0) > 1e-6:
            raise ValueError("Diversity distribution must sum to 1.0")
        if any(ratio < 0 for ratio in self.diversity_distribution):
            raise ValueError("All diversity distribution values must be non-negative")
        
        # Validate confidence threshold
        if not (0 <= self.min_confidence <= 1):
            raise ValueError("Minimum confidence must be between 0 and 1")
        
        # Validate max frames without detection
        if self.max_frames_without_detection <= 0:
            raise ValueError("Max frames without detection must be positive")
    
    def to_dict(self) -> Dict:
        """Convert configuration to dictionary format."""
        return {
            'population_sizes': self.population_sizes,
            'exploration_radii': self.exploration_radii,
            'expected_ball_counts': self.expected_ball_counts,
            'histogram_bins': self.histogram_bins,
            'color_space': self.color_space,
            'similarity_weights': self.similarity_weights,
            'diversity_distribution': self.diversity_distribution,
            'min_confidence': self.min_confidence,
            'max_frames_without_detection': self.max_frames_without_detection
        }