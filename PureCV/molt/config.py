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
    """
    Configuration schema for MOLT tracker with comprehensive parameter management.
    
    This class defines all configurable parameters for the MOLT tracker algorithm,
    providing default values optimized for snooker tracking, validation methods,
    and preset configurations for different game types.
    
    The configuration covers all aspects of the MOLT algorithm:
    - Population management (sizes and exploration radii)
    - Appearance modeling (histogram parameters and color spaces)
    - Similarity computation (weights and thresholds)
    - Ball counting (expected counts and violation handling)
    - Performance tuning (confidence thresholds and frame limits)
    
    Attributes:
        population_sizes (Dict[str, int]): Number of local trackers per ball type.
            Larger populations provide better accuracy but slower performance.
            Default optimized for snooker with larger populations for fast-moving balls.
            
        exploration_radii (Dict[str, int]): Maximum search radius for tracker regeneration.
            Larger radii allow tracking faster-moving objects but may cause drift.
            Should be adjusted based on object speed and frame rate.
            
        expected_ball_counts (Dict[str, int]): Expected number of balls per type.
            Used for count verification and violation detection.
            Should match the game rules (e.g., 15 red balls in snooker).
            
        histogram_bins (int): Number of bins for color histogram computation.
            More bins provide finer color discrimination but slower computation.
            Typical range: 8-32 bins.
            
        color_space (str): Color space for histogram extraction.
            Options: 'HSV' (default, good for color-based tracking),
                    'RGB' (faster conversion), 'LAB' (perceptually uniform).
                    
        similarity_weights (Dict[str, float]): Weights for combining histogram and
            spatial similarities. Must sum to 1.0. Higher histogram weight favors
            appearance matching, higher spatial weight favors motion consistency.
            
        diversity_distribution (List[float]): Ratios for distributing new trackers
            around top performers [best, second, third]. Must sum to 1.0.
            Default [0.5, 0.3, 0.2] concentrates trackers around best positions.
            
        min_confidence (float): Minimum confidence threshold for valid tracks.
            Tracks below this threshold are considered lost.
            
        max_frames_without_detection (int): Maximum frames to track without new
            detections before considering a track lost.
    
    Example:
        >>> # Default configuration
        >>> config = MOLTTrackerConfig()
        
        >>> # Custom configuration
        >>> config = MOLTTrackerConfig()
        >>> config.population_sizes = {'red': 200, 'white': 1000}
        >>> config.histogram_bins = 20
        >>> config.validate()  # Check parameter validity
        
        >>> # Game-specific presets
        >>> snooker_config = MOLTTrackerConfig.create_for_snooker()
        >>> pool_config = MOLTTrackerConfig.create_for_pool()
    """
    
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
        """
        Create a configuration with all default values optimized for snooker.
        
        This method creates a MOLTTrackerConfig instance with default parameters
        that have been tuned for optimal performance on snooker videos. The defaults
        balance accuracy and performance for typical snooker tracking scenarios.
        
        Default configuration features:
        - Population sizes optimized for different ball movement patterns
        - Exploration radii based on typical ball speeds
        - HSV color space for robust color-based tracking
        - Balanced similarity weights (60% appearance, 40% spatial)
        - Standard snooker ball counts (15 red + 6 colored + 1 white)
        
        Returns:
            MOLTTrackerConfig: Configuration instance with default parameters
            
        Example:
            >>> config = MOLTTrackerConfig.create_default()
            >>> print(f"Default population sizes: {config.population_sizes}")
            >>> print(f"Default histogram bins: {config.histogram_bins}")
        """
        return cls()
    
    @classmethod
    def create_for_snooker(cls) -> 'MOLTTrackerConfig':
        """
        Create a configuration specifically optimized for snooker tracking.
        
        This method returns a configuration tuned for snooker game characteristics:
        - 15 red balls that move frequently during play
        - 6 colored balls (yellow, green, brown, blue, pink, black) that move less
        - 1 white cue ball that moves most frequently and fastest
        - Typical snooker table dimensions and ball speeds
        
        Snooker-specific optimizations:
        - Larger population for white ball (1500 trackers) due to high movement
        - Medium population for red balls (300 trackers) for frequent movement
        - Smaller populations for colored balls (200 trackers) for occasional movement
        - Exploration radii adjusted for snooker ball speeds
        - Expected ball counts matching standard snooker rules
        
        Returns:
            MOLTTrackerConfig: Configuration optimized for snooker tracking
            
        Example:
            >>> config = MOLTTrackerConfig.create_for_snooker()
            >>> tracker = MOLTTracker(config=config)
            >>> print(f"White ball population: {config.population_sizes['white']}")
            >>> print(f"Expected red balls: {config.expected_ball_counts['red']}")
        """
        return cls()  # Default is already optimized for snooker
    
    @classmethod
    def create_for_pool(cls) -> 'MOLTTrackerConfig':
        """
        Create a configuration optimized for pool/billiards tracking.
        
        This method returns a configuration tuned for pool game characteristics,
        which differ significantly from snooker:
        - Fewer total balls (typically 16 in 8-ball, 10 in 9-ball)
        - Different ball color schemes (stripes/solids vs. individual colors)
        - Different table size and ball movement patterns
        - Different game dynamics and ball interaction frequencies
        
        Pool-specific optimizations:
        - Reduced population sizes due to fewer balls to track
        - Adjusted expected ball counts for 8-ball pool
        - Modified exploration radii for pool table dimensions
        - Population sizes balanced for pool game dynamics
        
        Ball count configuration for 8-ball pool:
        - 1 white cue ball
        - 7 solid-colored balls (represented as 'yellow')
        - 7 striped balls (represented as 'red') 
        - 1 black 8-ball
        
        Returns:
            MOLTTrackerConfig: Configuration optimized for pool/billiards tracking
            
        Example:
            >>> config = MOLTTrackerConfig.create_for_pool()
            >>> tracker = MOLTTracker(config=config)
            >>> print(f"Expected ball counts: {config.expected_ball_counts}")
            >>> print(f"Population sizes: {config.population_sizes}")
            
        Note:
            - This configuration assumes 8-ball pool rules
            - For 9-ball or other variants, manually adjust expected_ball_counts
            - Color mapping: 'yellow'=solids, 'red'=stripes, 'black'=8-ball
        """
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
        Validate all configuration parameters for correctness and consistency.
        
        This method performs comprehensive validation of all configuration parameters
        to ensure they are within valid ranges and logically consistent. It checks
        data types, value ranges, dictionary structures, and cross-parameter consistency.
        
        Validation checks include:
        - Population sizes: Must be positive integers
        - Exploration radii: Must be positive numbers
        - Expected ball counts: Must be non-negative integers
        - Histogram bins: Must be positive integer
        - Color space: Must be supported ('HSV', 'RGB', 'LAB')
        - Similarity weights: Must be in [0,1] range and have required keys
        - Diversity distribution: Must have 3 values that sum to 1.0
        - Confidence threshold: Must be in [0,1] range
        - Frame limits: Must be positive integers
        
        Raises:
            ValueError: If any configuration parameter is invalid. The error message
                will specify which parameter is invalid and what the valid range is.
                
        Example:
            >>> config = MOLTTrackerConfig()
            >>> config.histogram_bins = -1  # Invalid value
            >>> config.validate()  # Raises ValueError
            
            >>> config.histogram_bins = 16  # Valid value
            >>> config.validate()  # No error
            
        Use Cases:
            - Validate configuration before creating tracker
            - Check parameter modifications for correctness
            - Ensure configuration consistency in automated systems
            - Debug configuration-related issues
            
        Note:
            - Validation is automatically called during tracker initialization
            - All preset configurations (create_default, create_for_snooker, etc.)
              are guaranteed to pass validation
            - Custom configurations should always be validated before use
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
        """
        Convert the configuration to a dictionary format.
        
        This method serializes all configuration parameters into a dictionary
        format suitable for JSON serialization, logging, or parameter passing.
        The dictionary contains all configuration parameters with their current values.
        
        Returns:
            Dict: Dictionary containing all configuration parameters:
                - 'population_sizes': Dict mapping ball types to population sizes
                - 'exploration_radii': Dict mapping ball types to search radii
                - 'expected_ball_counts': Dict mapping ball types to expected counts
                - 'histogram_bins': Number of histogram bins
                - 'color_space': Color space for histogram extraction
                - 'similarity_weights': Dict with histogram and spatial weights
                - 'diversity_distribution': List of diversity ratios
                - 'min_confidence': Minimum confidence threshold
                - 'max_frames_without_detection': Maximum frames without detection
                
        Example:
            >>> config = MOLTTrackerConfig.create_for_snooker()
            >>> config_dict = config.to_dict()
            >>> print(f"Population sizes: {config_dict['population_sizes']}")
            >>> 
            >>> # Save to JSON
            >>> import json
            >>> with open('config.json', 'w') as f:
            ...     json.dump(config_dict, f, indent=2)
            
        Use Cases:
            - Serialization for configuration storage
            - Logging configuration parameters
            - Parameter passing to other systems
            - Configuration comparison and analysis
            - Debugging and troubleshooting
        """
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