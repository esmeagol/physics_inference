#!/usr/bin/env python3
"""
Unit tests for MOLTTrackerConfig class.

This module tests the configuration management functionality
of the MOLT tracker.
"""

import sys
import os
# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from src.tracking.trackers.molt.config import MOLTTrackerConfig


def test_default_config_creation() -> None:
    """Test creating a default configuration."""
    config = MOLTTrackerConfig.create_default()
    
    # Check that all required fields are present
    assert config.population_sizes is not None
    assert config.exploration_radii is not None
    assert config.expected_ball_counts is not None
    assert config.histogram_bins > 0
    assert config.color_space in ['HSV', 'RGB', 'LAB']
    assert config.similarity_weights is not None
    assert config.diversity_distribution is not None
    assert 0 <= config.min_confidence <= 1
    assert config.max_frames_without_detection > 0


def test_snooker_config_creation() -> None:
    """Test creating a snooker-specific configuration."""
    config = MOLTTrackerConfig.create_for_snooker()
    
    # Check snooker-specific values
    assert config.expected_ball_counts['red'] == 15
    assert config.expected_ball_counts['white'] == 1
    assert config.expected_ball_counts['black'] == 1


def test_pool_config_creation() -> None:
    """Test creating a pool-specific configuration."""
    config = MOLTTrackerConfig.create_for_pool()
    
    # Check pool-specific values
    assert config.expected_ball_counts['white'] == 1
    assert config.expected_ball_counts['black'] == 1
    assert 'red' in config.expected_ball_counts or 'yellow' in config.expected_ball_counts


def test_config_validation() -> None:
    """Test configuration validation."""
    config = MOLTTrackerConfig.create_default()
    
    # Valid config should pass validation
    config.validate()
    
    # Test invalid population size
    config.population_sizes['white'] = -1
    try:
        config.validate()
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    
    # Reset and test invalid exploration radius
    config = MOLTTrackerConfig.create_default()
    config.exploration_radii['white'] = -1
    try:
        config.validate()
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    
    # Reset and test invalid expected count
    config = MOLTTrackerConfig.create_default()
    config.expected_ball_counts['white'] = -1
    try:
        config.validate()
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_config_to_dict() -> None:
    """Test converting configuration to dictionary."""
    config = MOLTTrackerConfig.create_default()
    config_dict = config.to_dict()
    
    # Check that all expected keys are present
    expected_keys = [
        'population_sizes', 'exploration_radii', 'expected_ball_counts',
        'histogram_bins', 'color_space', 'similarity_weights',
        'diversity_distribution', 'min_confidence', 'max_frames_without_detection'
    ]
    
    for key in expected_keys:
        assert key in config_dict


if __name__ == "__main__":
    test_default_config_creation()
    test_snooker_config_creation()
    test_pool_config_creation()
    test_config_validation()
    test_config_to_dict()
    print("✓ All config tests passed!")