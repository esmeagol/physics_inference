"""
Unit tests for BallCountManager class.

This module contains comprehensive tests for the BallCountManager functionality
including count tracking, verification, and correction logic.
"""

import unittest
import numpy as np
from typing import Dict, List, Any

# Import the BallCountManager class
import sys
import os

from tracking.trackers.molt.ball_count_manager import BallCountManager


class TestBallCountManager(unittest.TestCase):
    """Test cases for BallCountManager class."""
    
    def setUp(self) -> None:
        """Set up test fixtures before each test method."""
        self.default_expected_counts = {
            'white': 1,
            'red': 15,
            'yellow': 1,
            'green': 1,
            'brown': 1,
            'blue': 1,
            'pink': 1,
            'black': 1
        }
        self.manager = BallCountManager(self.default_expected_counts)
    
    def test_initialization_valid(self) -> None:
        """Test valid initialization of BallCountManager."""
        # Test with valid expected counts
        expected_counts = {'red': 5, 'white': 1, 'yellow': 1}
        manager = BallCountManager(expected_counts)
        
        self.assertEqual(manager.expected_counts, expected_counts)
        self.assertEqual(manager.current_counts, {'red': 0, 'white': 0, 'yellow': 0})
        self.assertEqual(manager.track_assignments, {})
        self.assertEqual(manager.total_count_violations, 0)
    
    def test_initialization_invalid(self) -> None:
        """Test invalid initialization parameters."""
        # Test with empty expected counts
        with self.assertRaises(ValueError):
            BallCountManager({})
        
        # Test with invalid ball class (empty string)
        with self.assertRaises(ValueError):
            BallCountManager({'': 1, 'red': 5})
        
        # Test with invalid count (negative)
        with self.assertRaises(ValueError):
            BallCountManager({'red': -1, 'white': 1})
        
        # Test with invalid count (non-integer)
        with self.assertRaises(ValueError):
            BallCountManager({'red': 1.5, 'white': 1})
    
    def test_update_counts_from_tracks(self) -> None:
        """Test updating counts from tracking results."""
        # Create sample tracks
        tracks = [
            {'id': 1, 'class': 'red', 'x': 100, 'y': 100},
            {'id': 2, 'class': 'red', 'x': 200, 'y': 200},
            {'id': 3, 'class': 'white', 'x': 300, 'y': 300},
            {'id': 4, 'class': 'yellow', 'x': 400, 'y': 400}
        ]
        
        self.manager.update_counts_from_tracks(tracks)
        
        # Check updated counts
        expected_current_counts = {
            'white': 1,
            'red': 2,
            'yellow': 1,
            'green': 0,
            'brown': 0,
            'blue': 0,
            'pink': 0,
            'black': 0
        }
        self.assertEqual(self.manager.current_counts, expected_current_counts)
        
        # Check track assignments
        expected_assignments = {1: 'red', 2: 'red', 3: 'white', 4: 'yellow'}
        self.assertEqual(self.manager.track_assignments, expected_assignments)
    
    def test_update_counts_with_unknown_class(self) -> None:
        """Test updating counts with unknown ball class."""
        tracks = [
            {'id': 1, 'class': 'red', 'x': 100, 'y': 100},
            {'id': 2, 'class': 'unknown', 'x': 200, 'y': 200}
        ]
        
        self.manager.update_counts_from_tracks(tracks)
        
        # Unknown class should be added to current counts
        self.assertEqual(self.manager.current_counts['red'], 1)
        self.assertEqual(self.manager.current_counts['unknown'], 1)
        self.assertEqual(self.manager.track_assignments[2], 'unknown')
    
    def test_verify_counts_valid(self) -> None:
        """Test count verification with valid counts."""
        # Set up tracks that match expected counts
        tracks = [
            {'id': i, 'class': 'red', 'x': i*10, 'y': i*10} 
            for i in range(1, 16)  # 15 red balls
        ]
        tracks.append({'id': 16, 'class': 'white', 'x': 160, 'y': 160})  # 1 white ball
        tracks.append({'id': 17, 'class': 'yellow', 'x': 170, 'y': 170})  # 1 yellow ball
        tracks.append({'id': 18, 'class': 'green', 'x': 180, 'y': 180})  # 1 green ball
        tracks.append({'id': 19, 'class': 'brown', 'x': 190, 'y': 190})  # 1 brown ball
        tracks.append({'id': 20, 'class': 'blue', 'x': 200, 'y': 200})  # 1 blue ball
        tracks.append({'id': 21, 'class': 'pink', 'x': 210, 'y': 210})  # 1 pink ball
        tracks.append({'id': 22, 'class': 'black', 'x': 220, 'y': 220})  # 1 black ball
        
        self.manager.update_counts_from_tracks(tracks)
        result = self.manager.verify_counts()
        
        self.assertTrue(result)
        self.assertEqual(self.manager.total_count_violations, 0)
    
    def test_verify_counts_violations(self) -> None:
        """Test count verification with violations."""
        # Set up tracks with count violations
        tracks = [
            {'id': 1, 'class': 'red', 'x': 100, 'y': 100},  # Only 1 red (expected 15)
            {'id': 2, 'class': 'white', 'x': 200, 'y': 200},
            {'id': 3, 'class': 'white', 'x': 300, 'y': 300}  # 2 white (expected 1)
        ]
        
        self.manager.update_counts_from_tracks(tracks)
        result = self.manager.verify_counts()
        
        self.assertFalse(result)
        self.assertEqual(self.manager.total_count_violations, 1)
        self.assertTrue(len(self.manager.violation_history) > 0)
    
    def test_get_count_violations(self) -> None:
        """Test getting detailed count violation information."""
        tracks = [
            {'id': 1, 'class': 'red', 'x': 100, 'y': 100},  # 1 red (expected 15)
            {'id': 2, 'class': 'white', 'x': 200, 'y': 200},
            {'id': 3, 'class': 'white', 'x': 300, 'y': 300}  # 2 white (expected 1)
        ]
        
        self.manager.update_counts_from_tracks(tracks)
        violations = self.manager.get_count_violations()
        
        # Check red ball violation (under count)
        self.assertEqual(violations['red']['expected'], 15)
        self.assertEqual(violations['red']['current'], 1)
        self.assertEqual(violations['red']['difference'], -14)
        self.assertEqual(violations['red']['violation_type'], 'under_count')
        
        # Check white ball violation (over count)
        self.assertEqual(violations['white']['expected'], 1)
        self.assertEqual(violations['white']['current'], 2)
        self.assertEqual(violations['white']['difference'], 1)
        self.assertEqual(violations['white']['violation_type'], 'over_count')
        
        # Check yellow ball (no violation)
        self.assertEqual(violations['yellow']['expected'], 1)
        self.assertEqual(violations['yellow']['current'], 0)
        self.assertEqual(violations['yellow']['difference'], -1)
        self.assertEqual(violations['yellow']['violation_type'], 'under_count')
    
    def test_handle_lost_ball(self) -> None:
        """Test handling lost ball scenarios."""
        # Set up scenario with missing red balls
        tracks = [
            {'id': 1, 'class': 'red', 'x': 100, 'y': 100},  # Only 1 red (expected 15)
            {'id': 2, 'class': 'white', 'x': 200, 'y': 200}
        ]
        
        self.manager.update_counts_from_tracks(tracks)
        
        # Handle lost red ball
        result = self.manager.handle_lost_ball('red')
        
        # For now, the method returns None (placeholder implementation)
        self.assertIsNone(result)
        self.assertEqual(self.manager.lost_ball_recoveries, 1)
        
        # Test with ball class that's not actually lost
        result = self.manager.handle_lost_ball('white')
        self.assertIsNone(result)
        
        # Test with unknown ball class
        result = self.manager.handle_lost_ball('unknown')
        self.assertIsNone(result)
    
    def test_handle_duplicate_ball(self) -> None:
        """Test handling duplicate ball scenarios."""
        # Set up scenario with duplicate white balls
        tracks = [
            {'id': 1, 'class': 'white', 'x': 100, 'y': 100},
            {'id': 2, 'class': 'white', 'x': 200, 'y': 200},
            {'id': 3, 'class': 'white', 'x': 300, 'y': 300}  # 3 white (expected 1)
        ]
        
        self.manager.update_counts_from_tracks(tracks)
        
        # Get track IDs for white balls
        white_track_ids = [1, 2, 3]
        
        # Handle duplicate white balls
        excess_tracks = self.manager.handle_duplicate_ball('white', white_track_ids)
        
        # Should return 2 excess tracks (keep only 1)
        self.assertEqual(len(excess_tracks), 2)
        self.assertEqual(excess_tracks, [2, 3])  # Keep first track, return others
        self.assertEqual(self.manager.duplicate_ball_merges, 1)
        
        # Test with ball class that doesn't have duplicates
        red_track_ids = [4]
        excess_tracks = self.manager.handle_duplicate_ball('red', red_track_ids)
        self.assertEqual(len(excess_tracks), 0)
        
        # Test with unknown ball class
        excess_tracks = self.manager.handle_duplicate_ball('unknown', [5, 6])
        self.assertEqual(len(excess_tracks), 0)
    
    def test_suggest_track_merges(self) -> None:
        """Test suggesting track merges for count violations."""
        # Set up scenario with duplicate balls at different positions
        tracks = [
            {'id': 1, 'class': 'white', 'x': 100, 'y': 100},
            {'id': 2, 'class': 'white', 'x': 105, 'y': 105},  # Close to track 1
            {'id': 3, 'class': 'white', 'x': 200, 'y': 200},  # Far from others
            {'id': 4, 'class': 'red', 'x': 300, 'y': 300},
            {'id': 5, 'class': 'red', 'x': 310, 'y': 310}     # Close to track 4
        ]
        
        self.manager.update_counts_from_tracks(tracks)
        merge_suggestions = self.manager.suggest_track_merges(tracks)
        
        # Should suggest merging closest pairs
        self.assertTrue(len(merge_suggestions) > 0)
        
        # Check that suggested merges involve tracks of the same class
        for track_id1, track_id2 in merge_suggestions:
            class1 = self.manager.track_assignments.get(track_id1)
            class2 = self.manager.track_assignments.get(track_id2)
            self.assertEqual(class1, class2)
    
    def test_get_statistics(self) -> None:
        """Test getting ball count manager statistics."""
        # Set up some tracking data
        tracks = [
            {'id': 1, 'class': 'red', 'x': 100, 'y': 100},
            {'id': 2, 'class': 'white', 'x': 200, 'y': 200}
        ]
        
        self.manager.update_counts_from_tracks(tracks)
        self.manager.verify_counts()  # This will create violations
        
        stats = self.manager.get_statistics()
        
        # Check that all expected fields are present
        self.assertIn('expected_counts', stats)
        self.assertIn('current_counts', stats)
        self.assertIn('track_assignments', stats)
        self.assertIn('total_count_violations', stats)
        self.assertIn('lost_ball_recoveries', stats)
        self.assertIn('duplicate_ball_merges', stats)
        self.assertIn('recent_violations', stats)
        
        # Check values
        self.assertEqual(stats['expected_counts'], self.default_expected_counts)
        self.assertEqual(stats['current_counts']['red'], 1)
        self.assertEqual(stats['current_counts']['white'], 1)
        self.assertEqual(stats['track_assignments'], {1: 'red', 2: 'white'})
        self.assertGreater(stats['total_count_violations'], 0)
    
    def test_reset(self) -> None:
        """Test resetting the ball count manager."""
        # Set up some data
        tracks = [
            {'id': 1, 'class': 'red', 'x': 100, 'y': 100},
            {'id': 2, 'class': 'white', 'x': 200, 'y': 200}
        ]
        
        self.manager.update_counts_from_tracks(tracks)
        self.manager.verify_counts()
        self.manager.handle_lost_ball('red')
        self.manager.handle_duplicate_ball('white', [1, 2])
        
        # Reset
        self.manager.reset()
        
        # Check that everything is reset
        expected_zero_counts = {ball_class: 0 for ball_class in self.default_expected_counts.keys()}
        self.assertEqual(self.manager.current_counts, expected_zero_counts)
        self.assertEqual(self.manager.track_assignments, {})
        self.assertEqual(self.manager.total_count_violations, 0)
        self.assertEqual(self.manager.lost_ball_recoveries, 0)
        self.assertEqual(self.manager.duplicate_ball_merges, 0)
        self.assertEqual(self.manager.violation_history, [])
    
    def test_update_expected_counts(self) -> None:
        """Test updating expected counts."""
        new_expected_counts = {
            'white': 1,
            'red': 10,  # Changed from 15 to 10
            'yellow': 1,
            'blue': 2   # Changed from 1 to 2
        }
        
        self.manager.update_expected_counts(new_expected_counts)
        
        # Check that expected counts are updated
        self.assertEqual(self.manager.expected_counts, new_expected_counts)
        
        # Check that current counts are updated for new ball classes
        self.assertIn('blue', self.manager.current_counts)
        self.assertEqual(self.manager.current_counts['blue'], 0)
        
        # Check that old ball classes are removed from current counts
        self.assertNotIn('green', self.manager.current_counts)
        self.assertNotIn('brown', self.manager.current_counts)
        self.assertNotIn('pink', self.manager.current_counts)
        self.assertNotIn('black', self.manager.current_counts)
    
    def test_update_expected_counts_invalid(self) -> None:
        """Test updating expected counts with invalid data."""
        # Test with empty counts
        with self.assertRaises(ValueError):
            self.manager.update_expected_counts({})
        
        # Test with invalid ball class
        with self.assertRaises(ValueError):
            self.manager.update_expected_counts({'': 1, 'red': 5})
        
        # Test with invalid count
        with self.assertRaises(ValueError):
            self.manager.update_expected_counts({'red': -1, 'white': 1})
    
    def test_string_representations(self) -> None:
        """Test string representations of the manager."""
        # Test __str__
        str_repr = str(self.manager)
        self.assertIn('BallCountManager', str_repr)
        self.assertIn('violations=', str_repr)
        self.assertIn('total_tracks=', str_repr)
        
        # Test __repr__
        repr_str = repr(self.manager)
        self.assertIn('BallCountManager', repr_str)
        self.assertIn('expected=', repr_str)
        self.assertIn('current=', repr_str)
        self.assertIn('violations=', repr_str)


if __name__ == '__main__':
    unittest.main()