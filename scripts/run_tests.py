#!/usr/bin/env python3
"""
Comprehensive Test Suite Runner

This script runs all tests in the physics_inference project, including:
- Unit tests for tracking modules
- Integration tests
- Script functionality tests
- Import validation tests
"""

import os
import sys
import subprocess
import unittest
from pathlib import Path
from typing import List, Dict, Any
import importlib.util

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

class TestSuiteRunner:
    """Comprehensive test suite runner for the physics_inference project."""
    
    def __init__(self) -> None:
        self.project_root = project_root
        self.results: Dict[str, Any] = {}
        self.total_tests = 0
        self.passed_tests = 0
        self.failed_tests = 0
        
    def run_all_tests(self) -> bool:
        """Run all test suites and return overall success status."""
        print("🧪 Physics Inference - Comprehensive Test Suite")
        print("=" * 60)
        
        success = True
        
        # 1. Import validation tests
        print("\n1️⃣ Running Import Validation Tests...")
        success &= self._run_import_tests()
        
        # 2. Unit tests
        print("\n2️⃣ Running Unit Tests...")
        success &= self._run_unit_tests()
        
        # 3. Script functionality tests
        print("\n3️⃣ Running Script Functionality Tests...")
        success &= self._run_script_tests()
        
        # 4. Integration tests
        print("\n4️⃣ Running Integration Tests...")
        success &= self._run_integration_tests()
        
        # Print summary
        self._print_summary()
        
        return success
    
    def _run_import_tests(self) -> bool:
        """Test that all critical modules can be imported."""
        print("   Testing critical module imports...")
        
        critical_imports = [
            "src.tracking.tracker_benchmark.SnookerTrackerBenchmark",
            "src.tracking.trackers.molt.MOLTTracker",
            "src.tracking.moment_evaluator.MomentEvaluator",
            "src.tracking.ground_truth_evaluator.GroundTruthEvaluator",
            "src.tracking.ground_truth_visualizer.GroundTruthVisualizer",
            "src.tracking.state_reconstructor.StateReconstructor",
            "src.tracking.tracking.Tracker",
            "src.detection.local_pt_inference.LocalPT",
            "src.detection.inference_runner.InferenceRunner",
        ]
        
        success = True
        for import_path in critical_imports:
            try:
                module_path, class_name = import_path.rsplit(".", 1)
                module = importlib.import_module(module_path)
                getattr(module, class_name)
                print(f"   ✅ {import_path}")
                self.passed_tests += 1
            except Exception as e:
                print(f"   ❌ {import_path}: {e}")
                success = False
                self.failed_tests += 1
            self.total_tests += 1
        
        return success
    
    def _run_unit_tests(self) -> bool:
        """Run unittest discovery for all test files."""
        print("   Discovering and running unit tests...")
        
        # Use the main tests directory which now contains all tests
        test_dirs = [
            self.project_root / "tests",
        ]
        
        success = True
        for test_dir in test_dirs:
            if test_dir.exists():
                try:
                    # Discover and run tests
                    loader = unittest.TestLoader()
                    suite = loader.discover(str(test_dir), pattern="test_*.py")
                    
                    # Use a StringIO to capture the test output
                    from io import StringIO
                    output = StringIO()
                    
                    # Run tests with output captured
                    runner = unittest.TextTestRunner(verbosity=2, stream=output)
                    result = runner.run(suite)
                    
                    tests_run = result.testsRun
                    failures = len(result.failures)
                    errors = len(result.errors)
                    
                    self.total_tests += tests_run
                    self.passed_tests += (tests_run - failures - errors)
                    self.failed_tests += (failures + errors)
                    
                    if failures == 0 and errors == 0:
                        print(f"   ✅ {test_dir.name}: {tests_run} tests passed")
                    else:
                        print(f"   ❌ {test_dir.name}: {failures} failures, {errors} errors out of {tests_run} tests")
                        # Print the test output to show which tests failed
                        print("\n" + "="*50)
                        print(f"Test output for {test_dir.name}:")
                        print("-"*50)
                        print(output.getvalue())
                        print("="*50 + "\n")
                        success = False
                        
                except Exception as e:
                    print(f"   ❌ Error running tests in {test_dir}: {e}")
                    success = False
        
        return success
    
    def _run_script_tests(self) -> bool:
        """Test that all scripts can be executed without errors."""
        print("   Testing script functionality...")
        
        scripts_to_test = [
            ("scripts/tracking/compare_trackers.py", ["--help"]),
            ("scripts/tracking/track_objects.py", ["--help"]),
            ("scripts/tracking/molt_basic_usage.py", []),
            ("scripts/detection/compare_local_models.py", ["--help"]),
        ]
        
        success = True
        for script_path, args in scripts_to_test:
            try:
                full_path = self.project_root / script_path
                if full_path.exists():
                    result = subprocess.run(
                        [sys.executable, str(full_path)] + args,
                        capture_output=True,
                        text=True,
                        timeout=30
                    )
                    
                    if result.returncode == 0:
                        print(f"   ✅ {script_path}")
                        self.passed_tests += 1
                    else:
                        print(f"   ❌ {script_path}: Exit code {result.returncode}")
                        if result.stderr:
                            print(f"      Error: {result.stderr[:200]}...")
                        success = False
                        self.failed_tests += 1
                else:
                    print(f"   ⚠️  {script_path}: File not found")
                    
                self.total_tests += 1
                    
            except subprocess.TimeoutExpired:
                print(f"   ❌ {script_path}: Timeout")
                success = False
                self.failed_tests += 1
                self.total_tests += 1
            except Exception as e:
                print(f"   ❌ {script_path}: {e}")
                success = False
                self.failed_tests += 1
                self.total_tests += 1
        
        return success
    
    def _run_integration_tests(self) -> bool:
        """Run integration tests."""
        print("   Running integration tests...")
        
        # For now, just test that we can create and use key objects
        try:
            from src.tracking.trackers.molt import MOLTTracker, MOLTTrackerConfig
            from src.tracking.tracker_benchmark import SnookerTrackerBenchmark
            
            # Test MOLT tracker creation
            config = MOLTTrackerConfig()
            tracker = MOLTTracker(config)
            
            # Test benchmark creation
            benchmark = SnookerTrackerBenchmark()
            
            print("   ✅ Integration tests passed")
            self.passed_tests += 1
            self.total_tests += 1
            return True
            
        except Exception as e:
            print(f"   ❌ Integration tests failed: {e}")
            self.failed_tests += 1
            self.total_tests += 1
            return False
    
    def _print_summary(self) -> None:
        """Print test results summary."""
        print("\n" + "=" * 60)
        print("📊 TEST RESULTS SUMMARY")
        print("=" * 60)
        print(f"Total Tests: {self.total_tests}")
        print(f"✅ Passed: {self.passed_tests}")
        print(f"❌ Failed: {self.failed_tests}")
        
        if self.failed_tests == 0:
            print("\n🎉 ALL TESTS PASSED! 🎉")
            print("The physics_inference project is ready for use!")
        else:
            print(f"\n⚠️  {self.failed_tests} tests failed. Please review the errors above.")
        
        success_rate = (self.passed_tests / self.total_tests * 100) if self.total_tests > 0 else 0
        print(f"Success Rate: {success_rate:.1f}%")


def main() -> None:
    """Main entry point."""
    runner = TestSuiteRunner()
    success = runner.run_all_tests()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()