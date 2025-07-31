#!/usr/bin/env python3
"""
Comprehensive test runner for the MOLT tracker system.

This script runs all tests to ensure the reorganized code works correctly.
"""

import sys
import os
import subprocess
import time

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def run_test(test_file, description):
    """Run a single test file and report results."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"File: {test_file}")
    print('='*60)
    
    start_time = time.time()
    
    try:
        result = subprocess.run([sys.executable, test_file], 
                              capture_output=True, text=True, timeout=30)
        
        end_time = time.time()
        duration = end_time - start_time
        
        if result.returncode == 0:
            print(f"✅ PASSED ({duration:.2f}s)")
            if result.stdout.strip():
                print("Output:")
                print(result.stdout)
        else:
            print(f"❌ FAILED ({duration:.2f}s)")
            print("STDOUT:")
            print(result.stdout)
            print("STDERR:")
            print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ TIMEOUT (30s)")
        return False
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False
    
    return True

def main():
    """Run all tests."""
    print("🧪 MOLT Tracker Comprehensive Test Suite")
    print("=" * 60)
    
    # List of all tests to run
    tests = [
        ("PureCV/tests/test_config.py", "Configuration System Tests"),
        ("PureCV/tests/test_histogram_extractor.py", "Histogram Extractor Tests"),
        ("PureCV/test_ball_count_manager.py", "Ball Count Manager Tests"),
        ("PureCV/test_task_4_3.py", "Population Regeneration Tests"),
        ("PureCV/test_molt_tracker_init.py", "MOLT Tracker Initialization Tests"),
        ("PureCV/tests/test_integration.py", "Integration Tests"),
        ("PureCV/examples/basic_usage.py", "Usage Examples"),
    ]
    
    passed = 0
    failed = 0
    start_time = time.time()
    
    for test_file, description in tests:
        if os.path.exists(test_file):
            if run_test(test_file, description):
                passed += 1
            else:
                failed += 1
        else:
            print(f"\n❌ Test file not found: {test_file}")
            failed += 1
    
    end_time = time.time()
    total_duration = end_time - start_time
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 TEST SUMMARY")
    print('='*60)
    print(f"Total tests: {passed + failed}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"⏱️  Total time: {total_duration:.2f}s")
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ The reorganized MOLT tracker is working correctly!")
        return 0
    else:
        print(f"\n💥 {failed} TEST(S) FAILED!")
        print("❌ Please fix the failing tests before proceeding.")
        return 1

if __name__ == "__main__":
    sys.exit(main())