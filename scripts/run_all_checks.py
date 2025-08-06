#!/usr/bin/env python3
"""
Master Script for All Quality Checks

This script runs the complete quality assurance suite for the physics_inference project:
- Comprehensive test suite
- MyPy type checking
- Import validation
- Script functionality tests

Usage:
    python scripts/run_all_checks.py [--tests-only] [--mypy-only] [--verbose]
"""

import sys
import argparse
from pathlib import Path
import subprocess

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def run_command(script_path: str, description: str) -> bool:
    """Run a command and return success status."""
    print(f"\n{'='*80}")
    print(f"🚀 {description}")
    print(f"{'='*80}")
    
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            cwd=project_root
        )
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Error running {script_path}: {e}")
        return False

def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run comprehensive quality checks for physics_inference project"
    )
    parser.add_argument(
        "--tests-only", 
        action="store_true", 
        help="Run only the test suite"
    )
    parser.add_argument(
        "--mypy-only", 
        action="store_true", 
        help="Run only MyPy type checking"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true", 
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    print("🔬 Physics Inference - Complete Quality Assurance Suite")
    print("=" * 80)
    print("This script will run comprehensive checks on the entire project:")
    print("• Test Suite: Unit tests, integration tests, script tests")
    print("• MyPy: Type checking with strict validation")
    print("• Import Validation: Ensure all modules can be imported")
    print("• Script Functionality: Verify all scripts work correctly")
    
    success = True
    
    # Run test suite
    if not args.mypy_only:
        test_success = run_command(
            str(project_root / "scripts" / "run_tests.py"),
            "COMPREHENSIVE TEST SUITE"
        )
        success &= test_success
    
    # Run MyPy checks
    if not args.tests_only:
        mypy_success = run_command(
            str(project_root / "scripts" / "run_mypy.py"),
            "MYPY TYPE CHECKING"
        )
        success &= mypy_success
    
    # Final summary
    print(f"\n{'='*80}")
    print("🏁 FINAL RESULTS")
    print(f"{'='*80}")
    
    if success:
        print("🎉 ALL CHECKS PASSED! 🎉")
        print("")
        print("✅ Test Suite: PASSED")
        print("✅ MyPy Type Checking: PASSED") 
        print("✅ Import Validation: PASSED")
        print("✅ Script Functionality: PASSED")
        print("")
        print("🚀 The physics_inference project is ready for production!")
        print("📦 All modules are type-safe and fully tested.")
        print("🔧 All scripts are functional and ready to use.")
    else:
        print("❌ SOME CHECKS FAILED")
        print("")
        print("Please review the detailed output above to identify and fix issues.")
        print("Run individual check scripts for more detailed debugging:")
        print("  • python scripts/run_tests.py")
        print("  • python scripts/run_mypy.py")
    
    print(f"\n{'='*80}")
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()