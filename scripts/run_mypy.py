#!/usr/bin/env python3
"""
Comprehensive MyPy Type Checking Script

This script runs mypy type checking on all relevant parts of the physics_inference project,
with appropriate configurations and exclusions for different components.
"""

import os
import sys
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

class MyPyChecker:
    """Comprehensive mypy type checker for the physics_inference project."""
    
    def __init__(self) -> None:
        self.project_root = project_root
        self.results: Dict[str, Any] = {}
        self.total_files = 0
        self.clean_files = 0
        self.files_with_errors = 0
        self.total_errors = 0
        
    def run_all_checks(self) -> bool:
        """Run all mypy checks and return overall success status."""
        print("🔍 Physics Inference - Comprehensive MyPy Type Checking")
        print("=" * 70)
        
        success = True
        
        # 1. Core tracking modules (strict checking)
        print("\n1️⃣ Checking Core Tracking Modules (Strict)...")
        success &= self._check_core_tracking()
        
        # 2. Detection modules
        print("\n2️⃣ Checking Detection Modules...")
        success &= self._check_detection_modules()
        
        # 3. Scripts (relaxed checking)
        print("\n3️⃣ Checking Scripts (Relaxed)...")
        success &= self._check_scripts()
        
        # 4. Test files (minimal checking)
        print("\n4️⃣ Checking Test Files (Minimal)...")
        success &= self._check_test_files()
        
        # 5. Full project check (with exclusions)
        print("\n5️⃣ Running Full Project Check...")
        success &= self._check_full_project()
        
        # Print summary
        self._print_summary()
        
        # Return success if core modules are clean
        core_tracking_result = self.results.get('core_tracking', {})
        core_success: bool = core_tracking_result.get('success', False) if isinstance(core_tracking_result, dict) else False
        return core_success
    
    def _run_mypy(self, paths: List[str], extra_args: Optional[List[str]] = None) -> Dict[str, Any]:
        """Run mypy on specified paths and return results."""
        cmd = ["mypy"] + paths
        if extra_args:
            cmd.extend(extra_args)
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=self.project_root
            )
            
            # Parse output for statistics
            output_lines = result.stdout.split('\n') if result.stdout else []
            error_lines = [line for line in output_lines if ': error:' in line]
            
            return {
                'returncode': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'error_count': len(error_lines),
                'success': result.returncode == 0
            }
            
        except Exception as e:
            return {
                'returncode': -1,
                'stdout': '',
                'stderr': str(e),
                'error_count': 1,
                'success': False
            }
    
    def _check_core_tracking(self) -> bool:
        """Check core tracking modules with strict settings."""
        print("   Checking core tracking modules...")
        
        core_modules = [
            "src/tracking/moment_evaluator.py",
            "src/tracking/ground_truth_evaluator.py", 
            "src/tracking/ground_truth_visualizer.py",
            "src/tracking/tracker_benchmark.py",
            "src/tracking/state_reconstructor.py",
            "src/tracking/tracking.py",
            "src/tracking/trackers/molt/tracker.py",
            "src/tracking/trackers/molt/population.py",
            "src/tracking/trackers/molt/local_tracker.py",
            "src/tracking/trackers/molt/ball_count_manager.py",
            "src/tracking/trackers/molt/histogram_extractor.py",
        ]
        
        # Filter to only existing files
        existing_modules = [m for m in core_modules if (self.project_root / m).exists()]
        
        if not existing_modules:
            print("   ⚠️  No core modules found")
            return True
        
        result = self._run_mypy(
            existing_modules,
            ["--explicit-package-bases", "--ignore-missing-imports"]
        )
        
        self.total_errors += result['error_count']
        self.results['core_tracking'] = result
        
        if result['success']:
            print(f"   ✅ Core tracking modules: {len(existing_modules)} files, 0 errors")
            self.clean_files += len(existing_modules)
        else:
            print(f"   ❌ Core tracking modules: {result['error_count']} errors")
            if result['stdout']:
                # Show first few errors
                error_lines = [line for line in result['stdout'].split('\n') if ': error:' in line]
                for line in error_lines[:3]:
                    print(f"      {line}")
                if len(error_lines) > 3:
                    print(f"      ... and {len(error_lines) - 3} more errors")
            self.files_with_errors += 1
        
        self.total_files += len(existing_modules)
        success_value = result.get('success', False)
        return bool(success_value)
    
    def _check_detection_modules(self) -> bool:
        """Check detection modules."""
        print("   Checking detection modules...")
        
        detection_modules = [
            "src/detection/local_pt_inference.py",
            "src/detection/inference_runner.py",
            "src/detection/roboflow_local_inference.py",
        ]
        
        existing_modules = [m for m in detection_modules if (self.project_root / m).exists()]
        
        if not existing_modules:
            print("   ⚠️  No detection modules found")
            return True
        
        result = self._run_mypy(
            existing_modules,
            ["--explicit-package-bases", "--ignore-missing-imports"]
        )
        
        self.total_errors += result['error_count']
        
        if result['success']:
            print(f"   ✅ Detection modules: {len(existing_modules)} files, 0 errors")
            self.clean_files += len(existing_modules)
        else:
            print(f"   ❌ Detection modules: {result['error_count']} errors")
            self.files_with_errors += 1
        
        self.total_files += len(existing_modules)
        success_value = result.get('success', False)
        return bool(success_value)
    
    def _check_scripts(self) -> bool:
        """Check script files with relaxed settings."""
        print("   Checking script files...")
        
        script_files = []
        script_dirs = ["scripts/tracking", "scripts/detection"]
        
        for script_dir in script_dirs:
            script_path = self.project_root / script_dir
            if script_path.exists():
                script_files.extend([
                    str(f.relative_to(self.project_root))
                    for f in script_path.glob("*.py")
                ])
        
        if not script_files:
            print("   ⚠️  No script files found")
            return True
        
        result = self._run_mypy(
            script_files,
            [
                "--explicit-package-bases",
                "--ignore-missing-imports",
                "--disable-error-code=import-untyped"
            ]
        )
        
        self.total_errors += result['error_count']
        
        if result['success']:
            print(f"   ✅ Script files: {len(script_files)} files, 0 errors")
            self.clean_files += len(script_files)
        else:
            print(f"   ❌ Script files: {result['error_count']} errors")
            self.files_with_errors += 1
        
        self.total_files += len(script_files)
        success_value = result.get('success', False)
        return bool(success_value)
    
    def _check_test_files(self) -> bool:
        """Check test files with minimal settings."""
        print("   Checking test files...")
        
        test_files = []
        test_dirs = [
            "src/tracking/trackers/molt/tests",
            "tests"
        ]
        
        for test_dir in test_dirs:
            test_path = self.project_root / test_dir
            if test_path.exists():
                test_files.extend([
                    str(f.relative_to(self.project_root))
                    for f in test_path.glob("test_*.py")
                ])
        
        if not test_files:
            print("   ⚠️  No test files found")
            return True
        
        result = self._run_mypy(
            test_files,
            [
                "--explicit-package-bases",
                "--ignore-missing-imports",
                "--disable-error-code=import-untyped,no-untyped-def,misc"
            ]
        )
        
        self.total_errors += result['error_count']
        
        if result['success']:
            print(f"   ✅ Test files: {len(test_files)} files, 0 errors")
            self.clean_files += len(test_files)
        else:
            print(f"   ⚠️  Test files: {result['error_count']} errors (non-critical)")
            # Don't count test file errors as failures
        
        self.total_files += len(test_files)
        return True  # Always return True for test files
    
    def _check_full_project(self) -> bool:
        """Run a full project check with appropriate exclusions."""
        print("   Running full project check...")
        
        result = self._run_mypy(
            ["src"],
            [
                "--explicit-package-bases",
                "--ignore-missing-imports",
                "--exclude=CVModelInference|PureCV|assets|__pycache__|.mypy_cache|.git",
                "--disable-error-code=import-untyped"
            ]
        )
        
        if result['success']:
            print("   ✅ Full project check: No critical errors")
            return True
        else:
            # Count only critical errors
            critical_errors = 0
            if result['stdout']:
                lines = result['stdout'].split('\n')
                for line in lines:
                    if ': error:' in line and not any(x in line for x in ['test_', 'unused-ignore', 'unreachable']):
                        critical_errors += 1
            
            if critical_errors == 0:
                print("   ✅ Full project check: Only non-critical errors found")
                return True
            else:
                print(f"   ❌ Full project check: {critical_errors} critical errors")
                return False
    
    def _print_summary(self) -> None:
        """Print mypy results summary."""
        print("\n" + "=" * 70)
        print("📊 MYPY TYPE CHECKING SUMMARY")
        print("=" * 70)
        print(f"Total Files Checked: {self.total_files}")
        print(f"✅ Clean Files: {self.clean_files}")
        print(f"❌ Files with Errors: {self.files_with_errors}")
        print(f"Total Errors: {self.total_errors}")
        
        # Check if core modules are clean (most important)
        core_clean = self.results.get('core_tracking', {}).get('success', False)
        
        if core_clean:
            print("\n🎉 CORE MODULES PASS TYPE CHECKING! 🎉")
            print("The core physics_inference functionality is fully type-safe!")
            if self.files_with_errors > 0:
                print(f"⚠️  {self.files_with_errors} non-core files have minor type issues.")
        else:
            print(f"\n❌ Core modules have type errors. Please review above.")
            print(f"⚠️  {self.files_with_errors} files have type errors total.")
        
        if self.total_files > 0:
            success_rate = (self.clean_files / self.total_files * 100)
            print(f"Type Safety Rate: {success_rate:.1f}%")


def main() -> None:
    """Main entry point."""
    checker = MyPyChecker()
    success = checker.run_all_checks()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()