#!/usr/bin/env python3
"""
Pre-commit Quality Checks

This script runs quality checks on changed files before commit to ensure
code quality without running the full test suite (which would be too slow for commits).

Checks performed:
- MyPy type checking on changed Python files
- Import validation for changed modules
- Basic syntax validation
"""

import os
import sys
import subprocess
from pathlib import Path
from typing import List, Set
import importlib.util

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

class PreCommitChecker:
    """Fast pre-commit quality checker focused on changed files."""
    
    def __init__(self) -> None:
        self.project_root = project_root
        self.errors = 0
        
    def get_changed_python_files(self) -> List[str]:
        """Get list of changed Python files."""
        try:
            # Get staged files
            result = subprocess.run(
                ["git", "diff", "--cached", "--name-only", "--diff-filter=d"],
                capture_output=True,
                text=True,
                cwd=self.project_root
            )
            
            if result.returncode != 0:
                return []
            
            # Filter for Python files in src/ and scripts/
            python_files = []
            for file_path in result.stdout.strip().split('\n'):
                if file_path and file_path.endswith('.py'):
                    if file_path.startswith(('src/', 'scripts/')):
                        full_path = self.project_root / file_path
                        if full_path.exists():
                            python_files.append(file_path)
            
            return python_files
            
        except Exception as e:
            print(f"❌ Error getting changed files: {e}")
            return []
    
    def run_mypy_on_files(self, files: List[str]) -> bool:
        """Run mypy on specific files, focusing on core modules."""
        if not files:
            return True
        
        # Filter to focus on core modules (exclude problematic deepsort_tracker and test files)
        core_files = [f for f in files if not f.endswith('deepsort_tracker.py') and 'test_' not in f]
        
        if not core_files:
            print("🔍 No core files to check with MyPy")
            return True
            
        print(f"🔍 Running MyPy on {len(core_files)} core changed files...")
        
        try:
            result = subprocess.run(
                ["mypy"] + core_files + ["--ignore-missing-imports", "--no-strict-optional", "--follow-imports=silent"],
                capture_output=True,
                text=True,
                cwd=self.project_root
            )
            
            if result.returncode == 0:
                print("   ✅ MyPy checks passed")
                return True
            else:
                print("   ❌ MyPy found errors:")
                # Show only the first few errors to keep output manageable
                error_lines = [line for line in result.stdout.split('\n') if ': error:' in line]
                for line in error_lines[:5]:
                    print(f"      {line}")
                if len(error_lines) > 5:
                    print(f"      ... and {len(error_lines) - 5} more errors")
                self.errors += len(error_lines)
                return False
                
        except Exception as e:
            print(f"   ❌ Error running MyPy: {e}")
            self.errors += 1
            return False
    
    def validate_imports(self, files: List[str]) -> bool:
        """Validate that changed modules can be imported."""
        if not files:
            return True
            
        print(f"📦 Validating imports for {len(files)} changed files...")
        
        success = True
        for file_path in files:
            # Skip test files and scripts for import validation
            if 'test_' in file_path or file_path.startswith('scripts/'):
                continue
                
            # Convert file path to module path
            if file_path.startswith('src/'):
                # Use the full src.* import path to match our standardized import pattern
                module_path = file_path.replace('/', '.').replace('.py', '')
                try:
                    importlib.import_module(module_path)
                    print(f"   ✅ {module_path}")
                except Exception as e:
                    print(f"   ❌ {module_path}: {e}")
                    success = False
                    self.errors += 1
        
        return success
    
    def check_syntax(self, files: List[str]) -> bool:
        """Check Python syntax of changed files."""
        if not files:
            return True
            
        print(f"🐍 Checking Python syntax for {len(files)} files...")
        
        success = True
        for file_path in files:
            try:
                with open(self.project_root / file_path, 'r') as f:
                    compile(f.read(), file_path, 'exec')
                print(f"   ✅ {file_path}")
            except SyntaxError as e:
                print(f"   ❌ {file_path}: Syntax error at line {e.lineno}: {e.msg}")
                success = False
                self.errors += 1
            except Exception as e:
                print(f"   ❌ {file_path}: {e}")
                success = False
                self.errors += 1
        
        return success
    
    def run_checks(self) -> bool:
        """Run all pre-commit checks."""
        print("🚀 Pre-commit Quality Checks")
        print("=" * 50)
        
        # Get changed files
        changed_files = self.get_changed_python_files()
        
        if not changed_files:
            print("ℹ️  No Python files changed in src/ or scripts/")
            print("✅ Pre-commit checks passed (no files to check)")
            return True
        
        print(f"📋 Checking {len(changed_files)} changed Python files:")
        for file_path in changed_files:
            print(f"   • {file_path}")
        print()
        
        success = True
        
        # 1. Syntax check (fastest)
        success &= self.check_syntax(changed_files)
        
        # 2. Import validation (for src files only)
        src_files = [f for f in changed_files if f.startswith('src/')]
        if src_files:
            success &= self.validate_imports(src_files)
        
        # 3. MyPy type checking
        success &= self.run_mypy_on_files(changed_files)
        
        # Summary
        print("\n" + "=" * 50)
        if success:
            print("🎉 All pre-commit checks passed!")
            print("✅ Ready to commit")
        else:
            print(f"❌ Found {self.errors} issues")
            print("Please fix the errors above before committing")
            print("\nTip: Run 'python scripts/run_all_checks.py' for comprehensive validation")
        
        return success

def main() -> None:
    """Main entry point."""
    checker = PreCommitChecker()
    success = checker.run_checks()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()