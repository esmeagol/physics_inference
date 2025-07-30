#!/usr/bin/env python3
"""
Continuous type checking script for PureCV and CVModelInference directories.
This script runs mypy continuously and provides focused feedback on typing errors.
"""

import subprocess
import sys
import time
import os
from pathlib import Path
from typing import Tuple, Dict, List, Set
import argparse

# Core files to prioritize in type checking
CORE_FILES = [
    "CVModelInference/inference_runner.py",
    "CVModelInference/local_pt_inference.py", 
    "CVModelInference/roboflow_local_inference.py",
    "CVModelInference/tracker.py",
    "CVModelInference/tracking.py",
    "PureCV/molt_tracker.py",
    "PureCV/table_detection.py",
]

def run_mypy_check(files: List[str] | None = None, full_check: bool = False) -> Tuple[int, str, str]:
    """Run mypy check and return the result."""
    try:
        if full_check:
            cmd = ["mypy", "CVModelInference", "PureCV", "--config-file", "mypy.ini"]
        elif files:
            cmd = ["mypy"] + files + ["--config-file", "mypy.ini"]
        else:
            cmd = ["mypy"] + CORE_FILES + ["--config-file", "mypy.ini"]
            
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return 1, "", "mypy check timed out"
    except FileNotFoundError:
        return 1, "", "mypy not found. Please install with: pip install mypy"

def get_file_modification_times(directories: List[str] | None = None) -> Dict[str, float]:
    """Get modification times for all Python files in the target directories."""
    if directories is None:
        directories = ["CVModelInference", "PureCV"]
        
    mod_times = {}
    for directory in directories:
        if os.path.exists(directory):
            for root, dirs, files in os.walk(directory):
                # Skip certain directories
                dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'venv']]
                
                for file in files:
                    if file.endswith('.py'):
                        filepath = os.path.join(root, file)
                        mod_times[filepath] = os.path.getmtime(filepath)
    return mod_times

def parse_mypy_output(output: str) -> Dict[str, List[str]]:
    """Parse mypy output and group errors by file."""
    errors_by_file: Dict[str, List[str]] = {}
    if not output.strip():
        return errors_by_file
        
    lines = output.strip().split('\n')
    for line in lines:
        if ': error:' in line:
            parts = line.split(':', 2)
            if len(parts) >= 2:
                filepath = parts[0]
                if filepath not in errors_by_file:
                    errors_by_file[filepath] = []
                errors_by_file[filepath].append(line)
    
    return errors_by_file

def print_summary(errors_by_file: Dict[str, List[str]], show_details: bool = True) -> None:
    """Print a summary of type checking results."""
    if not errors_by_file:
        print("✅ No type errors found!")
        return
        
    total_errors = sum(len(errors) for errors in errors_by_file.values())
    print(f"❌ Found {total_errors} type errors in {len(errors_by_file)} files:")
    
    # Sort files by number of errors (most errors first)
    sorted_files = sorted(errors_by_file.items(), key=lambda x: len(x[1]), reverse=True)
    
    for filepath, errors in sorted_files:
        print(f"\n📁 {filepath} ({len(errors)} errors)")
        if show_details:
            for error in errors[:5]:  # Show first 5 errors per file
                print(f"   {error}")
            if len(errors) > 5:
                print(f"   ... and {len(errors) - 5} more errors")

def main() -> None:
    """Main function to run continuous mypy checking."""
    parser = argparse.ArgumentParser(description="Continuous type checking for PureCV and CVModelInference")
    parser.add_argument("--full", action="store_true", help="Check all files instead of just core files")
    parser.add_argument("--once", action="store_true", help="Run once and exit")
    parser.add_argument("--details", action="store_true", help="Show detailed error messages")
    parser.add_argument("--interval", type=int, default=5, help="Check interval in seconds (default: 5)")
    
    args = parser.parse_args()
    
    if args.once:
        print("Running one-time type check...")
        returncode, stdout, stderr = run_mypy_check(full_check=args.full)
        errors_by_file = parse_mypy_output(stdout)
        print_summary(errors_by_file, show_details=args.details)
        sys.exit(returncode)
    
    print("Starting continuous mypy type checking...")
    if args.full:
        print("Monitoring all Python files in PureCV and CVModelInference directories")
    else:
        print("Monitoring core files:")
        for file in CORE_FILES:
            print(f"  - {file}")
    print(f"Check interval: {args.interval} seconds")
    print("Press Ctrl+C to stop")
    
    last_mod_times = get_file_modification_times()
    last_check_time = 0.0
    
    try:
        while True:
            current_time = time.time()
            current_mod_times = get_file_modification_times()
            
            # Check if any files have been modified or if it's time for a periodic check
            files_changed = current_mod_times != last_mod_times
            time_to_check = current_time - last_check_time > args.interval
            
            if files_changed or time_to_check:
                print(f"\n{'='*60}")
                print(f"Running mypy check at {time.strftime('%H:%M:%S')}")
                
                if files_changed:
                    changed_files = []
                    for filepath in current_mod_times:
                        if filepath not in last_mod_times or current_mod_times[filepath] != last_mod_times[filepath]:
                            changed_files.append(filepath)
                    print(f"Changed files: {', '.join(changed_files[:3])}")
                    if len(changed_files) > 3:
                        print(f"  ... and {len(changed_files) - 3} more")
                
                print(f"{'='*60}")
                
                returncode, stdout, stderr = run_mypy_check(full_check=args.full)
                errors_by_file = parse_mypy_output(stdout)
                print_summary(errors_by_file, show_details=args.details)
                
                if stderr:
                    print(f"\nSTDERR: {stderr}")
                
                last_mod_times = current_mod_times
                last_check_time = current_time
            
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n\nStopping continuous type checking...")
        sys.exit(0)

if __name__ == "__main__":
    main()