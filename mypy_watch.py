#!/usr/bin/env python3
"""
Continuous mypy type checking script for PureCV and CVModelInference directories.
This script runs mypy in watch mode and provides continuous feedback on typing errors.
"""

import subprocess
import sys
import time
import os
from pathlib import Path
from typing import Tuple

def run_mypy_check() -> Tuple[int, str, str]:
    """Run mypy check and return the result."""
    try:
        result = subprocess.run(
            ["mypy", "CVModelInference", "PureCV", "--config-file", "mypy.ini"],
            capture_output=True,
            text=True,
            timeout=120
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return 1, "", "mypy check timed out"
    except FileNotFoundError:
        return 1, "", "mypy not found. Please install with: pip install mypy"

def get_file_modification_times() -> dict[str, float]:
    """Get modification times for all Python files in the target directories."""
    mod_times = {}
    for directory in ["CVModelInference", "PureCV"]:
        if os.path.exists(directory):
            for root, dirs, files in os.walk(directory):
                for file in files:
                    if file.endswith('.py'):
                        filepath = os.path.join(root, file)
                        mod_times[filepath] = os.path.getmtime(filepath)
    return mod_times

def main() -> None:
    """Main function to run continuous mypy checking."""
    print("Starting continuous mypy type checking...")
    print("Monitoring PureCV and CVModelInference directories")
    print("Press Ctrl+C to stop")
    
    last_mod_times = get_file_modification_times()
    last_check_time = 0.0
    
    try:
        while True:
            current_time = time.time()
            current_mod_times = get_file_modification_times()
            
            # Check if any files have been modified or if it's been 10 seconds
            files_changed = current_mod_times != last_mod_times
            time_to_check = current_time - last_check_time > 10
            
            if files_changed or time_to_check:
                print(f"\n{'='*60}")
                print(f"Running mypy check at {time.strftime('%H:%M:%S')}")
                if files_changed:
                    changed_files = []
                    for filepath in current_mod_times:
                        if filepath not in last_mod_times or current_mod_times[filepath] != last_mod_times[filepath]:
                            changed_files.append(filepath)
                    print(f"Changed files: {', '.join(changed_files)}")
                print(f"{'='*60}")
                
                returncode, stdout, stderr = run_mypy_check()
                
                if returncode == 0:
                    print("✅ No type errors found!")
                else:
                    print(f"❌ Found type errors:")
                    if stdout:
                        # Count errors
                        lines = stdout.strip().split('\n')
                        error_lines = [line for line in lines if ': error:' in line]
                        print(f"Total errors: {len(error_lines)}")
                        print(stdout)
                    if stderr:
                        print("STDERR:", stderr)
                
                last_mod_times = current_mod_times
                last_check_time = current_time
            
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n\nStopping continuous type checking...")
        sys.exit(0)

if __name__ == "__main__":
    main()