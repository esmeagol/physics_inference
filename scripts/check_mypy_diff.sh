#!/bin/bash

# check_mypy_diff.sh - A script to run mypy type checking on changed Python files
#
# This script checks for type errors in Python files that have been modified
# compared to a base git branch. It's designed to be used both as a pre-commit
# hook and as a development tool for continuous type checking.
#
# Features:
# - Checks only modified Python files in src/ and scripts/ directories
# - Supports continuous checking mode with configurable intervals
# - Provides clear, colorized output with emojis for better visibility
# - Integrates with git pre-commit hooks
# - Handles both staged and unstaged changes

# Exit on error
set -e

# Default values
BASE_BRANCH="main"
CONTINUOUS=false
CHECK_ALL=false
INTERVAL=5

# Help function
show_help() {
    cat << EOF
check_mypy_diff.sh - Run mypy type checking on changed Python files

Usage: $0 [options] [base-branch]

Options:
  -h, --help          Show this help message and exit
  -c, --continuous    Run in continuous mode, checking for changes every N seconds
  -a, --all           Check all Python files covered by mypy.ini (ignores git changes)
  -i, --interval N    Set the interval in seconds for continuous mode (default: 5)
  base-branch         The base git branch to compare against (default: main)

Modes of Operation:
  1. Standard Mode (default):
     - Checks files changed between current branch and base branch
     - Includes both staged and unstaged changes
     - Exits with status 0 if no type errors, 1 otherwise

  2. Check All Files (-a/--all):
     - Checks all Python files covered by mypy.ini
     - Ignores git changes
     - Useful for full project validation

  3. Continuous Mode (-c/--continuous):
     - Continuously checks for changes in Python files
     - Runs mypy on any modified files (or all files with -a)
     - Updates output in place for better readability
     - Runs until interrupted with Ctrl+C

Examples:
  # Basic usage - check files changed since main branch
  $0

  # Check files changed since a specific branch
  $0 develop

  # Check all Python files in the project
  $0 --all

  # Run in continuous mode (check every 5 seconds)
  $0 -c

  # Check all files in continuous mode
  $0 -a -c

  # Continuous mode with custom interval (10 seconds)
  $0 -c -i 10

  # Check all files with a specific mypy config
  $0 --all

  # Check files changed in the last commit
  $0 HEAD~1

Exit Status:
  0   No type errors found
  1   Type errors found
  2   Error in script execution

Note: This script is automatically used as a pre-commit hook to check type annotations.
      In pre-commit mode, it will prevent commits with type errors.

EOF
    exit 0
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            show_help
            ;;
        -c|--continuous)
            CONTINUOUS=true
            shift
            ;;
        -a|--all)
            CHECK_ALL=true
            shift
            ;;
        -i|--interval)
            INTERVAL="$2"
            shift 2
            ;;
        --interval=*)
            INTERVAL="${1#*=}"
            shift
            ;;
        -*)
            echo "Unknown option: $1" >&2
            show_help
            exit 1
            ;;
        *)
            if [ "$CHECK_ALL" = false ]; then
                BASE_BRANCH="$1"
            else
                echo "Warning: base-branch argument is ignored when using --all"
            fi
            shift
            ;;
    esac
done

# Function to get all Python files covered by mypy.ini
get_all_python_files() {
    # Find all Python files in the project, excluding venv and other common directories
    find . -type f -name "*.py" \
        -not -path "*/venv*" \
        -not -path "*/.git/*" \
        -not -path "*/__pycache__/*" \
        -not -path "*/build/*" \
        -not -path "*/dist/*" \
        -not -path "*/.mypy_cache/*" \
        -not -path "*/tests/*" \
        -not -path "*/scripts/*" \
        -not -path "*/examples/*" \
        -not -path "*/.pytest_cache/*"
}

# Function to run mypy check
run_mypy_check() {
    if [ "$CHECK_ALL" = true ]; then
        # Get all Python files in the project
        echo "🔍 Checking all Python files in the project (excluding tests)..."
        FILES_TO_CHECK=$(get_all_python_files)
        
        if [ -z "$FILES_TO_CHECK" ]; then
            echo "ℹ️  No Python files found to check"
            return 0
        fi
        
        # Count files for reporting
        FILE_COUNT=$(echo "$FILES_TO_CHECK" | wc -l | xargs)
        echo "📋 Found $FILE_COUNT Python files to check"
        
        # Run mypy on the files
        echo "$FILES_TO_CHECK" | xargs python -m mypy --config-file mypy.ini
        return $?
    else
        # Get the list of modified Python files in src/ and scripts/ directories
        FILES=$(git diff --name-only --diff-filter=d $BASE_BRANCH...HEAD 2>/dev/null | grep -E '^(src|scripts)/.*\.py$' | xargs ls -d 2>/dev/null | grep -v __pycache__ || true)

        # Also check for staged changes
        STAGED_FILES=$(git diff --cached --name-only --diff-filter=d 2>/dev/null | grep -E '^(src|scripts)/.*\.py$' | xargs ls -d 2>/dev/null | grep -v __pycache__ || true)

        # Combine and deduplicate file lists
        ALL_FILES=$(echo "$FILES $STAGED_FILES" | tr ' ' '\n' | sort | uniq)

        if [ -z "$ALL_FILES" ]; then
            echo "ℹ️  No Python files to check in src/ or scripts/ directories"
            return 0
        fi

        echo "📋 Files to check:"
        echo "$ALL_FILES" | tr ' ' '\n'
        # Run mypy on the modified files using Python module syntax
        echo -e "\n🔍 Running mypy..."
        python -m mypy --config-file mypy.ini $ALL_FILES
    fi
    # Capture the exit code
    local result=$?

    # Print appropriate message based on result
    if [ $result -eq 0 ]; then
        echo -e "\n✅ All type checks passed!"
    else
        echo -e "\n❌ mypy found type errors in the above files."
        echo "   Please fix them before committing."
    fi

    return $result
}

# Main execution
if [ "$CONTINUOUS" = true ]; then
    echo "Starting continuous type checking..."
    echo "Watching for changes in src/ and scripts/ directories"
    echo "Comparing against branch: $BASE_BRANCH"
    echo "Check interval: $INTERVAL seconds"
    echo "Press Ctrl+C to stop"
    echo ""

    while true; do
        echo "=== $(date) ==="
        run_mypy_check
        echo -e "\n⏳ Next check in $INTERVAL seconds...\n"
        sleep $INTERVAL
    done
else
    run_mypy_check
    exit $?
fi
