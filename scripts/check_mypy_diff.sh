#!/bin/bash

# Get the base branch (defaults to main)
BASE_BRANCH=${1:-main}

# Get the list of modified Python files in src/ and scripts/ directories
FILES=$(git diff --name-only --diff-filter=d $BASE_BRANCH...HEAD | grep -E '^(src|scripts)/.*\.py$' | xargs ls -d 2>/dev/null | grep -v __pycache__)

# Also check for staged changes
STAGED_FILES=$(git diff --cached --name-only --diff-filter=d | grep -E '^(src|scripts)/.*\.py$' | xargs ls -d 2>/dev/null | grep -v __pycache__)

# Combine and deduplicate file lists
ALL_FILES=$(echo "$FILES $STAGED_FILES" | tr ' ' '\n' | sort | uniq)

if [ -z "$ALL_FILES" ]; then
    echo "ℹ️  No Python files to check in src/ or scripts/ directories"
    exit 0
fi

echo "📋 Files to check:"
echo "$ALL_FILES" | tr ' ' '\n'
echo ""

# Run mypy on the modified files using Python module syntax
echo "🔍 Running mypy..."
python -m mypy --config-file mypy.ini $ALL_FILES

# Capture the exit code
RESULT=$?

# Print appropriate message based on result
if [ $RESULT -eq 0 ]; then
    echo "\n✅ All type checks passed!"
else
    echo "\n❌ mypy found type errors in the above files."
    echo "   Please fix them before committing."
fi

exit $RESULT
