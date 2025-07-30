#!/bin/bash
echo "Starting continuous type checking..."
echo "This will run in the background and check for type errors every 5 seconds."
echo "Press Ctrl+C to stop."
echo ""

python continuous_type_check.py --interval 5