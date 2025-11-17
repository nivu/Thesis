#!/bin/bash
# Quick status checker for automated demo

echo "=========================================="
echo "Automated Demo Status"
echo "=========================================="
echo ""

# Check if process is running
if ps -p 13574 > /dev/null 2>&1; then
    echo "✓ Auto-demo process: RUNNING (PID: 13574)"
else
    echo "✗ Auto-demo process: NOT RUNNING"
fi

echo ""
echo "Training Progress:"
echo "------------------"
python3 check_training_status.py 2>/dev/null | grep -A 20 "Training Progress" || echo "Training status not available"

echo ""
echo "Recent Activity:"
echo "----------------"
tail -20 auto_demo.log 2>/dev/null || echo "No log file yet"

echo ""
echo "=========================================="
echo "To view full log: tail -f auto_demo.log"
echo "=========================================="
