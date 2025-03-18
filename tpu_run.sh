#!/bin/bash

# Script to run AlgoNBA on TPU VMs safely

echo "AlgoNBA TPU VM Runner Script"
echo "============================"
echo ""

# Choose run mode:
echo "Select run mode:"
echo "1. Safe mode (CPU only, most reliable)"
echo "2. TPU detection mode (will detect but not use TPU)"
echo "3. TPU forced mode (will attempt to use TPU, may crash)"
echo ""
read -p "Enter choice [1-3] (default: 1): " choice

case $choice in
    2)
        echo ""
        echo "Running in TPU detection mode"
        echo "TPU will be detected but not used"
        # Use original command line args
        python main.py "$@"
        ;;
    3)
        echo ""
        echo "Running in TPU forced mode (may crash)"
        echo "Will attempt to use TPU acceleration"
        export ALGONBA_FORCE_TPU=1
        python main.py --use-tpu "$@"
        ;;
    *)
        # Default or option 1
        echo ""
        echo "Running in safe mode (CPU only)"
        echo "This is the most reliable option"
        # Disable all GPU/TPU devices
        export CUDA_VISIBLE_DEVICES=""
        export ALGONBA_DISABLE_TPU=1
        # Use original command line args
        python main.py "$@"
        ;;
esac
