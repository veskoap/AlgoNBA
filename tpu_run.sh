#!/bin/bash

# Script to run AlgoNBA on TPU VMs safely

# Check if dependencies are installed
check_dependencies() {
    echo "Checking dependencies..."
    PACKAGES="pandas numpy scikit-learn xgboost lightgbm torch nba_api geopy pytz joblib tqdm psutil requests"
    MISSING=""
    
    for pkg in $PACKAGES; do
        python -c "import $pkg" 2>/dev/null
        if [ $? -ne 0 ]; then
            MISSING="$MISSING $pkg"
        fi
    done
    
    if [ ! -z "$MISSING" ]; then
        echo "Missing dependencies:$MISSING"
        echo "Installing missing dependencies..."
        pip install $MISSING
    else
        echo "All dependencies installed."
    fi
}

echo "AlgoNBA TPU VM Runner Script"
echo "============================"
echo ""

# Choose run mode:
echo "Select run mode:"
echo "1. Safe mode (CPU only, most reliable)"
echo "2. TPU detection mode (will detect but not use TPU)"
echo "3. TPU forced mode (will attempt to use TPU, may crash)"
echo "4. Ultra-safe TPU mode (avoids PyTorch XLA initialization completely)"
echo "5. No-memory-allocation TPU mode (skips large tensor initialization)"
echo "6. Install dependencies only (doesn't run the model)"
echo ""
read -p "Enter choice [1-6] (default: 1): " choice

# Check and install dependencies
check_dependencies

case $choice in
    5)
        echo ""
        echo "Running in no-memory-allocation TPU mode"
        echo "Will skip large tensor initialization that crashes with topology error"
        # Set PJRT_DEVICE for TPU
        export XLA_FLAGS="--xla_cpu_enable_xprof=false"
        export ALGONBA_FORCE_TPU=1
        export ALGONBA_SAFE_TPU=1
        # Skip memory allocation entirely and use tiny batch size
        export ALGONBA_MAX_BATCH_SIZE=16
        export ALGONBA_SKIP_MEMORY_ALLOC=1
        # Run with quick mode to reduce memory requirements
        python main.py --use-tpu --quick "$@"
        ;;
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
        # Set SafeTPU environment variable to prevent SIGABRT
        export XLA_FLAGS="--xla_cpu_enable_xprof=false"
        export ALGONBA_FORCE_TPU=1
        # Use less aggressive TPU initialization
        export ALGONBA_SAFE_TPU=1
        # Use very small batch size to avoid topology error
        export ALGONBA_MAX_BATCH_SIZE=128
        python main.py --use-tpu --quick "$@"
        ;;
    4)
        echo ""
        echo "Running in ultra-safe TPU mode"
        echo "Will completely bypass PyTorch XLA initialization"
        # Disable XLA entirely
        export DISABLE_TORCH_XLA_RUNTIME=1
        export ALGONBA_ULTRA_SAFE_TPU=1
        export ALGONBA_DISABLE_TPU=1
        export CUDA_VISIBLE_DEVICES=""
        # Ensure Python doesn't try to load torch_xla at all
        PYTHONPATH=$(echo $PYTHONPATH | tr ':' '\n' | grep -v "torch_xla" | tr '\n' ':')
        export PYTHONPATH
        python main.py "$@"
        ;;
    6)
        echo ""
        echo "Installing dependencies only"
        echo "Installing all required packages from requirements.txt..."
        
        # First install PyTorch for TPU
        echo "Installing PyTorch for TPU..."
        pip install torch==1.13.1
        pip install torch_xla[tpu]==1.13.1
        
        # Then install the rest of the requirements
        echo "Installing other dependencies..."
        pip install -r requirements.txt
        
        echo "Dependencies installation complete."
        exit 0
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
