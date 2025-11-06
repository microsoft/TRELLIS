#!/bin/bash
set -e

cd /app

# Run post-install steps if not already done
if [ ! -f /app/.post_install_done ]; then
    echo "Running post-install steps..."
    
    # Install GPU-dependent packages
    ./setup.sh --mipgaussian --diffoctreerast
    
    # Verify installation
    export CXX=/usr/local/bin/gxx-wrapper
    python3.11 example.py
    
    # Mark completion
    touch /app/.post_install_done
    echo "Post-install steps completed successfully."
fi

# Set compiler wrapper for runtime
export CXX=/usr/local/bin/gxx-wrapper

echo "Launching RunPod handler..."
python3.11 -u rp_handler.py 