#!/bin/bash
set -e

cd /app

# Run post-install steps if not already done
if [ ! -f /app/.post_install_done ]; then
    echo "Running post-install steps..."
    
    # Install GPU-dependent packages
    conda run -n base ./setup.sh --mipgaussian --diffoctreerast
    
    # Verify installation
    export CXX=/usr/local/bin/gxx-wrapper
    python example.py
    
    # Mark completion
    touch /app/.post_install_done
    echo "Post-install steps completed successfully."
fi

# Set compiler wrapper for runtime
export CXX=/usr/local/bin/gxx-wrapper

echo "Launching headless API server..."
python3 headless_app.py
