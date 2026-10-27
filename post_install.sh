#!/bin/bash
set -e

# Check if post-install steps have already been run
if [ -f /app/.post_install_done ]; then
    echo "Post-install steps already completed."
    exit 0
fi

cd /app

echo "Install stuff that couldn't be installed without GPU"
# Run the demo setup
conda run -n base ./setup.sh --mipgaussian --diffoctreerast

echo "Proving it actually works..."

export CXX=/usr/local/bin/gxx-wrapper
python example.py

# Mark completion
touch /app/.post_install_done

echo "Post-install steps completed successfully."

