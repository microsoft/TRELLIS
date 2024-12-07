#!/bin/bash

cd /app

echo "Doing post install steps"
./post_install.sh

export CXX=/usr/local/bin/gxx-wrapper

echo "Launching app"
python3 app.py

