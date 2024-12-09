#!/bin/bash

cd /app

echo "doing post install steps"
./post_install.sh

export CXX=/usr/local/bin/gxx-wrapper

echo "launching app"
python3 app.py

