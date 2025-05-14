#!/bin/bash

cd /app

echo "Doing post install steps"
./post_install.sh

export CXX=/usr/local/bin/gxx-wrapper

echo "Launching app"
python3 app.py

echo "Something went wrong and it exited?"
