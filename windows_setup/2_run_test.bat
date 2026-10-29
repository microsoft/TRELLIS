@echo off
title TRELLIS - image to 3D
rem Double-click for the demo image, or drag an image file onto this .bat
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_test.ps1" %*
echo.
pause
