@echo off
title TRELLIS - seed sweep
rem Drag one or more images onto this file. Each image is generated with several seeds
rem (default: 1 2 3 4) and 25 sampling steps, into outputs\IMAGE-NAME_seedN\
rem Set TRELLIS_SEEDS / TRELLIS_STEPS before running to override.
if "%~1"=="" (
  echo Drag one or more image files onto %~nx0
  pause
  exit /b 1
)
if not defined TRELLIS_SEEDS set "TRELLIS_SEEDS=1 2 3 4"
if not defined TRELLIS_STEPS set "TRELLIS_STEPS=25"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_test.ps1" %*
echo.
pause
