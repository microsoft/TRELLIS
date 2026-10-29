@echo off
title TRELLIS - install
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install.ps1"
if errorlevel 1 (
  echo.
  echo INSTALL FAILED - details in windows_setup\logs\install.log
  pause
  exit /b 1
)
echo.
echo Install OK - starting the test generation...
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_test.ps1"
echo.
pause
