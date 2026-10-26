@echo off
title Demo IA Prostate - Lanceur
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -NoExit -File "%~dp0tools\launch_demo.ps1"
