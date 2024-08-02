#!/bin/bash

output_name="nia_start"

# Compile the core
./compile.sh -j 4

# Create executable with PyInstaller
pyinstaller --noconfirm \
            --onedir \
            --console \
            --clean \
            --log-level "INFO" \
            --name "nia_start_exe" \
            --add-data "./build:build/" \
            --add-data "./icons:icons/" \
            --add-data "./results:results/" \
            --add-data "./tests:tests/" \
            --add-data "./ui:ui/" \
            --add-binary "./nia_start_core:." \
            --paths "./ui" \
            "./ui/main.py"
