#!/bin/bash

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to install gource using the appropriate package manager
install_gource() {
    echo "Attempting to install gource..."
    if command_exists apt-get; then
        sudo apt-get update && sudo apt-get install -y gource
    elif command_exists yum; then
        sudo yum install -y gource
    elif command_exists dnf; then
        sudo dnf install -y gource
    elif command_exists pacman; then
        sudo pacman -Syu gource
    else
        return 1
    fi
}

# Check if gource is installed
if ! command_exists gource; then
    echo "gource is not installed."
    if ! install_gource; then
        echo "Failed to install gource. Please install it manually from https://gource.io/"
        exit 1
    fi
fi

# Run gource with the specified options
gource --key
