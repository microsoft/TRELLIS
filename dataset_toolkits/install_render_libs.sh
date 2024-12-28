#!/bin/bash

if [ -f /etc/os-release ]; then
    . /etc/os-release
    DISTRO_ID=$ID
else
    echo "Unknown Linux Distribution。"
    exit 1
fi

echo "Linux Distribution：$DISTRO_ID"

case "$DISTRO_ID" in
    ubuntu|debian)
        sudo apt-get update
        sudo apt-get install -y libxrender1 libxi6 libxkbcommon-x11-0 libsm6
        ;;
    centos|rhel)
        sudo dnf makecache
        sudo dnf install -y libXrender libXi libxkbcommon libSM
        ;;
    fedora)
        sudo dnf makecache
        sudo dnf install -y libXrender libXi libxkbcommon libSM
        ;;
    arch)
        sudo pacman -Syu --noconfirm
        sudo pacman -S --noconfirm libxrender libxi libxkbcommon-x11 libsm
        ;;
    opensuse*)
        sudo zypper refresh
        sudo zypper install -y libXrender1 libXi6 libxkbcommon-x11-0 libSM6
        ;;
    *)
        echo "Unsupported Linux：$DISTRO_ID"
        exit 1
        ;;
esac

echo "Done."
