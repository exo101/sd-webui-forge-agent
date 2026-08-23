#!/bin/bash
# System dependencies for UI testing on headless Ubuntu 22.04
# Run with: sudo bash install_system_deps.sh
#
# What each package does:
#   xvfb              - Virtual X11 framebuffer (renders GUI without a monitor)
#   libxcb-cursor0    - XCB cursor library (required by Qt6 xcb platform plugin)
#   xdotool           - X11 automation (send keystrokes, move windows, etc.)
#   scrot             - Screenshot capture tool
#
# Total download: ~5MB

set -e

apt update -qq
apt install -y --no-install-recommends \
    xvfb \
    libxcb-cursor0 \
    xdotool \
    scrot

echo ""
echo "Installed successfully. Verify:"
echo "  Xvfb --help 2>&1 | head -1"
echo "  xvfb-run echo 'xvfb-run works'"
echo "  dpkg -s libxcb-cursor0 | grep Status"
