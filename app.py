#!/usr/bin/env python3
"""ModelScope Studio entry point for sd-webui-forge-neo-v3"""

import os
import sys

# Path to the webui directory
WEBUI_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sd-webui-forge-neo-v3', 'webui')


def main():
    # Change to webui directory
    os.chdir(WEBUI_DIR)
    sys.path.insert(0, WEBUI_DIR)

    # Inject command-line args for the ModelScope environment
    sys.argv = [
        sys.argv[0],
        "--skip-python-version-check",
        "--skip-torch-cuda-test",
        "--no-half-vae",
        "--api",
    ]

    # Prepare environment
    from modules import launch_utils
    launch_utils.prepare_environment()

    # Start webui
    launch_utils.start()


if __name__ == '__main__':
    main()