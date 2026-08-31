from pathlib import Path
import launch
import os

aria2path = Path(__file__).resolve().parents[0] / 'aria2'

for item in aria2path.iterdir():
    if item.is_file():
        item.unlink()

req_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'requirements.txt')

def dist2package(dist: str):
    return ({
        # 'send2trash': 'send2trash',
        'ZipUnicod': 'zip_unicode',
        'beautifulsoup4': 'bs4',
    }).get(dist, dist)

with open(req_file) as file:
    for package in file:
        try:
            package = package.strip()
            if '==' in package:
                package_name = package.split('==')[0]
            else:
                package_name = package

            if not launch.is_installed(dist2package(package_name)):
                launch.run_pip(f"install {package}", f"CivitAI Browser Neo requirement: {package}")
        except Exception as e:
            print(e)
            print(f'Warning: Failed to install {package}, something may not work.')