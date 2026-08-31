import subprocess
import sys
import ensurepip

ensurepip.bootstrap()
subprocess.check_call([sys.executable, "-m", "pip", "install", "Pillow"])