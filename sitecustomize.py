# sitecustomize.py

# ensure src/ is on PYTHONPATH
import sys, pathlib

root = pathlib.Path(__file__).resolve().parent
src  = root / "src"
if str(src) not in sys.path:
    sys.path.insert(0, str(src))
