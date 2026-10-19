import sys
from pathlib import Path

# The tests import setup_tests.py from the repository root. Append rather than prepend it, so `import deepethogram`
# still finds the installed package; the Docker test images install it with --no-editable to test the built wheel.
sys.path.append(str(Path(__file__).resolve().parents[1]))
