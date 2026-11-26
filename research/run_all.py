"""One-command entry point; run with backend environment or research requirements."""
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parents[1]
cmd = [sys.executable, str(Path(__file__).with_name("evaluate.py")), *sys.argv[1:]]
raise SystemExit(subprocess.call(cmd, cwd=root))
