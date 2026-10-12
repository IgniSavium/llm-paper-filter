"""Consistent exit codes for script-compatible command-line wrappers."""
import subprocess
import sys


def run(main):
    try:
        code = main()
    except KeyboardInterrupt:
        code = 130
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
        print(f'Error: {exc}', file=sys.stderr)
        code = 1
    raise SystemExit(code or 0)
