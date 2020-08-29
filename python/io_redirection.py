from contextlib import contextmanager
import sys, os

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try: yield
        finally: sys.stdout = old_stdout

@contextmanager
def redirect_stdout(path):
    with open(path, "w") as outfile:
        old_stdout = sys.stdout
        sys.stdout = outfile
        try: yield
        finally: sys.stdout = old_stdout

@contextmanager
def redirect_stdout_stderr(stdout_path, stderr_path):
    with open(stdout_path, "w") as stdout_file:
        with open(stderr_path, "w") as stderr_file:
            old_stdout = sys.stdout
            sys.stdout = stdout_file
            old_stderr = sys.stderr
            sys.stderr = stderr_file
            try: yield
            finally:
                sys.stdout = old_stdout
                sys.stderr = old_stderr
