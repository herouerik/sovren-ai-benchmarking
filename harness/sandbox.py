import subprocess
import sys
import tempfile
import os
import textwrap


def execute_python(code: str, test_code: str, timeout: int = 10) -> dict:
    """Run generated code + test assertions in a subprocess. Never uses exec()."""
    full_code = textwrap.dedent(code) + "\n\n" + textwrap.dedent(test_code)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(full_code)
        tmp_path = f.name

    try:
        result = subprocess.run(
            # sys.executable, not "python3": the bare name resolves through
            # PATH, so it only picks up the venv when the venv happens to be
            # activated. Invoked as `.venv/bin/python run_benchmark.py` — no
            # activation — it landed on the system interpreter instead, which
            # has no numpy. EvalPlus's test suites import numpy, so 97 of 100
            # humaneval_plus samples failed with ModuleNotFoundError while the
            # model's own code was correct, scoring a capable model 0.00.
            # sys.executable is whichever Python is running the harness, which
            # is by definition the one with its dependencies installed.
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        passed = result.returncode == 0
        return {
            "passed": passed,
            "stdout": result.stdout[:2000],
            "stderr": result.stderr[:2000],
            "error": None,
        }
    except subprocess.TimeoutExpired:
        return {"passed": False, "stdout": "", "stderr": "", "error": "timeout"}
    except Exception as e:
        return {"passed": False, "stdout": "", "stderr": "", "error": str(e)}
    finally:
        os.unlink(tmp_path)


def execute_sql(sql: str, db_path: str, expected_rows: list) -> dict:
    """Execute SQL against a SQLite DB and compare result sets."""
    import sqlite3
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(sql)
        rows = cursor.fetchall()
        conn.close()
        # Compare as sets of tuples (order-insensitive)
        got = set(tuple(str(v) for v in row) for row in rows)
        expected = set(tuple(str(v) for v in row) for row in expected_rows)
        return {"passed": got == expected, "got": list(got), "error": None}
    except Exception as e:
        return {"passed": False, "got": [], "error": str(e)}
