import re
import os
from pathlib import Path
from datasets import load_dataset
from benchmarks.base import BaseBenchmark
from harness.sandbox import execute_sql

# Location where prefetch_datasets.py extracts the Spider SQLite files.
_SPIDER_DB_DIR = Path(__file__).parent.parent / "data" / "spider" / "database"


SYSTEM = "You are a SQL expert. Given a database schema and a natural language question, write a single valid SQLite SQL query. Return ONLY the SQL query with no explanation."


def extract_sql(response: str) -> str:
    # 1. Fenced code block (most reliable)
    match = re.search(r'```(?:sql)?\n(.*?)```', response, re.DOTALL)
    if match:
        return match.group(1).strip()
    # 2. Explicit semicolon terminator — handles multi-line SQL with JOINs/subqueries
    match = re.search(r'(SELECT\b.*?);', response, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    # 3. First SELECT line — stops at newline so trailing explanation is excluded.
    #    Then strip trailing prose sentence ("SELECT ... FROM t. This returns ...").
    #    The sub-pattern `. ` + word matches sentence endings but NOT schema.table dots.
    match = re.search(r'(SELECT\b[^\n]+)', response, re.IGNORECASE)
    if match:
        sql = match.group(1).strip()
        sql = re.sub(r'\.\s+\w.*$', '', sql).strip()
        return sql
    return response.strip()


class SpiderBenchmark(BaseBenchmark):
    name = "spider"

    def load_samples(self) -> list[dict]:
        ds = load_dataset("xlangai/spider", split="validation")
        samples = []
        for i, row in enumerate(ds):
            db_id = row.get("db_id", "")
            db_path = _SPIDER_DB_DIR / db_id / f"{db_id}.sqlite"
            samples.append({
                "id": f"spider_{i}",
                "question": row["question"],
                "schema": db_id,
                "prompt": f"Database: {db_id}\n\nQuestion: {row['question']}",
                "expected_sql": row["query"],
                "db_path": str(db_path) if db_path.exists() else None,
            })
        return samples

    def system_prompt(self) -> str:
        return SYSTEM

    def score(self, sample: dict, response: str) -> dict:
        predicted_sql = extract_sql(response)

        # If we have the DB file, execute both and compare result sets
        db_path = sample.get("db_path")
        if db_path and os.path.exists(db_path):
            import sqlite3
            try:
                conn = sqlite3.connect(db_path)
                expected_rows = conn.execute(sample["expected_sql"]).fetchall()
                conn.close()
                result = execute_sql(predicted_sql, db_path, expected_rows)
                return {
                    "passed": result["passed"],
                    "score": float(result["passed"]),
                    "predicted_sql": predicted_sql,
                    "exec_error": result.get("error"),
                    "method": "execution",
                }
            except Exception as e:
                pass

        # Fallback: normalised string match.
        # Spider's reference queries use quirky spacing (e.g. "name ,  country");
        # models generate standard spacing. Normalise both to a canonical form
        # before comparing so cosmetic differences don't cause false failures.
        def normalise(sql):
            sql = sql.lower().strip().rstrip(';')
            sql = re.sub(r'\s+', ' ', sql)
            sql = re.sub(r'\s*,\s*', ', ', sql)
            sql = re.sub(r'\s*=\s*', ' = ', sql)
            sql = re.sub(r'\s*<\s*', ' < ', sql)
            sql = re.sub(r'\s*>\s*', ' > ', sql)
            sql = re.sub(r'\binner\s+join\b', 'join', sql)
            return sql

        passed = normalise(predicted_sql) == normalise(sample["expected_sql"])
        return {
            "passed": passed,
            "score": float(passed),
            "predicted_sql": predicted_sql,
            "expected_sql": sample["expected_sql"],
            "method": "string_match",
        }
