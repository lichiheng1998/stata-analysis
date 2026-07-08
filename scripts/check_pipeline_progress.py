import argparse
import os
import sqlite3
import time
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = ROOT / "output" / "pipeline_state.sqlite"


def parse_args():
    parser = argparse.ArgumentParser(description="Check digital pipeline SQLite progress.")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--recent", type=int, default=12)
    parser.add_argument("--failed", type=int, default=10)
    parser.add_argument(
        "--watch",
        type=int,
        default=0,
        help="Refresh every N seconds. Use 0 for a one-time snapshot.",
    )
    return parser.parse_args()


def fetch_all(conn, query, params=()):
    return conn.execute(query, params).fetchall()


def print_rows(title, rows):
    print(f"\n{title}")
    if not rows:
        print("  (none)")
        return
    for row in rows:
        print(" ", tuple(row))


def parse_time(value):
    if not value:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(value.split("+")[0], fmt)
        except ValueError:
            pass
    return None


def format_seconds(seconds):
    if seconds is None:
        return "unknown"
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")


def print_snapshot(args):
    db_path = args.db
    print(f"db={db_path.resolve()}")
    print(f"db_exists={db_path.exists()}")
    if not db_path.exists():
        raise SystemExit(1)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    status_rows = fetch_all(
        conn,
        "SELECT status, COUNT(*) AS count FROM tasks GROUP BY status ORDER BY status",
    )
    status_counts = {row["status"]: row["count"] for row in status_rows}
    total = sum(status_counts.values())
    done = status_counts.get("done", 0)
    failed = status_counts.get("failed", 0)
    running = status_counts.get("running", 0)
    pending = status_counts.get("pending", 0)
    pct = done / total * 100 if total else 0.0
    remaining = pending + running

    timing = conn.execute(
        """
        SELECT MIN(updated_at) AS first_done, MAX(updated_at) AS last_done
        FROM tasks
        WHERE status='done'
        """
    ).fetchone()
    first_done = parse_time(timing["first_done"])
    last_done = parse_time(timing["last_done"])
    elapsed = (last_done - first_done).total_seconds() if first_done and last_done else None
    rate = done / elapsed if elapsed and elapsed > 0 else None
    eta = remaining / rate if rate else None

    print("\nsummary")
    print(f"  checked_at={datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  total={total}")
    print(f"  done={done}")
    print(f"  running={running}")
    print(f"  pending={pending}")
    print(f"  failed={failed}")
    print(f"  done_pct={pct:.2f}%")
    print(f"  avg_rate={rate:.3f} reports/sec" if rate else "  avg_rate=unknown")
    print(f"  eta={format_seconds(eta)}")

    print_rows("status_counts", status_rows)

    print_rows(
        "year_status",
        fetch_all(
            conn,
            """
            SELECT year, status, COUNT(*)
            FROM tasks
            GROUP BY year, status
            ORDER BY year, status
            """,
        ),
    )

    print_rows(
        "results_by_year",
        fetch_all(
            conn,
            """
            SELECT year, COUNT(*),
                   ROUND(AVG(digital_char_ratio), 6),
                   ROUND(AVG(digital_sent_ratio), 6)
            FROM report_results
            GROUP BY year
            ORDER BY year
            """,
        ),
    )

    print_rows(
        "failed_recent",
        fetch_all(
            conn,
            """
            SELECT year, stock_id, SUBSTR(error, 1, 180)
            FROM tasks
            WHERE status='failed'
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            (args.failed,),
        ),
    )

    print_rows(
        "recent_done",
        fetch_all(
            conn,
            """
            SELECT year, stock_id, updated_at
            FROM tasks
            WHERE status='done'
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            (args.recent,),
        ),
    )

    conn.close()


def main():
    args = parse_args()
    if args.watch < 0:
        raise SystemExit("--watch must be >= 0")

    while True:
        if args.watch:
            clear_screen()
        print_snapshot(args)
        if not args.watch:
            break
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
