
"""Standalone detection monitor for the prediction API.

Run from project root:
    python detection/monitor.py
    python detection/monitor.py --follow
"""

import argparse
import json
import time
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG = ROOT / "fahad_api" / "logs" / "queries.jsonl"
DEFAULT_ALERTS = Path(__file__).resolve().parent / "alerts.jsonl"

WINDOW_SECONDS = 60
BURST_THRESHOLD = 500
TOTAL_THRESHOLD = 2000
BLOCKED_THRESHOLD = 20

# Exclude the project's trusted test-set evaluation requests.
IGNORE_CLIENTS = {"zoe-extraction-eval"}


class Monitor:
    def __init__(self, alert_file):
        self.alert_file = alert_file
        self.total = defaultdict(int)
        self.recent = defaultdict(deque)
        self.blocked = defaultdict(deque)
        self.alerted = set()

    def alert(self, entry, client, rule, count, severity="HIGH"):
        key = (client, rule)

        # Alert only once per client/rule during this run.
        if key in self.alerted:
            return

        self.alerted.add(key)

        event = {
            "timestamp": entry["timestamp"],
            "client_id": client,
            "severity": severity,
            "rule": rule,
            "count": count,
        }

        self.alert_file.write(json.dumps(event) + "\n")
        self.alert_file.flush()

        print(
            f"[ALERT] {severity} | {client} | "
            f"{rule} | count={count}",
            flush=True,
        )

    def process(self, line):
        try:
            entry = json.loads(line)
            timestamp = datetime.fromisoformat(
                entry["timestamp"].replace("Z", "+00:00")
            )
        except (ValueError, KeyError, TypeError):
            return

        client = (
            entry.get("client_id")
            or entry.get("client_ip")
            or "unknown"
        )

        if client in IGNORE_CLIENTS:
            return

        # Track total requests by client.
        self.total[client] += 1

        # Track requests in the last 60 seconds.
        recent = self.recent[client]
        recent.append(timestamp)

        while (
            recent
            and (timestamp - recent[0]).total_seconds()
            > WINDOW_SECONDS
        ):
            recent.popleft()

        # Track HTTP 429 responses in the last 60 seconds.
        blocked = self.blocked[client]

        if entry.get("status") == 429:
            blocked.append(timestamp)

        while (
            blocked
            and (timestamp - blocked[0]).total_seconds()
            > WINDOW_SECONDS
        ):
            blocked.popleft()

        # Detection rule 1: rapid requests.
        if len(recent) >= BURST_THRESHOLD:
            self.alert(
                entry, client,
                "HIGH_QUERY_RATE", len(recent)
            )

        # Detection rule 2: large total query volume.
        if self.total[client] >= TOTAL_THRESHOLD:
            self.alert(
                entry, client,
                "HIGH_TOTAL_QUERIES", self.total[client]
            )

        # Detection rule 3: continued requests after blocking.
        if len(blocked) >= BLOCKED_THRESHOLD:
            self.alert(
                entry, client,
                "REPEATED_RATE_LIMIT_BLOCKS", len(blocked)
            )


def main():
    parser = argparse.ArgumentParser(
        description="Monitor API query logs"
    )

    parser.add_argument(
        "--follow",
        action="store_true",
        help="Watch new log lines continuously",
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=DEFAULT_LOG,
    )
    parser.add_argument(
        "--alerts",
        type=Path,
        default=DEFAULT_ALERTS,
    )

    args = parser.parse_args()
    args.alerts.parent.mkdir(parents=True, exist_ok=True)

    # Replay replaces previous alerts; live monitoring appends.
    mode = "a" if args.follow else "w"

    with args.alerts.open(mode, encoding="utf-8") as output:
        monitor = Monitor(output)

        # Mode 1: Process existing log file.
        if not args.follow:
            if not args.log.exists():
                print(f"No API log found: {args.log}")
                return

            with args.log.open(encoding="utf-8") as source:
                for line in source:
                    monitor.process(line)

            print(
                f"Replay complete. Alerts saved to "
                f"{args.alerts}"
            )
            return

        # Mode 2: Monitor newly appended requests.
        print(
            f"Watching new requests in {args.log} "
            f"(Ctrl+C to stop)"
        )

        try:
            while not args.log.exists():
                time.sleep(0.5)

            with args.log.open(encoding="utf-8") as source:
                source.seek(0, 2)

                while True:
                    line = source.readline()

                    if line:
                        monitor.process(line)
                    else:
                        time.sleep(0.25)

        except KeyboardInterrupt:
            print("Monitor stopped.")


if __name__ == "__main__":
    main()
