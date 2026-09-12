"""Run the persistent reminder scheduler."""

from __future__ import annotations

import argparse
import logging
import pathlib
import time

from voicebot.scheduler import ReminderScheduler
from voicebot.storage import Storage

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> None:
    """Run the scheduler until interrupted.

    Audio delivery is intentionally not implemented here. The bot runtime can import
    the scheduler and provide a callback which owns TTS and audio arbitration.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database", type=pathlib.Path, default=pathlib.Path("voicebot.db")
    )
    parser.add_argument("--interval", type=float, default=1.0)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)

    with Storage(args.database) as storage:
        scheduler = ReminderScheduler(storage=storage)
        try:
            while True:
                scheduler.tick()
                time.sleep(max(args.interval, 0.05))
        except KeyboardInterrupt:
            logger.info("Scheduler stopped")
        finally:
            scheduler.stop()


if __name__ == "__main__":
    main()
