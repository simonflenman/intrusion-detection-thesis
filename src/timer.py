# src/timer.py
import sys
import time
import threading

def start_timer():
    """
    Launch a daemon thread that writes an updating
    Elapsed: HH:MM:SS clock on the same line.
    """
    t0 = time.time()
    def _tick():
        while True:
            elapsed = int(time.time() - t0)
            h, rem = divmod(elapsed, 3600)
            m, s = divmod(rem, 60)
            sys.stdout.write(f"\rElapsed: {h:02d}:{m:02d}:{s:02d}")
            sys.stdout.flush()
            time.sleep(1)
    threading.Thread(target=_tick, daemon=True).start()
