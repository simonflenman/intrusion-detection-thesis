import sys
import time
import threading

def start_timer():
    """
    Starts a background daemon thread that continuously updates and displays
    an elapsed time clock in the console in the format HH:MM:SS.

    The timer runs in a loop, updating every second, and is useful for providing
    real-time feedback during long-running tasks or scripts.
    """
    t0 = time.time()

    def _tick():
        """
        Updates and prints the elapsed time on the same console line.
        Runs indefinitely in a daemon thread.
        """
        while True:
            elapsed = int(time.time() - t0)
            h, rem = divmod(elapsed, 3600)
            m, s = divmod(rem, 60)
            sys.stdout.write(f"\rElapsed: {h:02d}:{m:02d}:{s:02d}")
            sys.stdout.flush()
            time.sleep(1)

    # Start the ticking function in a daemon thread
    threading.Thread(target=_tick, daemon=True).start()
