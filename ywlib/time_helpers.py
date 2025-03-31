import time

time_at_last_run = 0


def fill_time(seconds):
    """ Sleeps in order to make the time passed since the last call to this function, at least as long as the
    given number of seconds given. """
    global time_at_last_run

    time_now = time.process_time()

    time_since_last_run = time_now - time_at_last_run
    wait_interval = seconds - time_since_last_run
    if wait_interval < 0:
        wait_interval = 0

    time_at_last_run = time_now
    time.sleep(wait_interval)
