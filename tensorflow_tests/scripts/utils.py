from timeit import default_timer as timer

class Timer(object):
    def __init__(self):
        self._timers = {}

    def checkin(self, name):
        self._timers[name] = timer()

    def diff(self, start, end):
        if start in self._timers and end in self._timers:
            time = self._timers[end] - self._timers[start]
            return str(int(time / 60)) + "m " + str(int(time % 60)) + "s " + str(int((time % 60) * 1000 % 1000)) + "ms"

