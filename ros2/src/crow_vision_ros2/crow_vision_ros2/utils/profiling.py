from time import time


class MicroTimer():

    def __init__(self, logger=None, enabled=True):
        self.enabled = enabled
        self.startTime = time()
        self.lastLapTime = self.startTime
        if logger is not None:
            self.log = logger.info
        else:
            self.log = print


    def lap(self, section="", ending=""):
        if not self.enabled:
            return
        self.lapTime = time()
        elapsedLap = self.lapTime - self.lastLapTime
        elapsedTolal = self.lapTime - self.startTime
        if self.lastLapTime == self.startTime:
            preamb = "==========>\n"
        else:
            preamb = ""
        self.lastLapTime = self.lapTime
        self.log("{}Elapsed time:\n\tper section {} = {}\n\ttotal = {}{}".format(preamb, "({})".format(section) if section else "", elapsedLap, elapsedTolal, ending))

    def end(self, section=""):
        self.lap(section=section, ending="\n<==========")
