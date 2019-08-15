import os

class Logger(object):
    def __init__(self, folder, log_file):
        self.folder = folder
        self.log_file = os.path.join(folder, log_file)
        with open(self.log_file, 'w+') as f:
            self._log(f"Logger initialized, logging to {self.log_file}", f)

    def _log(self, message, f):
        print(message)
        print(message, file=f, flush=True)

    def log(self, message):
        with open(self.log_file, 'a+') as f:
            self._log(message, f)

