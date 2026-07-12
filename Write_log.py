import sys
import os
from datetime import datetime

class SimLoggerTee:
    """Interceptors standard output and standard error streams to write to a file 
    and the terminal simultaneously with unbuffered real-time flushing."""
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        # Force flush instantly so data isn't lost if Isaac Sim crashes
        self.log.flush() 

    def flush(self):
        self.terminal.flush()
        self.log.flush()