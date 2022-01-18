import time
import sys
import traceback
import functools
import multiprocessing
from datetime import datetime


manager = multiprocessing.Manager()


class LogCollector:
    def __init__(self, writer=None, task_id=None, verbose=True):
        self.writer = writer
        self.task_id = task_id
        self.last_monitor_collect_time = None
        self.verbose = verbose
        self.current_status = ""
        self.error_msg = manager.list()

    def set_logger(self, writer, task_id, verbose):
        self.writer = writer
        self.task_id = task_id
        self.verbose = verbose

    def get_time(self):
        return int(time.time())

    def monitor_collect(self, percent, status, per_seconds=5, force=False):
        if self.last_monitor_collect_time is None or \
            (time.time() - self.last_monitor_collect_time) > per_seconds or force:
            current_time = self.get_time()
            percent = round(percent, 5)
            items = [self.task_id, current_time, percent, status]
            items = list(map(str, items))
            _log = "\t".join(items)
            # if self.verbose:
            #     print(_log)
            status_change = False
            if self.current_status != status:
                self.current_status = status
                status_change = True
            self.writer.write(_log, "progress", status_change)
            self.last_monitor_collect_time = time.time()

    def error_collect(self, msg):
        self.writer.write(msg, "error")
        # if self.verbose:
        #     print(msg)

    def summary_collect(self, msg):
        now = datetime.now()
        str_now = now.strftime("%Y-%m-%d %H:%M:%S")
        msg = str_now + "\t" + msg
        self.writer.write(msg, "summary")
        if self.verbose:
            print(msg)


class LogWriter:
    def __init__(self, monitor_path=None, monitor_pure_path=None, summary_path=None):
        self.monitor_file = open(monitor_path, "w")
        self.monitor_pure_file = open(monitor_pure_path, "w")
        self.summary_file = open(summary_path, "w")

    def write(self, log, log_type, status_change=False):
        if log_type == "progress":
            self.monitor_file.seek(0)
            self.monitor_file.write(log + "\n")
            self.monitor_file.truncate()
            self.monitor_file.flush()

            self.monitor_pure_file.write(log + "\n")
            self.monitor_pure_file.flush()
        elif log_type == "error":
            self.monitor_file.write(log + "\n")
            self.monitor_file.flush()

        elif log_type == "summary":
            self.summary_file.write(log + "\n")
            self.summary_file.flush()

    def __exit__(self):
        self.monitor_file.close()
        self.monitor_pure_file.close()
        self.summary_file.close()


log_collector = LogCollector()


def try_exception_log(func):
    @functools.wraps(func)
    def warpper(*args, **kw):
        try:
            func(*args, **kw)
        except Exception:
            exctype, value, tb = sys.exc_info()
            text = "".join(traceback.format_exception(exctype, value, tb))
            log_collector.error_msg.append(text)
    return warpper

# override default bahavior of sys.excepthook
# def foo_excepthook(exctype, value, tb):
#     text = "".join(traceback.format_exception(exctype, value, tb))
#     try:
#         print("in hook")
#         log_collector.error_collect(text)
#     except TypeError:
#         pass
#     print(text)


# sys.excepthook = foo_excepthook
