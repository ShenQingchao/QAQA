import logging
import time


def singleton(cls):
    instances = {}

    def _singleton(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    return _singleton


@singleton
class MyLogger(object):
    def __init__(self):
        self.logger = logging.getLogger()
        log_format = logging.Formatter('%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s')
        self.logger.setLevel(logging.INFO)

        if not self.logger.handlers:
            sh_handler = logging.StreamHandler()
            time_this = time.strftime("%Y-%m-%d_%H_%M", time.localtime())
            f_handler = logging.FileHandler(f"log.txt_{time_this}", mode='w')
            sh_handler.setFormatter(log_format)
            f_handler.setFormatter(log_format)
            self.logger.addHandler(sh_handler)
            self.logger.addHandler(f_handler)

    def get_logger(self):
        return self.logger


logger = MyLogger().get_logger()

