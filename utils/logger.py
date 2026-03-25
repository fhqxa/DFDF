import os
import sys
import time
import os.path as osp
from torch.utils.tensorboard import SummaryWriter

__all__ = ["Logger", "setup_logger"]


class Logger:
    """Write console output to external text file.

    Args:
        fpath (str): directory to save logging file.
    """

    def __init__(self, fpath=None, tb_log_dir=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            os.makedirs(osp.dirname(fpath), exist_ok=True)
            self.file = open(fpath, "w")

        # 初始化 TensorBoard writer
        if tb_log_dir is not None:
            os.makedirs(tb_log_dir, exist_ok=True)
            self._writer = SummaryWriter(log_dir=tb_log_dir)
        else:
            self._writer = None

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()
        if hasattr(self, '_writer') and self._writer is not None:
            self._writer.close()


def setup_logger(output=None):
    if output is None:
        return

    if output.endswith(".txt") or output.endswith(".log"):
        fpath = output
    else:
        fpath = osp.join(output, "log.txt")

    if osp.exists(fpath):
        # make sure the existing log file is not over-written
        fpath += time.strftime("-%Y-%m-%d-%H-%M-%S")

    # 设置 TensorBoard 日志目录
    tb_log_dir = osp.join(osp.dirname(fpath), "tensorboard") if osp.dirname(fpath) else "tensorboard"

    logger = Logger(fpath, tb_log_dir)
    sys.stdout = logger
    return logger  # 返回 logger 对象以便在其他地方使用
