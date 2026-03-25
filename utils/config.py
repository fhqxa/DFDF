from yacs.config import CfgNode as CN

_C = CN()

# ---------------------------------------------------------------------------- #
# DATA
# ---------------------------------------------------------------------------- #
_C.DATA = CN()
_C.DATA.NAME = ""
_C.DATA.IMG_SIZE = 224

# ---------------------------------------------------------------------------- #
# MODEL
# ---------------------------------------------------------------------------- #
_C.MODEL = CN()
_C.MODEL.NAME = ""
_C.MODEL.NUM_CLASSES = 0
_C.MODEL.NUM_SEGMENT = 1

# ---------------------------------------------------------------------------- #
# TRAIN
# ---------------------------------------------------------------------------- #
_C.TRAIN = CN()
_C.TRAIN.EPOCHS = 100
_C.TRAIN.BATCH_SIZE = 64
_C.TRAIN.DEVICE = "cuda:0"
_C.TRAIN.AMP = True
_C.TRAIN.SEED = 42
_C.TRAIN.STRATEGY = ""

# ---------------------------------------------------------------------------- #
# OPTIMIZER
# ---------------------------------------------------------------------------- #
_C.OPTIMIZER = CN()
_C.OPTIMIZER.NAME = "AdamW"
_C.OPTIMIZER.LR = 1e-4
_C.OPTIMIZER.WEIGHT_DECAY = 1e-2

# ---------------------------------------------------------------------------- #
# SCHEDULER
# ---------------------------------------------------------------------------- #
_C.SCHEDULER = CN()
_C.SCHEDULER.NAME = "cosine"

# ---------------------------------------------------------------------------- #
# LOSS
# ---------------------------------------------------------------------------- #
_C.LOSS = CN()

# classification loss
_C.LOSS.CLS = CN()
_C.LOSS.CLS.NAME = "CE"
_C.LOSS.CLS.SCALE = 25

# segmentation loss
_C.LOSS.SEG = CN()
_C.LOSS.SEG.NAME = "Dice"
_C.LOSS.SEG.DICE_WEIGHT = 1.0
_C.LOSS.SEG.CE_WEIGHT = 1.0
_C.LOSS.SEG.SMOOTH = 1e-6
_C.LOSS.SEG.IGNORE_INDEX = 255
_C.LOSS.SEG.FOCAL_GAMMA = 2
_C.LOSS.SEG.FOCAL_ALPHA = 0.5
_C.LOSS.SEG.TVERSKY_BETA = 0.5
_C.LOSS.SEG.TVERSKY_ALPHA = 1e-5
# ---------------------------------------------------------------------------- #
# TASK WEIGHT (cls / seg)
# ---------------------------------------------------------------------------- #
_C.TASK_WEIGHT = CN()
_C.TASK_WEIGHT.START_EPOCH = 0
_C.TASK_WEIGHT.END_EPOCH = 100

# ---------------------------------------------------------------------------- #
# TEST
# ---------------------------------------------------------------------------- #
_C.TEST = CN()
_C.TEST.DO_TEST = False
_C.TEST.INTERVAL = 5

# ---------------------------------------------------------------------------- #
# OUTPUT
# ---------------------------------------------------------------------------- #
_C.OUTPUT_DIR = None

_C.NOTE = ""

def get_cfg():
    return _C.clone()
