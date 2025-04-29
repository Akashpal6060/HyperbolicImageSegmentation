import os

CFG_DIR  = __file__
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(CFG_DIR)))
DSET_DIR = os.path.join(BASE_DIR, 'datasets')


def txt2dict(fn):
    """Reads a txt file like '0:road' → {0: 'road', …}."""
    with open(fn, 'r') as f:
        lines = f.read().splitlines()
    d = {}
    for line in lines:
        if not line.strip(): 
            continue
        idx, label = line.split(':', 1)
        d[int(idx)] = label.strip()
    return d


class DatasetConfig:
    """Base class for dataset‐specific settings."""
    _NAME = ''
    _DATASET_DIR = ''
    _DATA_DIR = ''
    _NUM_CLASSES = 0
    _NUM_TRAIN = 0
    _NUM_VALIDATION = 0
    _NUM_EPOCHS = 0
    _INITIAL_LEARNING_RATE = 0.0
    _JSON_FILE = ''
    _I2C_FILE = ''
    _I2C = {}
    _RGB_MEANS = [0.0, 0.0, 0.0]


class IDDAWConfig(DatasetConfig):
    _NAME = 'IDDAW'
    _DATASET_DIR = os.path.join(DSET_DIR, 'IDDAW')
    # Root folder containing “train/” and “val/”
    _DATA_DIR = _DATASET_DIR

    # you can tune these if you know the exact counts
    _NUM_CLASSES = 30
    _NUM_TRAIN = 3430         # (optional) fill in if you know how many train samples
    _NUM_VALIDATION = 475    # (optional)
    _NUM_EPOCHS = 50
    _INITIAL_LEARNING_RATE = 1e-4

    # Hierarchy & class‐map files
    _JSON_FILE = os.path.join(_DATASET_DIR, 'iddaw_hierarchy.json')
    _I2C_FILE  = os.path.join(_DATASET_DIR, 'iddaw_i2c.txt')
    _I2C       = txt2dict(_I2C_FILE)

    # mean pixel values (if you want to normalize)
    _RGB_MEANS = [123.68, 116.78, 103.94]


# Only include your dataset here:
DATASET_CFG_DICT = {
    'IDDAW': IDDAWConfig,
}
