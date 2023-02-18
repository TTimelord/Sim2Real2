import gzip
import json
from json import JSONEncoder
from pathlib import Path
from typing import Union

import numpy as np


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


def load_json(filename: Union[str, Path]):
    filename = str(filename)
    if filename.endswith(".gz"):
        f = gzip.open(filename, "rt")
    elif filename.endswith(".json"):
        f = open(filename, "rt")
    else:
        raise RuntimeError(f"Unsupported extension: {filename}")
    ret = json.loads(f.read())
    f.close()
    return ret


def dump_json(filename: Union[str, Path], obj, **kwargs):
    filename = str(filename)
    if filename.endswith(".gz"):
        f = gzip.open(filename, "wt")
    elif filename.endswith(".json"):
        f = open(filename, "wt")
    else:
        raise RuntimeError(f"Unsupported extension: {filename}")
    json.dump(obj, f, cls=NumpyArrayEncoder, **kwargs)
    f.close()
