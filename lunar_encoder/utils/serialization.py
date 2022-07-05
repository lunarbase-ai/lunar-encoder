import json
from typing import Any

import numpy as np
import datetime


class NumpyJSONEncoder(json.JSONEncoder):
    """
    A JSON encoder object used to make the data inferred by the profilers
    compatible with Python's JSON serialization requirements.

    Not used at the moment.
    """

    def default(self, obj: Any) -> Any:
        if isinstance(
            obj,
            (
                np.int_,
                np.intc,
                np.intp,
                np.int8,
                np.int16,
                np.int32,
                np.int64,
                np.uint8,
                np.uint16,
                np.uint32,
                np.uint64,
            ),
        ):
            return int(obj)

        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)

        elif isinstance(obj, (np.complex_, np.complex64, np.complex128)):
            return {"real": obj.real, "imag": obj.imag}

        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()

        elif isinstance(obj, np.bool_):
            return bool(obj)

        elif isinstance(obj, np.void):
            return None

        elif isinstance(
            obj,
            (
                datetime.date,
                datetime.time,
                datetime.datetime,
                datetime.timedelta,
                datetime.tzinfo,
                datetime.timezone,
            ),
        ):
            return str(obj)

        return json.JSONEncoder.default(self, obj)