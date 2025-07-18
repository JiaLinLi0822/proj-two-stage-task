#!/Users/lijialin/.julia/conda/3/aarch64/bin/python

from pybads import BADS
import numpy as np
import json
import sys
import logging

if len(sys.argv) < 2:
    print("EXCEPTION Missing config JSON", flush=True)
    sys.exit(1)
# print("GOT_CONF", sys.argv[1], flush=True)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, int64):
            return int(obj)
        elif isinstance(obj, np.float64):
            return float(obj)

        return json.JSONEncoder.default(self, obj)

def jsonify(obj):
    try:
        return json.dumps(obj, cls=NumpyEncoder)
    except Exception as e:
        logging.exception("Error converting json, falling back on string")
        return str(obj)

def target(x):
    print("REQUEST_EVALUATION", x.tolist())
    y = json.loads(input())
    if isinstance(y, list):
        return tuple(y)
    else:
        return y

try:
    conf = json.loads(sys.argv[1])
    args = [conf[k] for k in ["x0", "lower_bounds", "upper_bounds", "plausible_lower_bounds", "plausible_upper_bounds"]]

    if conf['options'].get('specify_target_noise'):
        conf['options']['uncertainty_handling'] = True
    bads = BADS(target, *args, options=conf["options"])
    result = bads.optimize()
    del result['fun']
    print("FINAL_RESULT", jsonify(result))
except:
    print("EXCEPTION")
    raise