import os
import sys

from datetime import datetime as dt
from pathlib import Path

def rel2abs(dir_path):
    return str(Path(dir_path).resolve())

def jnt(*args, **kwargs):

    if len(kwargs) == 0 or kwargs.get('checkparts') == False: return rel2abs(os.path.join(*args))

    result = args[0]
    for arg in args[1:]:
        if os.path.exists(os.path.join(result, arg)) or '.' in arg:
            result = os.path.join(result, arg)
        else:
            fnames = lst(result)
            for fname in fnames[::-1]:
                if fname.startswith(arg):
                    result = os.path.join(result, fname)
                    break
    return rel2abs(result)

def lst(dir_path, **kwargs):
    fnames_tmp = os.listdir(dir_path)
    fnames_ref = []
    for fname in fnames_tmp:
        if '__' in fname:
            fnames_ref.append(dt.strptime(fname.split('__')[-1], '%Y_%b_%d_%H_%M_%S').strftime('%Y_%m_%d_%H_%M_%S'))
        else:
            fnames_ref.append(fname)
    fnames = [fname for _, fname in sorted(zip(fnames_ref, fnames_tmp))]
    if len(kwargs) == 0: return fnames
    fmt = kwargs.get('fmt')
    getfmt = kwargs.get('getfmt')
    if getfmt: return [fname for fname in fnames if fname.endswith(fmt)]
    return [fname.split('.')[0] for fname in fnames if fname.endswith(fmt)]

def mkd(dir_path):
    Path(dir_path).resolve().mkdir(parents=True, exist_ok=True)



