from tinygrad.tensor import Tensor
from tinygrad.helpers import DType, dtypes, prod
from extra.utils import get_child
from typing import Dict, Any, Callable, Optional
import numpy as np
import pickle, os

SkipLoadFxn_t  = Callable[[str, Tensor, np.ndarray], bool]
SkipQuantFxn_t = Callable[[str, Tensor, np.ndarray], bool]
AssignFxn_t    = Callable[[Tensor, Tensor], Any]

def load_direct(
  mdl:Tensor, state_dict:Dict[str, Any], # from model
  skip_load_fxn:SkipLoadFxn_t=lambda k, o, d: False, skip_quant_fxn:SkipQuantFxn_t=lambda k, o, d: False, assign_fxn:AssignFxn_t=lambda o, d: o.assign(d), # optional overwrites for model
):
  for k, v in state_dict.items():
    obj = get_child(mdl, k)
    dat = v.detach().numpy()

    if skip_load_fxn(k, obj, dat):
      continue

    assign_fxn(obj, dat)

def quantize_std_scalar(
  mdl:Tensor, state_dict:Dict[str, Any], # from model
  bits:int, group_size:int=64, store_dtype:DType=dtypes.uint32, sigma:float=2.0, cache_dirpath:Optional[str]=None, # from caller through kwargs
  skip_load_fxn:SkipLoadFxn_t=lambda k, o, d: False, skip_quant_fxn:SkipQuantFxn_t=lambda k, o, d: False, assign_fxn:AssignFxn_t=lambda o, d: o.assign(d), # optional overwrites for model
):
  assert bits >= 2 and bits <= 8, f"bits must be between 2 and 8, got {bits}"
  loop = (store_dtype.itemsize*8 // bits)
  assert group_size > loop and group_size % loop == 0, f"group size must be greater than and multiple of loop ({loop}), got {group_size}"
  N = 2**bits

  if cache_dirpath is not None:
    cache_dirpath = f"{cache_dirpath}/q{bits}"
    if not os.path.exists(cache_dirpath):
      os.makedirs(cache_dirpath)

  for k, v in state_dict.items():
    obj = get_child(mdl, k)
    dat = v.detach().numpy()

    if skip_load_fxn(k, obj, dat):
      continue

    if skip_quant_fxn(k, obj, dat):
      assign_fxn(obj, dat)
    elif prod(dat.shape) % loop != 0:
      print(f"Skipping quantizing {k}, prod({dat.shape})={prod(dat.shape)}, loop={loop}, {prod(dat.shape)}%{loop}={prod(dat.shape) % loop}")
      assign_fxn(obj, dat)
    else:
      qinfo = []
      cache_filepath = None
      if cache_dirpath is not None:
        if not os.path.exists(cache_dirpath):
          os.makedirs(cache_dirpath)
        cache_filepath = f"{cache_dirpath}/{k}"
        if os.path.exists(cache_filepath):
          with open(cache_filepath, 'rb') as f:
            qinfo = pickle.load(f)
      
      if len(qinfo) == 0:
        grouped_dat = dat.reshape(-1, group_size)

        # min_vals = np.min(grouped_dat, axis=1)
        # max_vals = np.max(grouped_dat, axis=1)
        # bias  = ( min_vals                  ).reshape(-1, 1)
        # scale = ( (max_vals - min_vals) / N ).reshape(-1, 1)

        averages = np.mean(grouped_dat, axis=1)
        std_devs = np.std(grouped_dat, axis=1)
        bias  = (averages - (std_devs * sigma)).reshape(-1, 1)
        scale = (std_devs * (2 * sigma / (N + 1))).reshape(-1, 1)

        index_dat = np.clip(np.round((grouped_dat - bias) / scale).astype(np.int16), 0, N-1).astype(store_dtype.np)

        qdat = np.zeros((grouped_dat.shape[0],group_size//loop), store_dtype.np)
        for i in range(loop):
          qdat[:] += index_dat[:,i:i+group_size:loop] << i*bits
        qinfo = [qdat, scale, bias]

        if cache_filepath is not None:
          with open(cache_filepath, 'wb') as f:
            pickle.dump(qinfo, f)
      
      qdat, scale, bias = qinfo
      qt = Tensor(qdat).apply_quant_scaling(bits, scale, bias, target_shape=dat.shape).fill_temp()

      assign_fxn(obj, qt)
