from tinygrad.tensor import Tensor
from tinygrad.helpers import DType, dtypes, prod
from extra.utils import get_child
from typing import Dict, Any, Callable, Optional
import numpy as np
import pickle, os
from tqdm import tqdm

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
  bits:int, group_size:int=64, store_dtype:DType=dtypes.uint16, sigma:float=3.0, cache_dirpath:Optional[str]=None, # from caller through kwargs
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

  avg_dists = []

  for k, v in tqdm(state_dict.items()):
    obj = get_child(mdl, k)
    if not obj:
      continue
    dat = v.detach().numpy()

    if skip_load_fxn(k, obj, dat):
      continue

    if skip_quant_fxn(k, obj, dat):
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
        extra = prod(dat.shape) % group_size
        overflow = 0 if extra == 0 else group_size - extra

        grouped_dat = np.zeros(shape=((prod(dat.shape)+overflow),), dtype=dat.dtype)
        grouped_dat[:(-overflow if extra > 0 else prod(dat.shape))] = dat.flatten()
        grouped_dat = grouped_dat.reshape(-1, group_size)

        averages = np.mean(grouped_dat, axis=1)
        std_devs = np.std(grouped_dat, axis=1)
        if extra > 0: averages[-1] = np.mean(grouped_dat[-1,:extra])
        if extra > 0: averages[-1] = np.std(grouped_dat[-1,:extra])
        bias  = (averages - (std_devs * sigma)).reshape(-1, 1)
        scale = (std_devs * (2 * sigma / (N + 1))).reshape(-1, 1)

        index_dat = np.clip(np.round((grouped_dat - bias) / scale).astype(np.int16), 0, N-1).astype(store_dtype.np)

        qdat = np.zeros((index_dat.shape[0],group_size//loop), store_dtype.np)
        for i in range(loop):
          qdat[:] += index_dat[:,i:i+group_size:loop] << i*bits
        qinfo = [qdat, scale, bias]

        if cache_filepath is not None:
          with open(cache_filepath, 'wb') as f:
            pickle.dump(qinfo, f)
      
      qdat, scale, bias = qinfo
      a = Tensor(qdat)
      b = a.apply_quant_scaling(bits, scale, bias, target_shape=dat.shape)
      qt = b #b.fill_temp()

      # q_vals = qt.numpy()
      # delta = np.average(np.abs(q_vals - orig_dat_q.reshape(-1)[:prod(q_vals.shape)].reshape(q_vals.shape)))
      # print(delta)
      # qt = Tensor(qdat).apply_quant_scaling(bits, scale, bias, target_shape=dat.shape).fill_temp()

      assign_fxn(obj, qt)
