from tinygrad.tensor import Tensor
from tinygrad.helpers import dtypes, prod, DEBUG
from extra.utils import get_child, mem_to_string
import numpy as np
from sklearn.cluster import KMeans
from typing import Dict, Any, Callable, Optional
from tqdm import tqdm, trange
import pickle
import time
import os

QUANTIZE_COUNT = -1

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

def quantize_k_means(
  mdl:Tensor, state_dict:Dict[str, Any], # from model
  bits:int, groups:int=-1, check_values:bool=False, check_composition:bool=False, # from caller through kwargs
  skip_load_fxn:SkipLoadFxn_t=lambda k, o, d: False, skip_quant_fxn:SkipQuantFxn_t=lambda k, o, d: False, assign_fxn:AssignFxn_t=lambda o, d: o.assign(d), # optional overwrites for model
):
  assert bits >= 2 and bits <= 8, f"bits must be between 2 and 8, got {bits}"
  assert groups == -1 or groups >= 1, f"groups must be -1 or a positive integer, got {groups}"

  to_quantize = []
  store_dtype = dtypes.uint16
  loop = (store_dtype.itemsize*8 // bits)
  orig_size = 0
  count = 0

  # Determine which elements can be quantized
  for k, v in state_dict.items():
    obj = get_child(mdl, k)
    dat = v.detach().numpy()

    if skip_load_fxn(k, obj, dat):
      continue
    orig_size += prod(dat.shape)

    if skip_quant_fxn(k, obj, dat) or (QUANTIZE_COUNT >= 0 and count >= QUANTIZE_COUNT):
      assign_fxn(obj, dat)
    elif prod(dat.shape) % loop != 0:
      if DEBUG >= 1: print(f"Skipping quantizing {k}, prod({dat.shape})={prod(dat.shape)}, loop={loop}, {prod(dat.shape)}%{loop}={prod(dat.shape) % loop}")
      assign_fxn(obj, dat)
    else:
      to_quantize.append((obj, dat,))
      count += 1
  
  if len(to_quantize) == 0:
    if DEBUG >= 1: print("Found 0 chunks to quantize, exiting early!")
    return

  # Compute groupings
  if groups == -1: groups = len(to_quantize)
  groupings = [ [] for _ in range(groups) ]
  if groups == 1:
    groupings[0] = to_quantize
  else:
    if groups > len(to_quantize):
      if DEBUG >= 1: print(f"group size {groups} exceeded quntization element {len(to_quantize)}, clamping value")
      groups = len(to_quantize)
    if groups == len(to_quantize):
      for i in range(len(to_quantize)):
        groupings[i].append(to_quantize[i])
    else:
      if DEBUG >= 1: print("Computing groups...")
      raise NotImplementedError()


  # Quantize elements
  if DEBUG >= 1: print("Quantizing elements...")
  N = 2**bits
  total_X_size = 0
  for index, grouping in enumerate(groupings):
    suffix = f" for group {index}" if groups > 1 else ""

    if DEBUG >= 2: print("Computing KMeans clusters"+suffix+"...")
    X = np.concatenate([e[1].flatten() for e in grouping]).reshape(-1, 1)
    total_X_size += prod(X.shape)
    kmeans = KMeans(N, n_init='auto')
    kmeans.fit(X)
    cluster_t = Tensor(kmeans.cluster_centers_.reshape(-1))

    if DEBUG >= 2: print("Applying quantization to dat"+suffix+"...")
    offset = 0
    for obj, dat in grouping:
      sz = prod(dat.shape)
      chunk = np.array(kmeans.labels_[offset:offset+sz], dtype=np.uint16)
      qdat = np.zeros((sz//loop,), dtype=store_dtype.np)
      for i in range(loop):
        qdat[:] |= (chunk[i:i+sz:loop] << ((i%loop)*bits))
      qt = Tensor(qdat, dtype=store_dtype).apply_quant_map(cluster_t, target_shape=dat.shape)
      qt.lazydata.temporary = True
      assign_fxn(obj, qt)


      ########################################
      # Optional checks - used for debugging #
      # ------------------------------------ #
      if check_composition:
        dat_flat = dat.flatten()
        X_chunk  = X[offset:offset+prod(dat.shape)].flatten()
        assert dat_flat.shape == X_chunk.shape, f"dat_flat and X_chunk must have the same shape, {dat_flat.shape} != {X_chunk.shape}"
        if not (dat_flat == X_chunk).all():
          print("dat_flat to X_chunk difference detected:")
          print(dat_flat)
          print(X_chunk)
          assert False, "the above must be the same"
        qtn = qt.numpy()
        qtnf = qtn.flatten()
        struct_std_dev = np.sqrt(np.square(qtn - dat).mean())
        flat_std_dev   = np.sqrt(np.square(qtnf - dat_flat).mean())
        assert struct_std_dev == flat_std_dev, f"structured and flattened arrays had different std dev, {struct_std_dev} != {flat_std_dev}"
      # ------------------------------------ #
      if check_values:
        failed = False
        qtnf = qt.numpy().flatten()
        assert qtnf.shape == chunk.shape, "shape mismatch between original and realized tensor"
        for i in range(dat.shape[0]):
          value = kmeans.cluster_centers_[chunk[i]]
          if qtnf[i] != value:
            print(f"{i: >5}: FAIL {qtnf[i]} != {value}")
            failed = True
          # else:
          #   print(f"{i: >5}: GOOD {qtnf[i]}")
        assert not failed
      ########################################

      offset += prod(dat.shape)
            
    assert offset == prod(X.shape), "somehow did not go through the full offset"+suffix

  if DEBUG >= 1:
    total_size = orig_size * X.dtype.itemsize
    skipped_mem = (orig_size - total_X_size) * X.dtype.itemsize
    quant_mem_orig = (total_X_size * X.dtype.itemsize)
    quant_mem_comp = (total_X_size * (bits / 8)) + (N * X.dtype.itemsize * groups)
    quant_red = 100-(quant_mem_comp/quant_mem_orig*100) if quant_mem_orig > 0 else 0
    total_red = 100-((skipped_mem+quant_mem_comp)/(skipped_mem+quant_mem_orig)*100)
    print("\nTheorectical:")
    print(f"      | Original | Size   | Compress | Reduct |")
    print(f"Skips | {mem_to_string(skipped_mem)} | {f'{100.0*skipped_mem/total_size:.2f}': >5}% | {mem_to_string(skipped_mem)} | {f'{0.0:.2f}': >5}% |")
    print(f"Quant | {mem_to_string(quant_mem_orig)} | {f'{100.0*quant_mem_orig/total_size:.2f}': >5}% | {mem_to_string(quant_mem_comp)} | {f'{quant_red:.2f}': >5}% |")
    print(f"Total | {mem_to_string(total_size)} | 100.0% | {mem_to_string(skipped_mem+quant_mem_comp)} | {f'{total_red:.2f}': >5}% |")



def quantize_k_means_sliced(
  mdl:Tensor, state_dict:Dict[str, Any], # from model
  bits:int, max_chunk_size:int=4e5, cache_dirpath:Optional[str]=None, check_values:bool=False, check_composition:bool=False, # from caller through kwargs
  skip_load_fxn:SkipLoadFxn_t=lambda k, o, d: False, skip_quant_fxn:SkipQuantFxn_t=lambda k, o, d: False, assign_fxn:AssignFxn_t=lambda o, d: o.assign(d), # optional overwrites for model
):
  assert bits >= 2 and bits <= 8, f"bits must be between 2 and 8, got {bits}"

  N = 2**bits
  store_dtype = dtypes.uint16
  loop = (store_dtype.itemsize*8 // bits)
  orig_size = 0
  count = 0

  # Determine which elements can be quantized
  for k, v in state_dict.items():
    obj = get_child(mdl, k)
    dat = v.detach().numpy()

    if skip_load_fxn(k, obj, dat):
      continue
    orig_size += prod(dat.shape)

    if skip_quant_fxn(k, obj, dat) or (QUANTIZE_COUNT >= 0 and count >= QUANTIZE_COUNT):
      assign_fxn(obj, dat)
    elif prod(dat.shape) % loop != 0:
      if DEBUG >= 1: print(f"Skipping quantizing {k}, prod({dat.shape})={prod(dat.shape)}, loop={loop}, {prod(dat.shape)}%{loop}={prod(dat.shape) % loop}")
      assign_fxn(obj, dat)
    elif True:
      qinfo = []
      cache_filepath = None
      if cache_dirpath is not None:
        if not os.path.exists(cache_dirpath):
          os.makedirs(cache_dirpath)
        cache_filepath = f"{cache_dirpath}/q{bits}.{k}"
        if os.path.exists(cache_filepath):
          with open(cache_filepath, 'rb') as f:
            qinfo = pickle.load(f)

      if len(qinfo) == 0:
        flat = dat.flatten() 
        chunk_count = 1
        chunk_size  = flat.shape[0]
        while chunk_size > max_chunk_size:
          chunk_count *= 2
          chunk_size = chunk_size // 2
          assert chunk_size * chunk_count == flat.shape[0]
          assert chunk_size % loop == 0

        for i in trange(chunk_count):
          X = flat[i*chunk_size:(i+1)*chunk_size].reshape((-1, 1,))
          kmeans = KMeans(N, n_init='auto')
          kmeans.fit(X)

          cluster = np.array(kmeans.cluster_centers_, dtype=X.dtype).reshape(-1)

          sz = X.shape[0]
          labels = np.array(kmeans.labels_, dtype=np.uint16)
          qdat = np.zeros((sz//loop,), dtype=store_dtype.np)
          for i in range(loop):
            qdat[:] |= (labels[i:i+sz:loop] << ((i%loop)*bits))
          qinfo.append((qdat, cluster,))
        
        if cache_filepath is not None:
          with open(cache_filepath, 'wb') as f:
            pickle.dump(qinfo, f)
      
      tensors = []
      for qdat, cluster in qinfo:
        cluster_t = Tensor(cluster, dtype=dtypes.from_np(cluster.dtype))
        qt = Tensor(qdat, dtype=store_dtype).apply_quant_map(cluster_t).add(1).temporary().sub(1).temporary()
        tensors.append(qt)
      
      tensor = tensors[0].cat(*tensors[1:]).temporary() if len(tensors) > 1 else tensors[0]
      tensor = tensor.reshape(dat.shape).temporary()
      tensor.realize().lazydata.cleanse()
      assign_fxn(obj, tensor)
    else:
      X = dat.reshape((-1, 1,))
      kmeans = KMeans(N, n_init='auto')
      kmeans.fit(X)

      cluster_t = Tensor(kmeans.cluster_centers_.reshape(-1), dtype=dtypes.from_np(X.dtype))

      sz = X.shape[0]
      labels = np.array(kmeans.labels_, dtype=np.uint16)
      qdat = np.zeros((sz//loop,), dtype=store_dtype.np)
      for i in range(loop):
        qdat[:] |= (labels[i:i+sz:loop] << ((i%loop)*bits))
      qt = Tensor(qdat, dtype=store_dtype).apply_quant_map(cluster_t, target_shape=dat.shape)
      qt.realize().lazydata.cleanse()
      assign_fxn(obj, qt)
