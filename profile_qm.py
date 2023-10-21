from tinygrad.tensor import Tensor
from tinygrad.helpers import dtypes, prod
from tqdm import trange
import numpy as np
import time
import matplotlib.pyplot as plt

def profile_function(fnx, count=64):
  times = np.zeros((count,))

  for i in trange(count):
    s_time = time.time()
    fnx()
    times[i] = (time.time() - s_time) * 1000
  
  v1, v2 = f"{np.average(times):.2f}", f"{np.std(times):.2f}"
  m = min(len(v1), len(v2))
  v1, v2 = " "*(len(v2)-m)+v1, " "*(len(v1)-m)+v2

  print(f"Avg {v1} ms")
  print(f"Std {v2} ms")

  plt.hist(times, bins=10, range=(0, max(times)), edgecolor='black')
  plt.xlabel('Time (ms)')
  plt.ylabel('Frequency')
  plt.title('Time Histogram')
  plt.show()


OUT_DTYPE = dtypes.float32
IN_DTYPE = dtypes.uint16

REALIZE_COUNT = 4
COUNT = 1000000
SPLIT = 200
BITS  = 4
LOOP  = (OUT_DTYPE.itemsize*8) // BITS
COMP  = (IN_DTYPE.itemsize*8)  // BITS

def rand_inputs(shape) -> np.ndarray:
  return np.array(np.random.randint(0, 2**(IN_DTYPE.itemsize*8), size=shape), dtype=IN_DTYPE.np)

def get_inputs():
  qinfo = []
  for _ in range(SPLIT):
    size = COUNT // SPLIT // COMP
    qdata = rand_inputs((size,))
    v = np.random.rand(2**BITS)
    mapping = np.array(v, dtype=OUT_DTYPE.np)
    qinfo.append((qdata, mapping,))
  return qinfo

def load_mappings_split():
  qinfo = get_inputs()
  tensors = []
  for qdata, mapping in qinfo:
    tensors.append(Tensor(qdata, dtype=IN_DTYPE).apply_quant_map(Tensor(mapping)))
  qt = tensors[0].cat(*tensors[1:])
  assert prod(qt.shape) == COUNT

  for _ in range(REALIZE_COUNT):
    qt.realize().lazydata.cleanse()

def load_mappings_comb():
  qinfo = get_inputs()
  sz = COUNT // SPLIT // COMP

  all_np = np.zeros((COUNT // COMP,))
  for i in range(SPLIT):
    all_np[i*sz:(i+1)*sz] = qinfo[i][0]
  all_t = Tensor(all_np, dtype=IN_DTYPE)

  tensors = []
  for i, (_, mapping) in enumerate(qinfo):
    tensors.append(all_t[i*sz:(i+1)*sz].apply_quant_map(Tensor(mapping)))
  qt = tensors[0].cat(*tensors[1:])
  assert prod(qt.shape) == COUNT

  if REALIZE_COUNT > 0:
    qt.realize().lazydata.cleanse()

  for _ in range(REALIZE_COUNT):
    qt.realize().lazydata.cleanse()

profile_function(load_mappings_comb)
