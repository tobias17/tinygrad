from tinygrad.tensor import Tensor
from tinygrad.helpers import dtypes, GlobalCounters
from models.resnet import ResNet50
from extra.datasets.imagenet import iterate
from extra.helpers import cross_process
from extra.quantize import quantize_k_means
import time
import numpy as np

np.random.seed(1)
mdl = ResNet50().load_from_pretrained(loader=quantize_k_means, bits=4)

input_mean = Tensor([0.485, 0.456, 0.406]).reshape(1, -1, 1, 1)
input_std = Tensor([0.229, 0.224, 0.225]).reshape(1, -1, 1, 1)
def input_fixup(x):
  x = x.permute([0,3,1,2]).cast(dtypes.float32) / 255.0
  x -= input_mean
  x /= input_std
  return x

mdlrun = lambda x: mdl(input_fixup(x)).realize()

BS = 1
n,d = 0,0
st = time.perf_counter()
iterator = cross_process(lambda: iterate(BS, shuffle=False))
x,ny = next(iterator)
dat = Tensor(x)

tot_ops = 0
tot_tm = 0

for i in range(32):
  y = ny
  GlobalCounters.reset()
  mt = time.perf_counter()
  outs = mdlrun(dat)
  try:
    x,ny = next(iterator)
    dat = Tensor(x)
  except StopIteration:
    dat = None
  t = outs.argmax(axis=1).numpy()
  et = time.perf_counter()
  n += (t==y).sum()
  d += len(t)
  if i > 1:
    tot_ops += GlobalCounters.global_ops
    tot_tm += (et-mt)
  print(f"****** {n}/{d}  {n*100.0/d:.2f}% -- {(mt-st)*1000:.2f} ms loading data, {(et-mt)*1000:7.2f} ms to run model. {len(t)/(et-mt):.2f} examples/sec. {GlobalCounters.global_ops*1e-12/(et-mt):.2f} TFLOPS")
  st = time.perf_counter()

print(f"Average: {tot_ops*1e-12/tot_tm:.2f} TFLOPS")
