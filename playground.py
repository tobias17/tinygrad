from typing import List
from models.resnet import ResNet50
from tinygrad.tensor import Tensor
from tinygrad.ops import LoadOps, Device, Compiled
from tinygrad.codegen.linearizer import Linearizer
from tinygrad.features.search import time_linearizer, beam_search
from tinygrad.helpers import ansilen, DEBUG, getenv, dtypes
from tinygrad.graph import print_tree
from tinygrad.lazy import vars_from_ast
from tinygrad.shape.symbolic import sym_infer
from tinygrad.runtime.ops_cuda import renderer

Device.DEFAULT = "CUDA"
print(f"Using device: {Device.DEFAULT}")

def print_program(tensor):
  seen = set()
  device: Compiled = Device[Device.DEFAULT]

  sched = tensor.lazydata.schedule(seen)
  sched = [x for x in sched if x.ast.op not in LoadOps]

  for i,si in enumerate(sched):
    # if i != 4: continue
    if DEBUG >= 2: print_tree(si.ast)

    lin = Linearizer(si.ast, device.linearizer_opts)
    lin.linearize()
    lin.hand_coded_optimizations()
    # print("\n".join(map(str, lin.uops)))
    prg = renderer("test_func", lin.uops)

    print(prg[0])



# mapping = Tensor([0., 1., 2., 3.])

# a = Tensor([0b1110010011100100, 0b1110010011100100], dtype=dtypes.uint16).apply_quant_map(mapping)
# b = Tensor([0b0101010100000000, 0b1111111110101010], dtype=dtypes.uint16).apply_quant_map(mapping)

# c = a + b

# # print(a.numpy())
# # print(b.numpy())
# print(c.numpy())




# mapping = Tensor([20., 1., 2., 3.])
# t = [Tensor([0b1110010011100100], dtype=dtypes.uint16).apply_quant_map(mapping) for _ in range(4)]
# a = t[0].cat(*t[1:])

# print_program(a)

# print(a.numpy())


# data0 = [i for i in range(128)]
# data1 = [i for i in range(128)]
# data2 = [i for i in range(128)]
# for i in range(32):
#   gidx0 = i
#   alu0 = (gidx0 < 8)
#   val0 = data1[0] if alu0 else 0
#   val1 = data2[gidx0] if alu0 else 0
#   print(f"val0: {f'{str(val0)}     '[:5]}, val1: {f'{str(val1)}     '[:5]}, int1: {f'{str(val1 + (-1 if not alu0 else 0))}     '[:5]}")
#   # print(f"")
#   # print(f"int1: {val1 + (-1 if not alu0 else 0)}")

# def new_cat(t: List[Tensor]) -> Tensor:
#   assert len(t) > 1, f"must provide atleast 2 tensors to cat"
#   assert all(t[0].shape[:-1] == t[i].shape[:-1] for i in range(1, len(t))), f"shapes must match until the final index"
#   assert all(t[0].dtype == t[i].dtype and t[0].device == t[i].device for i in range(1, len(t))), f"all tensors must have the same dtype and device"
#   new_tensor = Tensor.zeros(*(t[0].shape[:-1]), sum([x.shape[-1] for x in t]), dtype=t[0].dtype, device=t[0].device)
#   for i in range(len(t)):
#     prev_sum = sum(map(lambda x: x.shape[-1], t[:i]))
#     start_shape = tuple((0, t[0].shape[j]) for j in range(len(t[0].shape) - 1))
#     s_t = new_tensor.shrink( start_shape + ((prev_sum, prev_sum+t[i].shape[-1],),) )
#     a_t = s_t + t[i]
#     p_t = a_t.pad(tuple((0,0) for _ in range(len(t[0].shape) - 1)) + ((prev_sum, new_tensor.shape[-1]-t[i].shape[-1]-prev_sum,),))
#     new_tensor += p_t
#   return new_tensor

t = [Tensor([1,2,3,4]) + Tensor([1,2,3,4]) for _ in range(4)]
a = t[0].cat(*t[1:])
# a = new_cat(t)

# T = Tensor([float(i) for i in range(128)])
# a = T.shrink(((24, 32,),))

print_program(a)
# print(a.numpy())


# t = []
# for i in range(10):
#   # t.append(Tensor.full((4,), i))
#   t.append(Tensor([0., 1., 2., 3.]))

# x = t[0].cat(*t[1:])
# v = x.numpy()

# print(v)
# print(Tensor.stack(t).numpy())


