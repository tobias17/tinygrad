from typing import List
from models.resnet import ResNet50
from tinygrad.tensor import Tensor
from tinygrad.ops import LoadOps, Device, Compiled, get_lazyop_info
from tinygrad.codegen.linearizer import Linearizer
from tinygrad.features.search import time_linearizer, beam_search
from tinygrad.helpers import ansilen, DEBUG, getenv, dtypes, prod
from tinygrad.graph import print_tree
from tinygrad.lazy import vars_from_ast, LazyBuffer
from tinygrad.shape.symbolic import sym_infer
from tinygrad.runtime.ops_cuda import renderer
import numpy as np

Device.DEFAULT = "CUDA"
print(f"Using device: {Device.DEFAULT}")

def print_program(tensor):
  seen = set()
  device: Compiled = Device[Device.DEFAULT]

  sched = tensor.lazydata.schedule(seen)
  sched = [x for x in sched if x.ast.op not in LoadOps]

  for i,si in enumerate(sched):
    # if i != 4: continue

    lin = Linearizer(si.ast, device.linearizer_opts)
    lin.linearize()
    lin.hand_coded_optimizations()
    # print("\n".join(map(str, lin.uops)))

    print_tree(si.ast)

    # prg = renderer("test_func", lin.uops)
    # print(prg[0])


# from models.resnet import ResNet50
# mdl = ResNet50()
# seen = set()
# mdl(Tensor.empty(64, 3, 224, 224)).lazydata.schedule(seen)

# x = Tensor.empty(64, 3, 224, 224)
# out = mdl(x)
# sched = out.lazydata.schedule(seen)
# sched = [x for x in sched if x.ast.op not in LoadOps]

# from examples.gpt2 import GPT2
# mdl = GPT2.build()

# mdl.greedy_until("The", 20, 0.5)

# x = 0

# for si in sched:
#   for inp in si.inputs:
#     if (m:=inp.st.views[-1].mask):
#       print(m)

# t = 0

# a = Tensor(np.random.randn(10, 10, 10))
# b = Tensor(np.random.randn(10, 10, 10))
# c = Tensor(np.random.randn(10, 10, 10))
# d = Tensor(np.random.randn(10, 10, 10))
# t = (a + b).cat((c + d), dim=-1)

# seen = set()
# sched = t.lazydata.schedule(seen)
# sched = [x for x in sched if x.ast.op not in LoadOps]

# def get_flop_breakdown(buffer: LazyBuffer):
#   sched = [x for x in buffer.schedule() if x.ast.op not in LoadOps]
#   if len(sched) == 0: return 0, 0
#   assert len(sched) == 1
#   si, ast = sched[0], sched[0].ast
  
#   total_flops = useful_flops = get_lazyop_info(ast).flops

#   total_size, mask_size = 0, 0
#   for b in buffer.op.buffers:
#     total_size += prod(b.shape)
#     if b.st.views[-1].mask:
#       mask_size += prod([x[1]-x[0] for x in b.st.views[-1].mask])
#   if mask_size > 0:
#     useful_flops = int(total_flops * mask_size / total_size)
#   for child in buffer.op.buffers:
#     a, b = get_flop_breakdown(child)
#     if a > 0:
#       u = 0

#   return total_flops, useful_flops

# for i,si in enumerate(sched):
#   total_flops = 0
#   for src in si.out.buffers:
#     total_flops, useful_flops = get_flop_breakdown(src)
#     print(f"{i}: {useful_flops} / {total_flops} ({100.0*useful_flops/total_flops:.2f}%)")
#     # new_flops = get_lazyop_info(src).flops
#     # total_flops += new_flops
#   X = 0
#   #   if (m:=inp.st.views[-1].mask):
#   #     print(m)

# t = Tensor([1.,2.]) + Tensor([3.,4.])

# print_program(t)
# t.numpy()




# t = [Tensor([1,2,3,4]) + Tensor([1,2,3,4]) for _ in range(4)]

# a = t[0].cat(*t[1:])

# print_program(a)
# print(a.numpy())


# t = (Tensor([4., 5.]) + Tensor([0., 4.])) * (Tensor([7., 2.]) + Tensor([6., 1.]))

t = [Tensor([1., 2., 3., 4.]) for _ in range(20)]
# for i in range(1, 100):
#   t[0] = t[0] + t[i]
# t = t[0]

t = t[0].cat(*t[1:])

# # t = Tensor.ones(1).expand((4,))

# t = t.pad(((100,100),))
# t += Tensor.ones(202)


# import time
# s_time = time.time()
# t.pad(((1000000,1000000),)).numpy()
# t.numpy()
# print(f"{1000*(time.time() - s_time):.4f} ms")

# print(t.shape)
print_program(t)
