from tinygrad.tensor import Tensor
from tinygrad.helpers import dtypes
import numpy as np

# t = Tensor([0b0011_0010_0001_0000, 0b0111_0110_0101_0100, 0b1011_1010_1001_1000, 0b1111_1110_1101_1100], dtype=dtypes.uint16).apply_quant_scaling(4, 1.0, 0.0)
# print(t.numpy())

# t = Tensor([0b0011_0010_0001_0000, 0b0111_0110_0101_0100, 0b1011_1010_1001_1000, 0b1111_1110_1101_1100], dtype=dtypes.uint16).apply_quant_scaling(4, -1.0, 0.0)
# print(t.numpy())

# t = Tensor([0b0011_0010_0001_0000, 0b0111_0110_0101_0100, 0b1011_1010_1001_1000, 0b1111_1110_1101_1100], dtype=dtypes.uint16).apply_quant_scaling(4, -1.0, 16.0)
# print(t.numpy())


# t = Tensor([ [0b0011_0010_0001_0000, 0b0111_0110_0101_0100, 0b1011_1010_1001_1000, 0b1111_1110_1101_1100] for _ in range(3) ], dtype=dtypes.uint16).apply_quant_scaling(
#   4, [1.0, -1.0, -1.0], [0.0, 0.0, 16.0]
# )
# print(t.numpy())


# t = Tensor([ [0b0111_0110_0101_0100_0011_0010_0001_0000, 0b1111_1110_1101_1100_1011_1010_1001_1000] for _ in range(3) ], dtype=dtypes.uint32).apply_quant_scaling(
#   4, [1.0, -1.0, -1.0], [0.0, 0.0, 16.0]
# )
# print(t.numpy())

N = 139811
scale  = Tensor(np.random.randn(N, 1), dtype=dtypes.float16)
bias   = Tensor(np.random.randn(N, 1), dtype=dtypes.float16)
inputs = Tensor(np.random.randn(N, 5))

e = (N,5,3,)
output = inputs.reshape(N,5,1).expand(e).mul(scale.reshape(N,1,1).expand(e)).add(bias.reshape(N,1,1).expand(e))

output.realize()
