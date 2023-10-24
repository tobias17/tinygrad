from tinygrad.tensor import Tensor
import numpy as np
import unittest

class TestQuantization(unittest.TestCase):
  def setUp(self) -> None:
    np.random.seed(1)
    self.in_dtype = np.uint16
    self.out_dtype = np.float32
    return super().setUp()

  def rand_inputs(self, shape) -> np.ndarray:
    return np.array(np.random.randint(0, 2**(self.in_dtype().itemsize*8), size=shape), dtype=self.in_dtype)

  def dequantize_numpy(self, arr: np.ndarray):
    flat = arr.flatten()
    loop = (self.in_dtype().itemsize*8) // self.bits
    out = np.zeros((flat.shape[0]*loop,))
    for i in range(loop):
      out[i:i+out.shape[0]:loop] = self.mapping[flat[:] >> (i*self.bits) & sum(1<<j for j in range(self.bits))]
    return out.reshape(*arr.shape[:-1],arr.shape[-1]*loop)

  def test_small_add(self):
    self.bits = 4
    a = self.rand_inputs((10,))
    b = self.rand_inputs((10,))
    self.mapping = np.array(np.random.rand(2**self.bits), dtype=self.out_dtype)
    expected_res = self.dequantize_numpy(a) + self.dequantize_numpy(b)

    mapping_t = Tensor(self.mapping)
    actual_res = (Tensor(a).apply_quant_map(mapping_t) + Tensor(b).apply_quant_map(mapping_t)).numpy()

    np.testing.assert_allclose(expected_res, actual_res)

if __name__ == "__main__":
  # unittest.main()
  t = TestQuantization()
  t.setUp()
  t.test_small_add()
