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
  
  # def test_large_complex(self):
  #   self.bits = 4
  #   count = 3

  #   np_arrs = []
  #   for _ in range(count*2):
  #     np_arrs.append(self.rand_inputs((4, 4, 2,)))
  #   self.mapping = np.array(np.random.rand(2**self.bits), dtype=self.out_dtype)
    
  #   qt_arrs = []
  #   mapping_t = Tensor(self.mapping)
  #   for np_arr in np_arrs:
  #     qt_arrs.append(Tensor(np_arr).apply_quant_map(mapping_t))

  #   np_arrs = [self.dequantize_numpy(np_arr) for np_arr in np_arrs]

  #   for i in range(count):
  #     np_arrs[2*i] = np_arrs[2*i] * np_arrs[2*i+1]
  #     qt_arrs[2*i] = qt_arrs[2*i] * qt_arrs[2*i+1]
  #   for i in range(1, count):
  #     np_arrs[0] += np_arrs[2*i]
  #     qt_arrs[0] += qt_arrs[2*i]
    
  #   np.testing.assert_allclose(np_arrs[0], qt_arrs[0].numpy(), rtol=1e-05)

  def test_bit_range(self):
    for b in range(2, 9):
      self.bits = b
      a = self.rand_inputs((4*b,))
      b = self.rand_inputs((4*b,))
      self.mapping = np.array(np.random.rand(2**self.bits), dtype=self.out_dtype)
      expected_res = self.dequantize_numpy(a) + self.dequantize_numpy(b)

      mapping_t = Tensor(self.mapping)
      actual_res = (Tensor(a).apply_quant_map(mapping_t) + Tensor(b).apply_quant_map(mapping_t)).numpy()

      np.testing.assert_allclose(expected_res, actual_res)

  def test_cat(self):
    self.bits = 4
    count = 8
    size = 4

    np_arr = self.rand_inputs((count, size,))
    self.mapping = np.array(np.random.rand(2**self.bits), dtype=self.out_dtype)
    
    mapping_t = Tensor(self.mapping)
    tensors = [Tensor(np_arr[i]).apply_quant_map(mapping_t) for i in range(count)]
    t_cat = tensors[0].cat(*tensors[1:])
    # t_cat = Tensor.stack(tensors).reshape(-1)
    
    expected = self.dequantize_numpy(np_arr.flatten())
    actual = t_cat.numpy()
    np.testing.assert_allclose(expected, actual)

  def test_temp(self):
    pass

if __name__ == "__main__":
  unittest.main()
