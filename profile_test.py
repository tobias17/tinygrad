from tinygrad.tensor import Tensor
from tinygrad.helpers import dtypes, prod
from tqdm import trange
import numpy as np
import time
import matplotlib.pyplot as plt

def profile_function(fnx, label, count=64, warmup=4):
  times = np.zeros((count,))

  for _ in range(warmup):
    fnx()

  for i in trange(count):
    s_time = time.time()
    fnx()
    times[i] = (time.time() - s_time) * 1000
  
  v1, v2 = f"{np.average(times):.2f}", f"{np.std(times):.2f}"
  m = min(len(v1), len(v2))
  v1, v2 = " "*(len(v2)-m)+v1, " "*(len(v1)-m)+v2

  print(f"{label}:")
  print(f"Avg {v1} ms")
  print(f"Std {v2} ms\n")

  # plt.hist(times, bins=10, range=(0, max(times)), edgecolor='black')
  # plt.xlabel('Time (ms)')
  # plt.ylabel('Frequency')
  # plt.title('Time Histogram')
  # plt.show()
  return np.average(times), np.std(times)

SIZE = 1000000

def test_cat(count=20):
  t = []
  amt = SIZE // count
  for _ in range(count):
    t.append(Tensor(np.random.randn(amt)) + Tensor(np.random.randn(amt)))
  
  t = t[0].cat(*t[1:])
  t.realize()

X, Y = [], []
for count in [1, 50, 100, 150, 200, 250]:
  X.append(count)
  Y.append(profile_function(lambda: test_cat(count), str(count)))

averages = [item[0] for item in Y]
std_devs = [item[1] for item in Y]

plt.errorbar(X, averages, yerr=std_devs, fmt='o', capsize=5)
plt.xlabel('Split Count')
plt.ylabel('Time (ms)')
plt.title('1,000,000 fp adds')
plt.ylim(0)  # Set the Y-axis lower limit to 0
plt.xlim(0)  # Set the Y-axis lower limit to 0
plt.show()
