from tinygrad.tensor import Tensor
from tinygrad.helpers import dtypes, prod, GlobalCounters
from tqdm import trange
import numpy as np
import time
import matplotlib.pyplot as plt
import os

def profile_function(fnx, count=20, warmup=0, pause_time=0):
  times = np.zeros((count,))
  kernel_times = []

  for _ in range(warmup):
    fnx()

  for i in trange(count):
    s_time = time.time()
    t = fnx()
    times[i] = (time.time() - s_time) * 1000
    if pause_time > 0: time.sleep(pause_time)
    # times[i] = (kernel_times[i] / times[i])
    # if i == 4:
    #   GlobalCounters.iteration = 4
    #   t.realize()
    #   GlobalCounters.iteration = -1
  
  # times = np.array(kernel_times)

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

  # plt.bar([i+1 for i in range(count)], times)
  # plt.title("Bar Chart of Times")
  # plt.xlabel("Time Description")
  # plt.ylabel("Time Value")
  # plt.xticks(rotation=45)  # Rotate x-axis labels if necessary
  # plt.tight_layout()  # Ensure that labels fit within the figure
  # plt.ylim(0.0, 1.0)
  # plt.show()

  # labels = [str(i) for i in range(6)] + ['rem']
  # categories = [str(i+1) for i in range(count)]
  # all_values = []
  # for i in range(count):
  #   values = [kernel_times[i][j]/times[i] for j in range(len(kernel_times[i]))]
  #   # values.append(1-sum(values))
  #   # sum_values = [sum(values[:i+1]) for i in range(len(labels))]
  #   # all_values.append(values)
  #   print(" | ".join(f"[{i}]  {values[i]:.3f} " for i in range(len(values))) + f" | [r]  {1-sum(values):.3f}")

  # prev_data = None
  # for j in list(range(len(labels)))[::-1]:
  #   data = [all_values[i][j] for i in range(count)]
  #   plt.bar(categories, data, label=labels[j])
  #   prev_data = data
  
  # # Set labels and title
  # plt.xlabel('Categories')
  # plt.ylabel('Values')
  # plt.title('Stacked Bar Graph')
  # plt.legend()
  # plt.ylim(0.0, 1.0)
  # plt.show()

  # return np.average(times), np.std(times)


OUT_DTYPE = dtypes.float32
IN_DTYPE = dtypes.uint16

REALIZE_COUNT = 10
COUNT = 10000000
SPLIT = 20
BITS  = 4
LOOP  = (OUT_DTYPE.itemsize*8) // BITS
COMP  = (IN_DTYPE.itemsize*8)  // BITS

def rand_inputs(shape) -> np.ndarray:
  return np.array(np.random.randint(0, 2**(IN_DTYPE.itemsize*8), size=shape), dtype=IN_DTYPE.np)

def get_inputs(split=SPLIT):
  qinfo = []
  for _ in range(split):
    size = COUNT // split // COMP
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
  qt = tensors[0].cat(*tensors[1:]).fill_temp()
  assert prod(qt.shape) == COUNT

  for _ in range(REALIZE_COUNT):
    qt.realize().lazydata.cleanse()
  
  return qt

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
  qt = tensors[0].cat(*tensors[1:]).fill_temp()
  assert prod(qt.shape) == COUNT

  if REALIZE_COUNT > 0:
    qt.realize().lazydata.cleanse()

  for _ in range(REALIZE_COUNT):
    qt.realize().lazydata.cleanse()

for filename in os.listdir(f"cache"):
  if filename.startswith("prgs_"):
    os.remove(f"cache/{filename}")

def load_mapping_direct():
  qdata, mapping = get_inputs(split=1)[0]
  qt = Tensor(qdata, dtype=IN_DTYPE).apply_quant_map(Tensor(mapping))

  qt.realize()

  for _ in range(REALIZE_COUNT):
    qt.realize().lazydata.cleanse()

profile_function(load_mapping_direct)

# ITER_COUNT = 2

# data = []
# colors = []
# print("warmup")
# for _ in range(4):
#   profile_function(load_mappings_split)
# print("0 pause time")
# for _ in range(ITER_COUNT):
#   data.append(profile_function(load_mappings_split))
#   colors.append('g')
# print("0.2 pause time")
# for _ in range(ITER_COUNT):
#   colors.append('b')
#   data.append(profile_function(load_mappings_split, pause_time=0.2))
# print("3.0 pause time")
# for _ in range(ITER_COUNT):
#   colors.append('r')
#   data.append(profile_function(load_mappings_split, pause_time=3.0))

# averages = [item[0] for item in data]
# std_devs = [item[1] for item in data]

# # Bar positions
# x = np.arange(len(data))

# # Create the figure and axis objects
# fig, ax = plt.subplots()

# # Create the bar chart with error bars
# bars = ax.bar(x, averages, yerr=std_devs, color=colors, align='center', capsize=5)

# # Set labels and title
# ax.set_xlabel('Time (ms)')
# ax.set_ylabel('Values')
# ax.set_title('Bar Graph')

# # Set x-axis tick labels
# ax.set_xticks(x)
# ax.set_xticklabels([f'Item {i+1}' for i in range(len(data))])

# # Show the plot
# plt.show()


# values = [v * 1e6 for v in GlobalCounters.kernel_times]
# print(len(values))

# indices = list(range(len(values)))
# print(len(indices))

# # Create a scatterplot of values
# plt.scatter(indices, values, label='Values', color='b', marker='o', s=3)

# # Connect consecutive points with lines
# for i in range(1, len(values)):
#   plt.plot([indices[i - 1], indices[i]], [values[i - 1], values[i]], 'b-')

# # Set labels and title
# plt.xlabel('Kernel')
# plt.ylabel('Time (us)')
# plt.title('Scatterplot with Lines between Consecutive Points')

# # Set the x and y axis limits to start at 0
# plt.xlim(0)
# plt.ylim(0)  # You can adjust the maximum value as needed

# # Show the plot
# plt.show()


