from pycuda.compiler import compile as cuda_compile # type: ignore
from tqdm import tqdm
from uuid import uuid4
import random

with open("cache/_prg_2.c") as f:
  template = f.read()

def get_text():
  text = template.replace("%%FUNC_NAME%%", str(uuid4()).replace("-", "_"))
  while "%%OFFSET%%" in text:
    text = text.replace("%%OFFSET%%", str(random.randint(0, 100000)), 1)
  return text

texts = [get_text() for _ in range(100)]

for text in tqdm(texts):
  prg = cuda_compile(text, target="ptx", no_extern_c=True, options=['-Wno-deprecated-gpu-targets']).decode('utf-8')
