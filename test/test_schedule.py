# this will be the new test_ops for the next level
# schedule confirms the right things are capable of fusing
# NOTE: this has overlap with external_test_opt.py

import unittest
import numpy as np
from typing import List, Optional, Union, cast
from tinygrad import nn, dtypes
from tinygrad.device import Device
from tinygrad.dtype import PtrDType
from tinygrad.shape.shapetracker import ShapeTracker
from tinygrad.shape.view import View
from tinygrad.tensor import Tensor
from tinygrad.ops import BinaryOps, MetaOps, UOp, UnaryOps, UOps, graph_rewrite
from tinygrad.helpers import AST_REWRITE, CI, DEBUG, FUSE_ARANGE, FUSE_CONV_BW, GlobalCounters, flatten, getenv, SPLIT_REDUCEOP
from tinygrad.codegen.kernel import Kernel, verify_ast
from tinygrad.engine.schedule import create_schedule, get_output_st, reduceop_fusor, st_fixup, ScheduleItem
from tinygrad.engine.realize import CompiledRunner, run_schedule
from test.helpers import assert_equiv_uops, is_dtype_supported, Context, timeit
from tinygrad.lazy import LazyBuffer, view_supported_devices
from extra.models.llama import precompute_freqs_cis

class KernelCountException(Exception): pass
def check_schedule(t:Union[Tensor, List[Tensor], LazyBuffer], allowed:int, to_prerealize:Optional[List[Tensor]]=None, filter_sink=True):
  if isinstance(t, Tensor): outs = t.lazydata.lbs
  elif isinstance(t, List): outs = flatten([r.lazydata.lbs for r in t])
  else: outs = [t]
  if to_prerealize:
    for pre in to_prerealize: pre.schedule()
  sched = create_schedule(outs)
  if filter_sink: sched = [s for s in sched if s.ast.op is UOps.SINK]
  if len(sched) != allowed: print(f"SCHEDULE ISSUE, expecting {allowed} got {len(sched)}")
  if len(sched) != allowed or DEBUG >= 3:
    for i, s in enumerate(sched):
      print("kernel", i+1)
      print(s.ast)
  if len(sched) != allowed: raise KernelCountException(f"{len(sched)=} != {allowed}")
  # test the (sink) ops linearize
  for s in sched:
    if s.ast.op is not UOps.SINK: continue
    l = Kernel(s.ast)
    l.hand_coded_optimizations()
    l.linearize()
  return sched

class TestSchedule(unittest.TestCase):
  def test_basic_binop_fusion(self):
    a = Tensor.empty(10)
    b = Tensor.empty(10)
    c = Tensor.empty(10)
    d = a+b+c
    check_schedule(d, 1)

  def test_basic_binop_fusion_deep(self):
    a = Tensor.empty(10)
    b = Tensor.empty(10)
    c = Tensor.empty(10)
    d = Tensor.empty(10)
    e = a+b+c+d
    check_schedule(e, 1)

  def test_mulacc_fusion(self):
    a = Tensor.empty(10)
    b = Tensor.empty(10)
    c = (a*b).sum()
    check_schedule(c, 1)

  def test_mulacc_relu_fusion(self):
    a = Tensor.empty(10)
    b = Tensor.empty(10)
    c = (a*b).sum().relu()
    check_schedule(c, 1)

  def test_binop_reshape_fusion(self):
    a = Tensor.empty(10)
    b = Tensor.empty(10)
    c = Tensor.empty(5,2)
    d = (a+b).reshape(5,2)+c
    check_schedule(d, 1)

  def test_binop_permute_fusion(self):
    a = Tensor.empty(2,5)
    b = Tensor.empty(2,5)
    c = Tensor.empty(5,2)
    d = (a+b).permute(1,0)+c
    check_schedule(d, 1)

  def test_constants_are_embedded(self):
    a = Tensor.empty(3,3) * 2
    check_schedule(a, 2, filter_sink=False)

  def test_binop_elu_fusion(self):
    a = Tensor.empty(10)
    b = a.elu()
    check_schedule(b, 1)

  def test_binop_reshape_reduce_fusion(self):
    a = Tensor.empty(100)
    b = Tensor.empty(100)
    c = (a+b).reshape(10, 10).sum(axis=0, keepdim=True)
    check_schedule(c, 1)

  def test_reduce_reshape_binop_fusion(self):
    a = Tensor.empty(10,10)
    b = Tensor.empty(10)
    c = a.sum(axis=0) + b
    check_schedule(c, 1)

  # not pushing permutes through reduces
  def test_reduce_permute_binop_fusion(self):
    a = Tensor.empty(10,10,10)
    b = Tensor.empty(10,10,1)
    c = a.sum(axis=0, keepdim=True).permute(2,1,0) + b
    with self.assertRaises(KernelCountException): check_schedule(c, 1)

  def test_binop_early_reshape_reduce_fusion(self):
    a = Tensor.empty(100)
    b = Tensor.empty(100)
    c = Tensor.empty(10,10)
    d = ((a+b).reshape(10,10) + c).sum(axis=0)
    check_schedule(d, 1)

  def test_diamond_folded(self):
    a = Tensor.empty(10)
    b = Tensor.empty(10)
    c = Tensor.empty(10)
    d = Tensor.empty(10)
    ab = a+b
    e = (ab+c) + (ab+d)
    check_schedule(e, 1)

  def test_cache_binaryop(self):
    a = Tensor.empty(10)
    b = Tensor.empty(10)
    c = a+b
    d = a+b
    check_schedule(d, 0, [c])

  # failing in new lazy
  def test_cache_binaryop_reshaped(self):
    a = Tensor.empty(10)
    b = Tensor.empty(10)
    c = a+b
    d = a.reshape(10,1)+b.reshape(10,1)
    with self.assertRaises(KernelCountException): check_schedule(d, 0, [c])

  # failing in new lazy
  def test_cache_binaryop_transpose(self):
    a = Tensor.empty(10,10)
    b = Tensor.empty(10,10)
    c = (a.T*b.T).T #.contiguous()
    d = a*b
    with self.assertRaises(KernelCountException): check_schedule(d, 0, [c])

  def test_cache_two_reduceops(self):
    a = Tensor.empty(10)
    b = a.sum()
    c = a.sum()
    bc = b+c
    check_schedule(bc, 1)

  def test_cache_reduce_parent(self):
    x = Tensor.empty(32)
    r0 = x.mean(axis=0, keepdim=True)
    r1 = (x - r0).sum(axis=0).div(2)
    out = r0 + r1
    schedule = check_schedule(out, 2)
    reduceops = [x for si in schedule for x in si.ast.parents if x.op is UOps.REDUCE_AXIS]
    assert len(reduceops) == 2

  def test_cache_reduce_multiple_children(self):
    x = Tensor.empty(32)
    y = Tensor.empty(4, 4)
    r0 = x.mean(axis=0, keepdim=True)
    r1 = (x - r0).sum(axis=0).div(2)
    out0 = r0 + y
    out1 = r1 + y
    schedule = check_schedule([out0, out1], 4)
    reduceops = [x for si in schedule for x in si.ast.parents if x.op is UOps.REDUCE_AXIS]
    assert len(reduceops) == 2

  def test_fold_double_unary(self):
    y = Tensor.empty(2)
    out = y.sum(keepdim=True).sqrt().__neg__()
    check_schedule(out, 1)

  #@unittest.skip("may want to reconsider this")
  def test_fold_batchnorm(self):
    with Tensor.train():
      img = Tensor.empty(1,32,4,4)
      bn = nn.BatchNorm2d(32, track_running_stats=False)
      out = bn(img)
      check_schedule(out, 3)

  def test_fold_conv_batchnorm_notrain(self):
    with Tensor.train(False):
      img = Tensor.empty(1,3,8,8)
      c1 = nn.Conv2d(3,32,3)
      bn = nn.BatchNorm2d(32, track_running_stats=True)
      out = bn(c1(img)).relu()
      check_schedule(out, 1, [c1.weight, c1.bias])

  def test_fold_conv_batchnorm_notrain_no_running_stats(self):
    with Tensor.train(False):
      img = Tensor.empty(1,3,8,8)
      c1 = nn.Conv2d(3,32,3)
      bn = nn.BatchNorm2d(32, track_running_stats=False)
      out = bn(c1(img)).relu()
      check_schedule(out, 4, [c1.weight, c1.bias])

  def test_fold_conv_batchnorm(self):
    with Tensor.train():
      img = Tensor.empty(1,3,8,8)
      c1 = nn.Conv2d(3,32,3)
      bn = nn.BatchNorm2d(32, track_running_stats=False)
      out = bn(c1(img)).relu()
      check_schedule(out, 4, [c1.weight, c1.bias])

  def test_fold_conv_batchnorm_optim(self):
    # this is too high
    for optim, cnt in [(nn.optim.Adam, 19), (nn.optim.SGD, 17)]:
      with self.subTest(optim=optim.__name__):
        with Tensor.train():
          img = Tensor.ones(1,3,4,4)
          c1 = nn.Conv2d(3,32,3)
          bn = nn.BatchNorm2d(32, track_running_stats=False)
          opt = optim(nn.state.get_parameters([c1, bn]))
          img_bn = bn(c1(img)).elu().sum()
          opt.zero_grad()
          img_bn.backward()
          check_schedule(opt.schedule_step(), cnt)

  def test_fold_conv_relu_backward(self):
    c1 = nn.Conv2d(3,16,3, bias=False)
    c1.weight.requires_grad = True
    img = Tensor.rand(2,3,64,64, requires_grad=True)

    # run
    c1(img).relu().mean().backward()
    assert img.grad is not None and c1.weight.grad is not None
    run_schedule(check_schedule([img.grad, c1.weight.grad], 7, filter_sink=False))

    # compare
    import torch
    c1_torch = torch.nn.Conv2d(3,16,3, bias=False)
    c1_torch.weight.requires_grad = True
    c1_torch.weight = torch.nn.Parameter(torch.tensor(c1.weight.numpy(), dtype=torch.float32))
    img_torch = torch.tensor(img.numpy(), requires_grad=True)
    c1_torch(img_torch).relu().mean().backward()
    assert img_torch.grad is not None and c1_torch.weight.grad is not None
    np.testing.assert_allclose(c1.weight.grad.numpy(), c1_torch.weight.grad.numpy(), atol=5e-4, rtol=1e-5)
    np.testing.assert_allclose(img.grad.numpy(), img_torch.grad.numpy(), atol=5e-4, rtol=1e-5)

  @unittest.skipUnless(is_dtype_supported(dtypes.half), "need half")
  def test_fold_conv_relu_backward_half(self):
    old_float = dtypes.default_float
    dtypes.default_float = dtypes.float16

    c1 = nn.Conv2d(3,16,3, bias=False)
    c1.weight.requires_grad = True
    img = Tensor.rand(2,3,64,64, requires_grad=True)

    # run
    c1(img).relu().mean().backward()
    assert img.grad is not None and c1.weight.grad is not None
    run_schedule(check_schedule([img.grad, c1.weight.grad], 7, filter_sink=False))
    dtypes.default_float = old_float

    # compare
    import torch
    c1_torch = torch.nn.Conv2d(3,16,3, bias=False, dtype=torch.half)
    c1_torch.weight.requires_grad = True
    c1_torch.weight = torch.nn.Parameter(torch.tensor(c1.weight.numpy(), dtype=torch.half))
    img_torch = torch.tensor(img.numpy(), requires_grad=True)
    c1_torch(img_torch).relu().mean().backward()
    assert img_torch.grad is not None and c1_torch.weight.grad is not None
    np.testing.assert_allclose(c1.weight.grad.numpy(), c1_torch.weight.grad.numpy(), atol=5e-4, rtol=1e-5)
    np.testing.assert_allclose(img.grad.numpy(), img_torch.grad.numpy(), atol=5e-4, rtol=1e-5)

  def test_fold_batchnorm_backward(self):
    with Context(FUSE_CONV_BW=1):
      with Tensor.train():
        x = Tensor.empty((2, 16, 8, 8)).contiguous()
        bn = nn.BatchNorm2d(16)
        bn.weight.requires_grad = bn.bias.requires_grad = x.requires_grad = True
        fw = bn(x).contiguous_backward().relu().contiguous()
        fw.sum().backward()
        # TODO: this is too many
        check_schedule([x.grad, bn.weight.grad, bn.bias.grad, fw], 10)

  def test_fold_conv_relu(self):
    c1 = nn.Conv2d(3,16,3)

    # run
    img = Tensor.ones(2,3,64,64)
    out = c1(img).relu()
    check_schedule(out, 1, [c1.weight, c1.bias])

  def test_fold_conv_relu_alt(self):
    img = Tensor.ones(1,4,8,8)
    c1 = nn.Conv2d(4, 4, kernel_size=3)
    c2 = nn.Conv2d(4, 4, kernel_size=3)
    img_conv = img.sequential([c1, Tensor.relu, c2, Tensor.relu])
    check_schedule(img_conv, 2, [*nn.state.get_parameters(c1), *nn.state.get_parameters(c2), img])

  def test_fold_conv_relu_nobias(self):
    img = Tensor.ones(1,4,8,8)
    c1 = nn.Conv2d(4, 4, kernel_size=3, bias=False)
    c2 = nn.Conv2d(4, 4, kernel_size=3, bias=False)
    out = img.sequential([c1, Tensor.relu, c2, Tensor.relu])
    check_schedule(out, 2, [c1.weight, c2.weight, img])

  def test_fold_conv_elu(self):
    c1 = nn.Conv2d(3,16,3)

    # run
    img = Tensor.rand(2,3,64,64)
    out = c1(img).elu()
    check_schedule(out, 1, [c1.weight, c1.bias, img])

  def test_fold_conv_elu_alt(self):
    img = Tensor.ones(1,4,8,8).contiguous()
    c1 = nn.Conv2d(4, 4, kernel_size=3)
    c2 = nn.Conv2d(4, 4, kernel_size=3)
    img_conv = img.sequential([c1, Tensor.elu, c2, Tensor.elu])
    check_schedule(img_conv, 2, [*nn.state.get_parameters(c1), *nn.state.get_parameters(c2), img])

  def test_two_sum(self):
    img = Tensor.empty(64,64)
    x = (img.sum(0) + img.sum(1))
    out = x.relu()
    del x    # is 3 without this
    check_schedule(out, 2)

  #@unittest.skip("failing in old lazy")
  def test_push_permute_through_reshape(self):
    a = Tensor.empty(16,16)
    b = Tensor.empty(16,16)
    c = (a+b).reshape(4,4,4,4).permute(2,3,0,1).contiguous()
    check_schedule(c, 1)

  #@unittest.skip("failing in old lazy")
  def test_push_permute_through_reshape_alt(self):
    a = Tensor.empty(4,4,4,4)
    b = Tensor.empty(4,4,4,4)
    c = (a+b).reshape(16,16).permute(1,0).contiguous()
    check_schedule(c, 1)

  def test_no_binop_rerun(self):
    a = Tensor.empty(16)
    b = Tensor.empty(16)
    c = a+b
    d = (a+b).reshape(16,1)
    check_schedule(d, 0, [c])

  def test_multi_permute_should_collapse(self):
    a = Tensor.empty(4,4,4,4)
    b = Tensor.empty(16)
    c = a.sum((0,1)).cast(dtypes.float16).permute(1,0).reshape(4,4,1).permute(1,0,2).reshape(16) + b
    check_schedule(c, 1)

  def test_fancy_reshape_fusion(self):
    a = Tensor.empty(10)
    b = Tensor.empty(10)
    c = a+b
    d = a.reshape(10,1)+b.reshape(10,1)
    out = c.sum() + d.sum()
    with self.assertRaises(KernelCountException): check_schedule(out, 1)

  def test_children_dont_push(self):
    a = Tensor.empty(10, 10, 1)
    b = Tensor.empty(10, 10, 1)
    d = (a+b).expand(10, 10, 10)
    e = (a+b).permute(2,1,0)
    f = d+e
    check_schedule(f, 2)

  # failing in new lazy
  def test_dont_fuse_binops_with_children(self):
    a = Tensor.empty(10)
    b = Tensor.empty(10)
    c = Tensor.empty(10)
    keep_me = a+b
    e = keep_me.sum() # noqa: F841 give keep_me a child (NOTE: BinaryOps won't be a child since it will instant fuse)
    d = keep_me+c
    with self.assertRaises(KernelCountException): check_schedule(d, 2)
    with self.assertRaises(KernelCountException): check_schedule(keep_me, 0, [d])

  #@unittest.skip("failing in old lazy")
  def test_permute_breaks_fusion(self):
    a = Tensor.empty(10, 10, 10)
    b = Tensor.empty(10, 10)
    c = (a.sum(axis=2) + b).permute(1,0)
    d = c.permute(1,0)
    check_schedule(d, 1)

  def test_some_permute_fusion(self):
    a = Tensor.empty(8192, 16)
    b = Tensor.empty(1, 16)
    d = (a.T + b.expand(8192, 16).T)
    c = a + b.expand(8192, 16)
    e = d.T
    check_schedule(c, 1)
    check_schedule(e, 1)

  def test_shrink_fuse(self):
    a = Tensor.empty(8192, 16)
    b = Tensor.empty(8192, 16)
    c = a * b
    d = Tensor.empty(1, 16)
    e = c[0] * d
    check_schedule(e, 1)

  def test_expand_nofuse(self):
    a = Tensor.empty(1, 16)
    b = Tensor.empty(1, 16)
    c = a * b
    d = Tensor.empty(8192, 16)
    e = c * d
    check_schedule(e, 2)

  # this is the failing case in openpilot...it's very simple like this
  def test_image_conv_fusion(self):
    w1 = Tensor.empty(16, 16, 1, 1)
    b1 = Tensor.empty(16)
    w2 = Tensor.empty(16, 16, 1, 1)
    b2 = Tensor.empty(16)
    w3 = Tensor.empty(16, 16, 1, 1)
    b3 = Tensor.empty(16)

    x = Tensor.empty(1, 16, 32, 32)
    x = base = x.image_conv2d(w1, b1)
    x = x.image_conv2d(w2, b2) + base
    x = x.image_conv2d(w3, b3)

    # NOOP, 3 convs, contiguous
    with self.assertRaises(KernelCountException): check_schedule(x, 5)

  def test_image_conv_fusion_minimal(self):
    b1 = Tensor.empty(16)
    b2 = Tensor.empty(16)
    def p(x): return x.permute(1,0).contiguous().reshape(32,16,1).expand(32,16,16).sum(axis=2).permute(1,0)

    x = Tensor.empty(16, 32)
    x = base = p(x) + b1.reshape(16,1)
    x = p(x)
    x = x + b2.reshape(16,1)
    x = x + base
    del base
    x = p(x)
    check_schedule(x, 4)

  def test_image_conv_fusion_more_minimal(self):
    b1 = Tensor.empty(16)
    def p(x): return x.permute(1,0).contiguous().reshape(32,16,1).expand(32,16,16).sum(axis=2).permute(1,0)

    x = Tensor.empty(16, 32)
    x = base = p(x) + b1.reshape(16,1)
    x = p(x)
    del base
    check_schedule(x, 3)

  def test_resnet_block(self):
    old_training = Tensor.training
    Tensor.training = False

    in_planes, planes = 64, 64
    conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
    bn1 = nn.BatchNorm2d(planes)
    conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, stride=1, bias=False)
    bn2 = nn.BatchNorm2d(planes)

    x = Tensor.empty(1, 64, 32, 32)
    out = bn1(conv1(x)).relu()
    out = bn2(conv2(out))
    out = (out + x).relu()
    check_schedule(out, 2, [conv1.weight, conv2.weight])
    Tensor.training = old_training

  def test_contiguous_while_contiguous(self):
    x = Tensor.empty(1, 64, 32, 32)
    out = x.contiguous()
    check_schedule(out, 1, filter_sink=False)

  def test_contiguous_while_not_contiguous(self):
    x = Tensor.empty(1, 64, 32, 32)
    out = x.permute(0,2,3,1).contiguous()
    check_schedule(out, 2, filter_sink=False)

  def test_fold_with_contiguous(self):
    a = Tensor.randn(16, 16, 16).realize()
    b = Tensor.randn(16, 16).realize()
    c = (a.sum(2).contiguous() + b).contiguous()
    check_schedule(c, 2)

  def test_double_from(self):
    x = Tensor([1,2,3,4])
    out = x.to('npy')
    check_schedule(out, 0, filter_sink=False)

  def test_pow_const_tensor_simplified(self):
    x = Tensor([1,2,3,4])
    # NOTE: this does not test ** Tensor(2) is simpler in ast than ** Tensor(2.5)
    out = x ** Tensor(2)
    check_schedule(out, 1)

  def test_pow_const_tensor_to_zero(self):
    x = Tensor([1,2,3,4])
    out = x ** Tensor(0)
    # NOTE: this is ConstBuffer 0 + ConstBuffer 1
    check_schedule(out, 0)

  def test_zero_size(self):
    x = Tensor.empty(2, 3, 0)
    out = x + 1
    check_schedule(out, 0, filter_sink=False)

  def test_reduce_permute_nofuse(self):
    x = Tensor.empty(32, 32, 32)
    y = Tensor.empty(32, 32)
    out = x.sum(axis=2).T+y
    check_schedule(out, 2)

  def test_two_elus_sum(self):
    x = Tensor.empty(32, 32)
    y = Tensor.empty(32, 32)
    out = x.sum(1).relu().elu() + y.sum(1).relu().elu()
    check_schedule(out, 2)

  # multireduce spec
  @unittest.skipUnless(SPLIT_REDUCEOP, "Testing split reducop requires SPLIT_REDUCEOP")
  def test_preserve_multistage_reduce(self):
    big_enough = getenv("REDUCEOP_SPLIT_THRESHOLD", 32768)
    x = Tensor.randn(big_enough).realize()
    out = (x - x.max(keepdim=True)).max()
    run_schedule(check_schedule(out, 4))
    np.testing.assert_allclose(out.numpy(), (x.numpy() - x.numpy().max(keepdims=True)).max())

  def test_multistage_reduce(self):
    x = Tensor.empty(32, 32, 32)
    out = x.sum(2).relu().sum(1)
    check_schedule(out, 2)

  def test_multistage_reduce_fork(self):
    x = Tensor.empty(32, 32, 32)
    x = x.sum(2)
    out2 = x + 1
    out = x.relu().sum(1) + out2[0]
    check_schedule(out, 2)

  # multireduce spec
  def test_example_matmul(self):
    x = Tensor.eye(64, requires_grad=True)
    y = Tensor.eye(64, requires_grad=True)
    z = y.matmul(x).sum()
    z.backward()
    out = x.grad.contiguous()
    run_schedule(check_schedule(out, 2))
    np.testing.assert_allclose(out.numpy(), np.ones((64,64)))

  def test_contiguous_add(self):
    x = Tensor.empty(32)
    y = Tensor.empty(32)
    z = Tensor.empty(32)
    out = (x+y).contiguous()+z
    check_schedule(out, 2)

  def test_double_sum_ref(self):
    x = Tensor.empty(32, 32, 32)
    x = x.sum(2)
    out = x + x[:, 4]
    check_schedule(out, 2)

  def test_reduce_shrink(self):
    x = Tensor.empty(32, 32)
    y = Tensor.empty(16)
    x = x.sum(1)
    x = x[:16]
    out = x + y
    check_schedule(out, 2)  # TODO: this should be 1

  # multireduce spec
  def test_multireduce_shrink(self):
    Tensor.manual_seed(0)
    a = Tensor.randn(32, 32).realize()
    b = Tensor.randn(32, 32).realize()
    c = Tensor.randn(16).realize()
    a_out = a.sum(1)
    a_out = a_out[:16]
    b_out = b.sum(1)
    b_out = b_out[:16]
    out = a_out + b_out + c
    # run_schedule(check_schedule(out, 2))  # TODO: this should be 1 (can we make it 1 with the new linearizer?)
    run_schedule(check_schedule(out, 3))
    np.testing.assert_allclose(out.numpy(), a.numpy().sum(axis=1)[:16] + b.numpy().sum(axis=1)[:16] + c.numpy(), atol=1e-4, rtol=1e-4)

  # broken due to const folding and two contiguous are different kernels
  def test_const_no_recompute(self):
    x = Tensor(2) + Tensor(2)
    y = Tensor(2) + Tensor(2)
    out = x.contiguous() + y.contiguous()
    with self.assertRaises(KernelCountException): check_schedule(out, 2, filter_sink=False)

  # multireduce spec
  def test_reduce_same_size(self):
    Tensor.manual_seed(0)
    a = Tensor.randn(4, 4).realize()
    out0 = a.sum() + 2
    out1 = a.sum() + 4
    out2 = out0 * out1
    run_schedule(check_schedule([out0, out1, out2], 1))
    np.testing.assert_allclose(out0.numpy(), out0_np:=a.numpy().sum()+2, atol=1e-4, rtol=1e-6)
    np.testing.assert_allclose(out1.numpy(), out1_np:=a.numpy().sum()+4, atol=1e-4, rtol=1e-6)
    np.testing.assert_allclose(out2.numpy(), out0_np*out1_np, atol=1e-4, rtol=1e-6)

  # multireduce spec
  def test_reduce_multiple_paths(self):
    Tensor.manual_seed(0)
    a = Tensor.randn(4, 4).realize()
    out0 = a.sum().exp2()
    # out1 has two paths to a.sum()
    out1 = a.sum() + out0
    run_schedule(check_schedule([out0, out1], 1))
    np.testing.assert_allclose(out0.numpy(), out0_np:=np.exp2(a.numpy().sum()), atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(out1.numpy(), a.numpy().sum()+out0_np, atol=1e-4, rtol=1e-6)

  # multireduce spec
  def test_multireduce_reduce_multiple_paths(self):
    Tensor.manual_seed(0)
    a = Tensor.randn(4, 4).realize()
    out0 = a.sum().exp2()
    out1 = a.sum() + out0
    b = (a + out0 + out1)
    out2 = b.sum().exp2()
    out3 = b.sum() + out2
    # run_schedule(check_schedule([out0, out1, out2, out3], 1))
    run_schedule(check_schedule([out0, out1, out2, out3], 2))
    np.testing.assert_allclose(out0.numpy(), np_out0:=np.exp2(a.numpy().sum()), atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(out1.numpy(), np_out1:=a.numpy().sum()+np_out0, atol=1e-4, rtol=1e-4)
    np_b = (a.numpy() + np_out0 + np_out1)
    np.testing.assert_allclose(out2.numpy(), np_out2:=np.exp2(np_b.sum()), atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(out3.numpy(), np_b.sum()+np_out2, atol=1e-4, rtol=1e-4)

  # multireduce spec
  def test_reduce_ext_reduce_child(self):
    Tensor.manual_seed(0)
    a = Tensor.randn(4, 4).realize()
    b = Tensor.randn(4, 4).realize()
    # b.sum() is not a descendant of the fused nodes
    out0 = a.sum() + b.sum() + 2
    out1 = a.sum() + b.sum() + 4
    # run_schedule(check_schedule([out0, out1], 1))
    run_schedule(check_schedule([out0, out1], 4))
    np.testing.assert_allclose(out0.numpy(), a.numpy().sum()+b.numpy().sum()+2, atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(out1.numpy(), a.numpy().sum()+b.numpy().sum()+4, atol=1e-4, rtol=1e-4)

  # multireduce spec
  def test_reduce_multiple_paths_midreduce(self):
    Tensor.manual_seed(0)
    a = Tensor.randn(4, 4).realize()
    r = a.sum()
    out0 = r.exp2()
    # reduce node in the indirect path from r to out2
    out1 = (a - out0).max()
    out2 = r + out1
    # run_schedule(check_schedule([r, out0, out1, out2], 1))
    run_schedule(check_schedule([r, out0, out1, out2], 4))
    np.testing.assert_allclose(r.numpy(), r_np:=a.numpy().sum(), atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(out0.numpy(), out0_np:=np.exp2(r_np), atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(out1.numpy(), out1_np:=(a.numpy() - out0_np).max(), atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(out2.numpy(), r_np + out1_np, atol=1e-4, rtol=1e-4)

  # multireduce spec
  def test_reduce_multiple_paths_midreduce_fused(self):
    Tensor.manual_seed(0)
    a = Tensor.randn(4, 4).realize()
    b = Tensor.randn(4, 4).realize()
    out0 = a.sum() + 4
    out1 = b.max() + out0*2
    out2 = a.sum() + out1
    # run_schedule(check_schedule([out0, out1, out2], 1))
    run_schedule(check_schedule([out0, out1, out2], 4))
    np.testing.assert_allclose(out0.numpy(), out0_np:=a.numpy().sum()+4, atol=1e-4, rtol=1e-6)
    np.testing.assert_allclose(out1.numpy(), out1_np:=b.numpy().max() + out0_np*2, atol=1e-4, rtol=1e-6)
    np.testing.assert_allclose(out2.numpy(), a.numpy().sum() + out1_np, atol=1e-4, rtol=1e-6)

  # multireduce spec
  def test_reduce_multiple_paths_midexpand(self):
    Tensor.manual_seed(0)
    a = Tensor.randn(4, 4).realize()
    b = Tensor.randn(4, 4, 4).realize()
    r = a.sum()
    out0 = r.exp2()
    # e1 is in the indirect path from a.sum() to out1
    e = b + out0
    out1 = r + e[0][0][0]
    # run_schedule(check_schedule([r, out0, out1, e], 3)) # 1 or 2 or 3? should be 1 (one reduce) but the different outputs might make it 3
    run_schedule(check_schedule([r, out0, out1, e], 4))
    np.testing.assert_allclose(r.numpy(), r_np:=a.numpy().sum(), atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(out0.numpy(), out0_np:=np.exp2(r_np), atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(e.numpy(), e_np:=b.numpy() + out0_np, atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(out1.numpy(), r_np + e_np[0][0][0], atol=1e-4, rtol=1e-4)

  # changed by multireduce
  def test_reduce_expand_child(self):
    Tensor.manual_seed(0)
    a = Tensor.randn((32, 32, 32)).realize()
    b = Tensor.randn((1, 16)).realize()
    out0 = a.sum() + 2
    out1 = a.sum() + b
    # run_schedule(check_schedule([out0, out1], 2))
    run_schedule(check_schedule([out0, out1], 4))
    np.testing.assert_allclose(out0.numpy(), a.numpy().sum()+2, atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(out1.numpy(), a.numpy().sum()+b.numpy(), atol=1e-4, rtol=1e-4)

  def test_reduce_shrink_child(self):
    a = Tensor.empty(100, 100)
    b = Tensor.empty(10,)
    c = a.sum() + b[0]
    d = a.sum() + 2
    check_schedule([c, d], 1)

  def test_reduce_multiple_paths_midshrink(self):
    a = Tensor.empty(4, 4)
    r = a.sum(axis=1)
    out0 = r.exp2()
    out1 = out0[0] + out0
    check_schedule([r, out0, out1], 3)

  def test_reduce_shrink_output(self):
    a = Tensor.empty(4, 4)
    r = a.sum(keepdim=True)
    out0 = r.exp2()
    out1 = out0[0] + Tensor.empty(1, )
    check_schedule([r, out0, out1], 3)

  # multireduce spec
  def test_std_multireduce_fusion(self):
    Tensor.manual_seed(0)
    x = Tensor.randn(4, 32).realize()
    out = x.std(-1)
    run_schedule(check_schedule(out, 2))
    np.testing.assert_allclose(out.numpy(), x.numpy().std(axis=-1, ddof=1), atol=1e-4, rtol=1e-4)

  # multireduce spec
  def test_argmin_multireduce_fusion(self):
    Tensor.manual_seed(0)
    x = Tensor.randn(4, 32).realize()
    out = x.argmin(-1)
    run_schedule(check_schedule(out, 3))
    np.testing.assert_equal(out.numpy(), x.numpy().argmin(axis=-1))

  # multireduce spec
  def test_argmax_multireduce_fusion(self):
    Tensor.manual_seed(0)
    x = Tensor.randn(4, 32).realize()
    out = x.argmax(-1)
    run_schedule(check_schedule(out, 3))
    np.testing.assert_equal(out.numpy(), x.numpy().argmax(axis=-1))

  # multireduce spec
  def test_scaled_dot_product_attention_multireduce_fusion(self):
    Tensor.manual_seed(0)
    q = Tensor.randn(32,8,16,64).realize()
    k = Tensor.randn(32,8,16,64).realize()
    v = Tensor.randn(32,8,16,64).realize()
    out = Tensor.scaled_dot_product_attention(q,k,v)
    check_schedule(out, 5) # correctness checked in test_ops

  # multireduce spec
  def test_ugly_reduceop_pairing(self):
    Tensor.manual_seed(0)
    a = Tensor.randn(4, 32).realize()
    b = Tensor.randn(4, 32).realize()
    c = Tensor.randn(4, 32).realize()
    out = (c * a.sum(-1, keepdim=True)).sum(-1) + (b * a.sum(-1, keepdim=True)).sum(-1) # a.sum has >1 children but should still fuse
    # run_schedule(check_schedule(out, 1))
    run_schedule(check_schedule(out, 3))
    np.testing.assert_allclose(out.numpy(), \
      (c.numpy()*a.numpy().sum(axis=-1,keepdims=True)).sum(-1) + (b.numpy()*a.numpy().sum(axis=-1,keepdims=True)).sum(-1), atol=1e-4, rtol=1e-4)

  # multireduce spec
  def test_reduce_expand_reduce_fusion(self):
    Tensor.manual_seed(0)
    a = Tensor.randn(4, 32).realize()
    out = (a+a.sum(-1, keepdim=True)).sum(-1)
    # run_schedule(check_schedule(out, 1))
    run_schedule(check_schedule(out, 2))
    np.testing.assert_allclose(out.numpy(), (a.numpy()+a.numpy().sum(axis=-1,keepdims=True)).sum(axis=-1), atol=1e-4, rtol=1e-4)

  # multireduce spec
  def test_reduce_expand_reduce_expand_fusion(self):
    Tensor.manual_seed(0)
    a = Tensor.randn(4, 32).realize()
    out = a+(a+a.sum(-1,keepdim=True)).sum(-1, keepdim=True)
    # run_schedule(check_schedule(out, 2))
    run_schedule(check_schedule(out, 3))
    np.testing.assert_allclose(out.numpy(), \
      a.numpy()+(a.numpy()+a.numpy().sum(axis=-1,keepdims=True)).sum(axis=-1,keepdims=True), atol=1e-4, rtol=1e-4)

  # multireduce spec
  def test_branching_reduces_and_expands_fusion(self):
    Tensor.manual_seed(0)
    a = Tensor.randn(4, 32).realize()
    out0 = a+a.sum(-1, keepdim=True)
    out1 = out0.sum(-1)
    # run_schedule(check_schedule(out, 2))
    run_schedule(check_schedule([out0, out1], 3))
    np.testing.assert_allclose(out0.numpy(), a.numpy()+a.numpy().sum(axis=-1,keepdims=True), atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(out1.numpy(), (a.numpy()+a.numpy().sum(axis=-1,keepdims=True)).sum(axis=-1), atol=1e-4, rtol=1e-4)

  # multireduce spec
  def test_multireduce_fusion_simple_sequential(self):
    Tensor.manual_seed(0)
    x = Tensor.randn(4, 32).realize()
    y = Tensor.randn(4, 32).realize()
    out = (y + x.sum(axis=-1, keepdim=True)).sum(axis=-1)
    # run_schedule(check_schedule(out, 1))
    run_schedule(check_schedule(out, 2))
    np.testing.assert_allclose(out.numpy(), (y.numpy() + x.numpy().sum(axis=-1, keepdims=True)).sum(axis=-1), atol=1e-4, rtol=1e-4)

  # multireduce spec
  def test_multireduce_fusion_simple_parallel(self):
    Tensor.manual_seed(0)
    x = Tensor.randn(4, 32).realize()
    y = Tensor.randn(4, 32).realize()
    out = y.sum(axis=-1) + x.sum(axis=-1)
    # run_schedule(check_schedule(out, 1))
    run_schedule(check_schedule(out, 2))
    np.testing.assert_allclose(out.numpy(), y.numpy().sum(axis=-1) + x.numpy().sum(axis=-1), atol=1e-4, rtol=1e-4)

  # multireduce spec
  def test_multireduce_fusion_sequential(self):
    Tensor.manual_seed(0)
    x = Tensor.randn(4, 32).realize()
    out = x.std(-1)
    # run_schedule(check_schedule(out, 1))
    run_schedule(check_schedule(out, 2))
    np.testing.assert_allclose(out.numpy(), x.numpy().std(axis=-1, ddof=1), atol=1e-4, rtol=1e-4)

  # multireduce spec
  def test_multireduce_fusion_parallel(self):
    Tensor.manual_seed(0)
    x = Tensor.randn(4, 32).realize()
    y = Tensor.randn(4, 32).realize()
    out = x.std(-1) + y.std(-1)
    # run_schedule(check_schedule(out, 1))
    run_schedule(check_schedule(out, 4))
    np.testing.assert_allclose(out.numpy(), x.numpy().std(axis=-1, ddof=1) + y.numpy().std(axis=-1, ddof=1), atol=1e-4, rtol=1e-4)

  # multireduce spec
  def test_multireduce_diffops_sequential(self):
    Tensor.manual_seed(0)
    x = Tensor.randn(4, 32).realize()
    out = (x - x.max(-1, keepdim=True)).sum(-1)
    # run_schedule(check_schedule(out, 1))
    run_schedule(check_schedule(out, 2))
    np.testing.assert_allclose(out.numpy(), (x.numpy() - x.numpy().max(axis=-1, keepdims=True)).sum(axis=-1), atol=1e-4, rtol=1e-4)

  # multireduce spec
  def test_multireduce_fusion_diffops_parallel(self):
    Tensor.manual_seed(0)
    x = Tensor.randn(4, 32).realize()
    y = Tensor.randn(4, 32).realize()
    out = x.sum(-1) + y.max(-1)
    # run_schedule(check_schedule(out, 1))
    run_schedule(check_schedule(out, 2))
    np.testing.assert_allclose(out.numpy(), x.numpy().sum(axis=-1) + y.numpy().max(axis=-1), atol=1e-4, rtol=1e-4)

  # multireduce spec
  def test_multireduce_fusion_sequential_and_parallel(self):
    Tensor.manual_seed(0)
    x = Tensor.randn(4, 32).realize()
    y = Tensor.randn(4, 32).realize()
    mu = (x - x.max(axis=-1, keepdim=True)).mean(axis=-1, keepdim=True) + (y - y.max(axis=-1, keepdim=True)).mean(axis=-1, keepdim=True)
    out = [((x - mu).square().sum(-1)/x.shape[-1]).sqrt(), ((y - mu).square().sum(-1)/y.shape[-1]).sqrt()]
    np_mu = (x.numpy() - x.numpy().max(axis=-1, keepdims=True)).mean(axis=-1, keepdims=True) + \
      (y.numpy() - y.numpy().max(axis=-1, keepdims=True)).mean(axis=-1, keepdims=True)
    # run_schedule(check_schedule(out, 1))
    run_schedule(check_schedule(out, 6))
    np.testing.assert_allclose(out[0].numpy(), np.sqrt(np.square(x.numpy() - np_mu).sum(-1)/x.shape[-1]), atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(out[1].numpy(), np.sqrt(np.square(y.numpy() - np_mu).sum(-1)/y.shape[-1]), atol=1e-4, rtol=1e-4)

  # multireduce spec
  def test_multimatmul_fusion(self):
    Tensor.manual_seed(0)
    a,b = Tensor.randn(4, 64).realize(), Tensor.rand(64,8).realize()
    c,d = Tensor.randn(4, 64).realize(), Tensor.rand(64,8).realize()
    out = a@b + c@d
    # run_schedule(check_schedule(out, 1))
    run_schedule(check_schedule(out, 2))
    np.testing.assert_allclose(out.numpy(), a.numpy()@b.numpy() + c.numpy()@d.numpy(), atol=1e-4, rtol=1e-4)

  def test_softmax_fusion(self):
    Tensor.manual_seed(0)
    x = Tensor.randn(4, 12, 64, 64).realize()
    out = x.softmax()
    # run_schedule(check_schedule(out, 2))
    run_schedule(check_schedule(out, 3))
    expected = (x_exp:=np.exp(x.numpy()-x.numpy().max(-1, keepdims=True)))/x_exp.sum(-1, keepdims=True)
    np.testing.assert_allclose(out.numpy(), expected, atol=1e-4, rtol=1e-4)

  # changed by: multireduce spec
  def test_layernorm_onelayer_fusion(self):
    Tensor.manual_seed(0)
    layer = nn.LayerNorm([10, 10])
    layer.weight = Tensor.randn(10,10).realize()
    layer.bias = Tensor.randn(10,10).realize()
    x = Tensor.randn(20, 5, 10, 10).realize()
    out = layer(x)
    # run_schedule(check_schedule(out, 2))
    run_schedule(check_schedule(out, 3))
    y = (x.numpy() - x.numpy().mean(layer.axis, keepdims=True))
    expected = y / np.sqrt((y*y).mean(layer.axis, keepdims=True) + layer.eps)
    np.testing.assert_allclose(out.numpy(), expected * layer.weight.numpy() + layer.bias.numpy(), atol=1e-4, rtol=1e-4)

  def test_scaled_dot_product_attention_fusion(self):
    x, y, z, m = (Tensor.empty(32, 8, 16, 16) for _ in range(4))
    out = Tensor.scaled_dot_product_attention(x, y, z, attn_mask=m)
    check_schedule(out, 5)

  def test_scaled_dot_product_attention_causal_fusion(self):
    x, y, z, m = (Tensor.empty(32, 8, 16, 16) for _ in range(4))
    out = Tensor.scaled_dot_product_attention(x, y, z, attn_mask=m, is_causal=True)
    check_schedule(out, 6)

  def test_adam_step_fusion(self):
    with Tensor.train():
      x = Tensor.empty(4, 64, 768)
      layer = nn.Linear(768, 768*4)
      opt = nn.optim.Adam(nn.state.get_parameters(layer), lr=1e-4)
      layer(x).relu().sum().backward()
      check_schedule(opt.schedule_step(), 11)

  def test_adam_conv_fuse(self):
    with Tensor.train():
      img = Tensor.empty(2,3,4,4)
      c1 = nn.Conv2d(3,32,3)
      opt = nn.optim.Adam(nn.state.get_parameters(c1), lr=1e-4)
      opt.zero_grad()
      c1(img).relu().sum().backward()
      check_schedule(opt.schedule_step(), 11)

  def test_adam_2convs_fuse(self):
    with Tensor.train():
      img = Tensor.empty(2,3,4,4)
      c1 = nn.Conv2d(3,16,3,bias=False)
      c2 = nn.Conv2d(16,32,3,bias=False)
      opt = nn.optim.Adam(nn.state.get_parameters([c1, c2]), lr=1e-4)
      opt.zero_grad()
      c2(c1(img).relu()).relu().sum().backward()
      check_schedule(opt.schedule_step(), 13)

  def test_sgd_conv_fuse(self):
    with Tensor.train():
      img = Tensor.empty(2,3,4,4)
      c1 = nn.Conv2d(3,32,3)
      opt = nn.optim.SGD(nn.state.get_parameters(c1))
      opt.zero_grad()
      c1(img).relu().sum().backward()
      check_schedule(opt.schedule_step(), 7)

  def test_sgd_2convs_fuse(self):
    with Tensor.train():
      img = Tensor.empty(2,3,4,4)
      c1 = nn.Conv2d(3,16,3,bias=False)
      c2 = nn.Conv2d(16,32,3,bias=False)
      opt = nn.optim.SGD(nn.state.get_parameters([c1, c2]))
      opt.zero_grad()
      c2(c1(img).relu()).relu().sum().backward()
      check_schedule(opt.schedule_step(), 7)

  def test_fold_2convs_sgd_nesterov_momentum_wd(self):
    with Tensor.train():
      img = Tensor.empty(2,3,4,4)
      c1 = nn.Conv2d(3,16,3,bias=False)
      c2 = nn.Conv2d(16,32,3,bias=False)
      opt = nn.optim.SGD(nn.state.get_parameters([c1, c2]), nesterov=True, momentum=0.9, weight_decay=0.1)
      opt.zero_grad()
      c2(c1(img).relu()).relu().sum().backward()
      check_schedule(opt.schedule_step(), 9)

  def test_sgd_4convs_fuse(self):
    with Tensor.train():
      img = Tensor.empty(2,3,64,64)
      c1 = nn.Conv2d(3,4,3,bias=False)
      c2 = nn.Conv2d(4,8,3,bias=False)
      c3 = nn.Conv2d(8,16,3,bias=False)
      c4 = nn.Conv2d(16,32,3,bias=False)
      opt = nn.optim.SGD(nn.state.get_parameters([c1, c2, c3, c4]))
      opt.zero_grad()
      c4(c3(c2(c1(img).relu()).relu()).relu()).relu().sum().backward()
      check_schedule(opt.schedule_step(), 22)

  @unittest.skipUnless(is_dtype_supported(dtypes.half), "need half")
  def test_prefer_half_buffer(self):
    x = Tensor.ones(4).contiguous().realize()
    # y = Tensor.ones(4).contiguous().realize()
    z = Tensor.ones(4, 4).contiguous().realize()

    # should not create extra kernel if output will be realized anyways
    dummy = x.sum().half().float()
    check_schedule(dummy, 1)
    dummy = x.sum().half().float().contiguous() + 1
    check_schedule(dummy, 2)

    # shared between two outputs
    shared = x.sum().half().float()
    a = shared * 2
    b = shared * 3
    sched = check_schedule([a, b], 1)
    for si in sched[:-2]: assert all(out.dtype == dtypes.half for out in si.outputs)

    # reduce
    a = z.sum(axis=0).half().float().sum(axis=0)
    sched = check_schedule(a, 2)
    for si in sched[:-1]: assert all(out.dtype == dtypes.half for out in si.outputs)

    # expand
    # expand will realize just after the .float(), so requires change to realize-before-expand
    # normal = (x.sum().half().float().reshape(1) * y).sum()
    # sched = check_schedule(normal, 2)
    # for si in sched[:-1]: assert all(out.dtype == dtypes.half for out in si.outputs[:-1])

    # parallel reduce
    # a = x.sum().half().float() * y.sum().half().float()
    # b = a + 1
    # c = a + 2
    # sched = check_schedule([b, c], 4)
    # doesn't store either in half because it doesn't chase

  def test_reduce_simple_chase(self):
    a = Tensor.empty(4, 4, 4)
    r = a.sum(0) + 6
    b = r.sum(0) * 4
    c = r.sum(1) * 2
    schedule = check_schedule([b, c], 3)
    self.assertIs(schedule[0].ast.src[0].src[2].arg, BinaryOps.ADD)

  # multireduce spec
  def test_multireduce_simple_chase(self):
    Tensor.manual_seed(0)
    a = Tensor.randn(4, 4, 4).realize()
    r = (a + (a.sum(0, keepdim=True) + 6)).sum(0) * 2
    b = r.sum(0) + 8
    c = r.sum(1) + 12
    np_r = (a.numpy() + (a.numpy().sum(0) + 6)).sum(0) * 2
    # schedule = check_schedule([b,c], 3)
    # self.assertIs(schedule[0].ast[0].src[0].arg, BinaryOps.MUL)
    schedule = check_schedule([b,c], 4)
    run_schedule(schedule)
    np.testing.assert_allclose(b.numpy(), np_r.sum(0) + 8, atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(c.numpy(), np_r.sum(1) + 12, atol=1e-4, rtol=1e-4)

  def test_push_permute_chase(self):
    a = Tensor.empty(4, 4, 4)
    b = Tensor.empty(4, 4)
    r = a.sum(2) + b
    d = r.T * 4
    e = r * d
    schedule = check_schedule([d, e], 3)
    self.assertIs(schedule[0].ast.src[0].src[2].arg, BinaryOps.ADD)

  # multireduce spec
  def test_multireduce_push_permute_chase(self):
    Tensor.manual_seed(0)
    a = Tensor.randn(4, 4, 4).realize()
    b = Tensor.randn(4, 4).realize()
    r = a.sum(2) + b
    d = r.T * 4
    e = r * (d + a).sum(2)
    schedule = check_schedule([d, e], 3) # make sure it doesn't fuse
    self.assertIs(schedule[0].ast.src[0].src[2].arg, BinaryOps.ADD)
    run_schedule(schedule)
    np.testing.assert_allclose(d.numpy(), (a.numpy().sum(2) + b.numpy()).T * 4, atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(e.numpy(), (a.numpy().sum(2) + b.numpy()) * (d.numpy() + a.numpy()).sum(2), atol=1e-4, rtol=1e-4)

  def test_push_shrink_chase(self):
    a = Tensor.empty(16, 16)
    b = Tensor.empty(4)
    c = Tensor.empty(16, )
    r = a.sum(1) + c
    d = r[:4] * b
    schedule = check_schedule(d, 2)
    self.assertIs(schedule[0].ast.src[0].src[2].arg, BinaryOps.ADD)

  # multireduce spec
  def test_multireduce_push_shrink_chase(self):
    Tensor.manual_seed(0)
    a = Tensor.randn(16, 16).realize()
    b = Tensor.randn(4).realize()
    c = Tensor.randn(16, ).realize()
    d = Tensor.randn(16, 16).realize()
    r = a.sum(1) + c
    out = r[:4] * b + d.sum(1)[:4]
    # schedule = check_schedule(out, 2)
    schedule = check_schedule(out, 3)
    self.assertIs(schedule[0].ast.src[0].src[2].arg, BinaryOps.ADD)
    run_schedule(schedule)
    np.testing.assert_allclose(out.numpy(), (a.numpy().sum(1) + c.numpy())[:4] * b.numpy() + d.numpy().sum(1)[:4], atol=1e-4, rtol=1e-4)

  def test_midreduce_nochase(self):
    a = Tensor.empty(16, 16)
    b = (a.sum(0) + a.max(1)) + 2
    schedule = check_schedule(b, 2)
    self.assertIs(schedule[0].ast.src[0].src[2].op, UOps.REDUCE_AXIS)

  # multireduce spec
  def test_multireduce_midreduce_nochase(self):
    Tensor.manual_seed(0)
    a = Tensor.randn(16, 16).realize()
    b = (a.sum(0)+a.max(0) + a.max(1)+a.sum(1)) + 2
    # schedule = check_schedule(b, 2)
    schedule = check_schedule(b, 4)
    self.assertIs(schedule[0].ast.src[0].src[2].op, UOps.REDUCE_AXIS)
    run_schedule(schedule)
    np.testing.assert_allclose(b.numpy(), a.numpy().sum(0)+a.numpy().max(0) + a.numpy().max(1)+a.numpy().sum(1)+2, atol=1e-4, rtol=1e-4)

  # changed by: multireduce spec
  # pattern in test_transformer
  def test_partial_fuse1(self):
    Tensor.manual_seed(0)
    a = Tensor.randn(16, 16).realize()
    b = Tensor.randn(16, 16).realize()
    c = a.sum() + 2
    d = (a.sum() - b.sum()) * 4
    # run_schedule(check_schedule([c, d], 1))
    run_schedule(check_schedule([c, d], 3))
    np.testing.assert_allclose(c.numpy(), a.numpy().sum()+2, atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(d.numpy(), (a.numpy().sum() - b.numpy().sum()) * 4, atol=1e-4, rtol=1e-4)

  # changed by: multireduce spec
  # pattern in conv
  def test_partial_fuse2(self):
    Tensor.manual_seed(0)
    a = Tensor.randn(16, 16).realize()
    b = Tensor.randn(16, 16).realize()
    c = a.sum() + 2
    d = b.sum() - c
    # run_schedule(check_schedule([c, d], 1))
    run_schedule(check_schedule([c, d], 2))
    np.testing.assert_allclose(c.numpy(), a.numpy().sum()+2, atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(d.numpy(), b.numpy().sum()-(a.numpy().sum()+2), atol=1e-4, rtol=1e-4)

  # changed by: multireduce spec
  # pattern in adam
  def test_partial_fuse3(self):
    Tensor.manual_seed(0)
    a = Tensor.randn(16, 16).realize()
    b = Tensor.randn(16, 16).realize()
    c = a.sum() + 2
    d = a.sum() * 2
    e = c * d
    f = b.sum() - e
    # run_schedule(check_schedule([c, d, e, f], 1))
    run_schedule(check_schedule([c, d, e, f], 2))
    np.testing.assert_allclose(c.numpy(), c_np:=a.numpy().sum()+2, atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(d.numpy(), d_np:=a.numpy().sum()*2, atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(e.numpy(), e_np:=c_np*d_np, atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(f.numpy(), b.numpy().sum() - e_np, atol=1e-4, rtol=1e-4)

  # changed by: multireduce spec
  def test_partial_fuse4(self):
    Tensor.manual_seed(0)
    a = Tensor.randn(16, 16).realize()
    b = Tensor.randn(16, 16).realize()
    c = a.sum() + 2
    d = a.sum() * 2
    e = c * d
    f = (b - d).sum() - e
    # run_schedule(check_schedule([c, d, e, f], 1))
    run_schedule(check_schedule([c, d, e, f], 3))
    np.testing.assert_allclose(c.numpy(), c_np:=a.numpy().sum()+2, atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(d.numpy(), d_np:=a.numpy().sum()*2, atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(e.numpy(), e_np:=c_np*d_np, atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(f.numpy(), (b.numpy()-d_np).sum()-e_np, atol=1e-4, rtol=1e-4)

  def test_pad_reduce_safe(self):
    Tensor.manual_seed(0)
    a = Tensor.rand(3, 4, 5).realize()
    b = Tensor.rand(3, 4, 5).realize()
    out = (a + b).pad(((0, 1), (0, 1), (0, 1)), 1.0).sum().contiguous()
    run_schedule(check_schedule(out, 1))
    np.testing.assert_allclose(out.numpy(), np.pad(a.numpy()+b.numpy(), ((0, 1), (0, 1), (0, 1)), constant_values=1.0).sum())

  # multireduce spec
  def test_multireduce_pad_reduce_safe(self):
    Tensor.manual_seed(0)
    a = Tensor.randn(3, 4, 5).realize()
    b = Tensor.randn(3, 4, 5).realize()
    out = (a.pad(((0, 1), (0, 1), (0, 1)), 1.0).sum(keepdim=True)+b.pad(((0, 1), (0, 1), (0, 1)), 1.0).sum()).contiguous()
    # run_schedule(check_schedule(out, 1))
    run_schedule(check_schedule(out, 2))
    np.testing.assert_allclose(out.numpy(), np.pad(a.numpy(), ((0, 1), (0, 1), (0, 1)), constant_values=1.0).sum(keepdims=True) + \
                                                   np.pad(b.numpy(), ((0, 1), (0, 1), (0, 1)), constant_values=1.0).sum(), atol=1e-4, rtol=1e-4)

  def test_pad_reduce_unsafe(self):
    Tensor.manual_seed(0)
    a = Tensor.rand(3, 4, 5).realize()
    out = a.log2().pad(((0, 1), (0, 1), (0, 1)), 1.0).sum().contiguous()
    run_schedule(check_schedule(out, 2))
    np.testing.assert_allclose(out.numpy(), np.pad(np.log2(a.numpy()), ((0, 1), (0, 1), (0, 1)), constant_values=1.0).sum(), rtol=1e-6)

  # multireduce spec
  def test_multireduce_pad_reduce_unsafe(self):
    Tensor.manual_seed(0)
    a = Tensor.randn(3, 4, 5).abs().realize()
    b = Tensor.randn(3, 4, 5).abs().realize()
    out = (a.log2().pad(((0, 1), (0, 1), (0, 1)), 1.0).sum()+b).abs().log2().pad(((0, 1), (0, 1), (0, 1)), 1.0).sum().contiguous()
    # run_schedule(check_schedule(out, 1))
    run_schedule(check_schedule(out, 4))
    np.testing.assert_allclose(out.numpy(), np.pad(np.log2(np.abs(np.pad(np.log2(a.numpy()), ((0, 1), (0, 1), (0, 1)), constant_values=1.0).sum() + \
                                                   b.numpy())), ((0, 1), (0, 1), (0, 1)), constant_values=1.0).sum(), atol=1e-4, rtol=1e-6)

  def test_shrink_pad_safe(self):
    a = Tensor.ones((3, )).contiguous().realize()
    b = Tensor.ones((3, )).contiguous().realize()
    out = (a + b).shrink(((0, 1),)).pad(((0, 1),)).contiguous()
    run_schedule(check_schedule(out, 1))
    np.testing.assert_equal(out.numpy(), [2, 0])

  def test_shrink_pad_unsafe(self):
    a = Tensor.ones((3, )).contiguous().realize()
    out = a.exp2().shrink(((0, 1),)).pad(((0, 1),)).contiguous()
    run_schedule(check_schedule(out, 2))
    np.testing.assert_equal(out.numpy(), [2, 0])

  def test_base_change_shrink_pad(self):
    a = Tensor.ones(3, 3).contiguous().realize()
    b = a.exp2()
    c = b[:-1, :-1]
    d = c.pad(((0, 1), (0, 1))) * 2
    run_schedule(check_schedule(d, 2))
    np.testing.assert_equal(d.numpy(), np.pad(np.exp2(a.numpy())[:-1, :-1], ((0, 1), (0, 1)))*2)

  def test_base_change_expand_pad(self):
    a = Tensor.ones(3, 3).contiguous().realize()
    b = a.exp2()
    c = b[:, None, :]
    d = c.pad(((0, 0), (1, 1), (0, 0))) * 2
    run_schedule(check_schedule(d, 2))
    np.testing.assert_equal(d.numpy(), np.pad(np.exp2(a.numpy())[:, None, :], ((0, 0), (1, 1), (0, 0)))*2)

  # TODO like openpilot with imagef
  @unittest.skipUnless(is_dtype_supported(dtypes.half), "need half")
  def test_base_change_expand_expand(self):
    a = Tensor.ones(4, 4).contiguous().realize()
    b = a.cast(dtypes.half).expand(2, 4, 4)
    c = b.cast(dtypes.int).expand(2, 2, 4, 4)
    run_schedule(check_schedule(c, 2))
    np.testing.assert_equal(c.numpy(), np.ones(((2, 2, 4, 4)), dtype=np.int32))

  def test_base_change_pad_expand(self):
    a = Tensor.full((4, 4), 1.).contiguous().realize()
    b = Tensor.full((4, 4), 2.).contiguous().realize()
    c = (a + b).pad(((1, 1), (1, 1)))
    d = c.cast(dtypes.int).expand((2, 6, 6)) * 4
    run_schedule(check_schedule(d, 2))
    c_np = np.pad((np.full((4, 4), 2., dtype=np.float32) + np.full((4, 4), 1., dtype=np.float32)), ((1, 1), (1, 1)), constant_values=0.0)
    np.testing.assert_equal(d.numpy(), np.broadcast_to(c_np.astype(np.half), (2, *c_np.shape)) * 4)

  def test_pad_reduce_unsafe_multiview_st(self):
    P = Tensor.ones(3, 3).contiguous()
    sums = P.sum(axis=1, keepdim=True)
    P /= sums
    p = P[0]
    p = p.pad(((1, 0), ))
    p = p.repeat([2])
    run_schedule(check_schedule(p, 3))
    tiny_ret = p.numpy()

    P = np.ones((3, 3), dtype=np.float32)
    sums = P.sum(axis=1, keepdims=True)
    P /= sums
    p = P[0]
    p = np.pad(p, (1, 0), 'constant')
    p = np.tile(p, 2)
    np.testing.assert_allclose(tiny_ret, p)

  @unittest.skipIf(Device.DEFAULT not in view_supported_devices, "subbuffer not supported")
  def test_bitcast_subbufer(self):
    x = cast(LazyBuffer, Tensor.empty(1, dtype=dtypes.float32).realize().lazydata)
    a = x.e(UnaryOps.EXP2).cast(dtypes.int32, True, allow_buffer_view=True)
    b = x.cast(dtypes.int32, True, allow_buffer_view=True)
    b = a.e(BinaryOps.ADD, b)
    check_schedule(b, 2) # this should fuse when it makes sense

  def test_bitcast_disable_subbufer(self):
    x = cast(LazyBuffer, Tensor.empty(1, dtype=dtypes.float32).realize().lazydata)
    a = x.e(UnaryOps.EXP2).cast(dtypes.int32, True, allow_buffer_view=False)
    b = x.cast(dtypes.int32, True, allow_buffer_view=False)
    b = a.e(BinaryOps.ADD, b)
    check_schedule(b, 1)

  def test_reduceop_reshape_dont_push(self):
    Tensor.manual_seed(0)
    x = Tensor.randn(10, 20).realize()
    out = x.argmax(1)
    run_schedule(check_schedule(out, 3)) # TODO: push a reduceop through a reshape

class TestConvBW(unittest.TestCase):
  def check_schedule(self, xt, cnt:int, flops=None) -> List[ScheduleItem]:
    with Context(FUSE_CONV_BW=getenv("FUSE_CONV_BW", 1), NOOPT=flops is not None):
      s = create_schedule(flatten([r.lazydata.lbs for r in xt]))
      kernels = [si for si in s if si.ast.op is UOps.SINK]
      for si in kernels: verify_ast(si.ast)
      GlobalCounters.reset()
      run_schedule(s)
      if flops is not None: assert GlobalCounters.global_ops <= flops, f"too many ops {GlobalCounters.global_ops}"
      if FUSE_CONV_BW: self.assertEqual(len(kernels), cnt)
      return kernels

  def test_fold_conv_relu_backward(self):
    c1 = nn.Conv2d(3,16,3, bias=False)
    c1.weight.requires_grad = True
    img = Tensor.rand(2,3,64,64, requires_grad=True)

    # run
    c1(img).relu().mean().backward()
    assert img.grad is not None and c1.weight.grad is not None
    self.check_schedule([img.grad, c1.weight.grad], 4)

    # compare
    import torch
    c1_torch = torch.nn.Conv2d(3,16,3, bias=False)
    c1_torch.weight.requires_grad = True
    c1_torch.weight = torch.nn.Parameter(torch.tensor(c1.weight.numpy(), dtype=torch.float32))
    img_torch = torch.tensor(img.numpy(), requires_grad=True)
    c1_torch(img_torch).relu().mean().backward()
    assert img_torch.grad is not None and c1_torch.weight.grad is not None
    np.testing.assert_allclose(c1.weight.grad.numpy(), c1_torch.weight.grad.numpy(), atol=5e-4, rtol=1e-5)
    np.testing.assert_allclose(img.grad.numpy(), img_torch.grad.numpy(), atol=5e-4, rtol=1e-5)

  def test_fold_conv_relu_backward_ast_rewrite(self):
    # shared params
    Tensor.manual_seed(0)
    img_np = Tensor.randn(2,3,64,64).numpy()
    c1_w = Tensor.randn(16,3,3,3).numpy()
    # graph_rewrite
    GlobalCounters.reset()
    c1 = nn.Conv2d(3,16,3, bias=False)
    c1.weight = Tensor(c1_w, requires_grad=True)
    img = Tensor(img_np, requires_grad=True)
    c1(img).relu().mean().backward()
    assert img.grad is not None and c1.weight.grad is not None
    with Context(AST_REWRITE=1): compare_ast = self.check_schedule([img.grad, c1.weight.grad], 3)[1].ast
    rw_flops = GlobalCounters.global_ops
    # ref
    GlobalCounters.reset()
    c1_ref = nn.Conv2d(3,16,3, bias=False)
    c1_ref.weight = Tensor(c1_w, requires_grad=True)
    img_ref = Tensor(img_np, requires_grad=True)
    c1_ref(img_ref).relu().mean().backward()
    assert img_ref.grad is not None and c1_ref.weight.grad is not None
    with Context(AST_REWRITE=0): ref_ast = self.check_schedule([img_ref.grad, c1_ref.weight.grad], 3)[1].ast
    ref_flops = GlobalCounters.global_ops
    # correctness
    np.testing.assert_allclose(c1.weight.grad.numpy(), c1_ref.weight.grad.numpy(), atol=5e-4, rtol=1e-5)
    np.testing.assert_allclose(img.grad.numpy(), img_ref.grad.numpy(), atol=5e-4, rtol=1e-5)
    # flops, TODO: This will be fixed once SWIZZLE merges view strides.
    with self.assertRaises(AssertionError):
      self.assertEqual(rw_flops, ref_flops)
      assert_equiv_uops(compare_ast, ref_ast)

  @unittest.expectedFailure
  @unittest.skipUnless(is_dtype_supported(dtypes.half), "need half")
  def test_fold_conv_relu_backward_half(self):
    old_float = dtypes.default_float
    dtypes.default_float = dtypes.float16

    c1 = nn.Conv2d(3,16,3, bias=False)
    c1.weight.requires_grad = True

    # run
    img = Tensor.rand(2,3,64,64, requires_grad=True)
    c1(img).relu().mean().backward()
    dtypes.default_float = old_float
    self.check_schedule([img.grad, c1.weight.grad], 4)

    # compare
    import torch
    c1_torch = torch.nn.Conv2d(3,16,3, bias=False, dtype=torch.half)
    c1_torch.weight.requires_grad = True
    c1_torch.weight = torch.nn.Parameter(torch.tensor(c1.weight.numpy(), dtype=torch.half))
    img_torch = torch.tensor(img.numpy(), requires_grad=True)
    c1_torch(img_torch).relu().mean().backward()
    assert img_torch.grad is not None and c1_torch.weight.grad is not None
    np.testing.assert_allclose(c1.weight.grad.numpy(), c1_torch.weight.grad.numpy(), atol=5e-4, rtol=1e-5)
    np.testing.assert_allclose(img.grad.numpy(), img_torch.grad.numpy(), atol=5e-4, rtol=1e-5)

  def test_sgd_4convs_fuse(self):
    with Tensor.train():
      img = Tensor.empty(2,3,64,64)
      c1 = nn.Conv2d(3,4,3,bias=False)
      c2 = nn.Conv2d(4,8,3,bias=False)
      c3 = nn.Conv2d(8,16,3,bias=False)
      c4 = nn.Conv2d(16,32,3,bias=False)
      opt = nn.optim.SGD(nn.state.get_parameters([c1, c2, c3, c4]))
      opt.zero_grad()
      c4(c3(c2(c1(img).relu()).relu()).relu()).relu().sum().backward()
      self.check_schedule(opt.schedule_step(), 19)

class TestIndexing(unittest.TestCase):
  def check_schedule(self, xt:Union[Tensor,List[Tensor]], cnt:int):
    with Context(FUSE_ARANGE=getenv("FUSE_ARANGE", 1)):
      lst = [xt] if isinstance(xt, Tensor) else xt
      s = Tensor.schedule(*lst)
      kernels = [si for si in s if si.ast.op is UOps.SINK]
      for si in kernels: verify_ast(si.ast)
      run_schedule(s)
      if FUSE_ARANGE: self.assertEqual(len(kernels), cnt)

  def test_simple_indexing(self):
    X = Tensor.randn(10, 10).realize()
    idxs = Tensor([0, 2]).realize()
    xt = X[idxs]
    self.check_schedule(xt, 2)
    np.testing.assert_equal(xt.numpy(), X.numpy()[idxs.numpy()])

  def test_simple_indexing_alt(self):
    X = Tensor.arange(16).reshape(4, 4)
    xt = X[[1, 2], [1, 2]]
    self.check_schedule(xt, 3)
    np.testing.assert_equal(xt.numpy(), (np.arange(16).reshape(4, 4))[[1, 2], [1, 2]])

  @unittest.expectedFailure
  def test_advanced_indexing(self):
    X = Tensor.arange(10)+1
    xt = X[[0]]
    self.check_schedule(xt, 3)
    np.testing.assert_equal(xt.numpy(), (np.arange(10)+1)[[0]])

  @unittest.expectedFailure
  def test_advanced_indexing_alt(self):
    X = Tensor.arange(6).reshape(3, 2)+1
    xt = X[[Tensor([2]), Tensor([1])]]
    self.check_schedule(xt, 6)
    np.testing.assert_equal(xt.numpy(), 6)

  def test_advanced_simple_indexing_combined(self):
    X = Tensor.arange(16).reshape(4, 4)
    xt = X[1:2, [1, 2]]
    self.check_schedule(xt, 2)

  def test_push_through_reshape(self):
    Tensor.manual_seed(0)
    x = Tensor.randn(10, 20).realize()
    out = x.argmax(1)
    self.check_schedule(out, 2)
    np.testing.assert_allclose(out.numpy(), np.argmax(x.numpy(), 1))

  def test_arange_push_through_expand(self):
    Tensor.manual_seed(0)
    a = Tensor.arange(4,)
    b = Tensor.randn(4, 4).realize()
    out = a+b
    self.check_schedule(out, 1)
    np.testing.assert_allclose(out.numpy(), np.arange(4)+b.numpy())

  def test_argmin(self):
    Tensor.manual_seed(0)
    x = Tensor.randn(4, 32).realize()
    out = x.argmin(-1)
    self.check_schedule(out, 2)
    np.testing.assert_equal(out.numpy(), x.numpy().argmin(axis=-1))

  def test_argmax(self):
    Tensor.manual_seed(0)
    x = Tensor.randn(4, 32).realize()
    out = x.argmax(-1)
    self.check_schedule(out, 2)
    np.testing.assert_equal(out.numpy(), x.numpy().argmax(axis=-1))

  def test_arange_transposed(self):
    Tensor.manual_seed(0)
    x = Tensor.randint(4, 1)
    a = (Tensor.arange(4,)*x).T
    self.check_schedule(a, 2)
    np.testing.assert_equal(a.numpy(), (np.arange(4)*x.numpy()).T)

  def test_arange_transposed_descendants(self):
    Tensor.manual_seed(0)
    x = Tensor.randint(4, 1)
    a = (Tensor.arange(4,)*x).T
    b = Tensor.randint(4, 4).realize()
    out = a+b
    self.check_schedule(out, 2)
    np.testing.assert_equal(out.numpy(), (np.arange(4)*x.numpy()).T+b.numpy())

  def test_arange_index(self):
    Tensor.manual_seed(0)
    x = Tensor.randn(5, 2).realize()
    a = Tensor.arange(10)
    out = (x + a[2]).sum()
    self.check_schedule(out, 1)
    np.testing.assert_allclose(out.numpy(), (x.numpy()+np.arange(10)[2]).sum())

  def test_arange_index_contiguous(self):
    Tensor.manual_seed(0)
    x = Tensor.randn(5, 2).realize()
    a = Tensor.arange(10).contiguous()
    out = (x + a[2]).sum()
    self.check_schedule(out, 2)
    np.testing.assert_allclose(out.numpy(), (x.numpy()+np.arange(10)[2]).sum())

  def test_arange_index_child(self):
    Tensor.manual_seed(0)
    x = Tensor.randn(5, 2).realize()
    a = Tensor.arange(10)+1
    out = (x + a[2]).sum()
    self.check_schedule(out, 1)
    np.testing.assert_allclose(out.numpy(), (x.numpy()+(np.arange(10)+1)[2]).sum())

  def test_arange_index_contiguous_child(self):
    Tensor.manual_seed(0)
    x = Tensor.randn(5, 2).realize()
    a = (Tensor.arange(10)+1).contiguous()
    out = (x + a[2]).sum()
    self.check_schedule(out, 2)
    np.testing.assert_allclose(out.numpy(), (x.numpy()+(np.arange(10)+1)[2]).sum())

  def test_arange_childless_base(self):
    a = Tensor.arange(4)
    self.check_schedule(a, 1)
    np.testing.assert_equal(a.numpy(), np.arange(4))

  def test_arange_childless_view(self):
    a = Tensor.arange(4).reshape(2, 2)
    a[0] = 4
    np.testing.assert_equal(a.numpy(), [[4, 4], [2, 3]])

  def test_arange_group_childless_base(self):
    Tensor.manual_seed(0)
    x = Tensor.randint(4)
    a = Tensor.arange(4)+x
    self.check_schedule(a, 1)
    np.testing.assert_equal(a.numpy(), np.arange(4)+x.numpy())

  def test_arange_group_childless_view(self):
    Tensor.manual_seed(0)
    x = Tensor.ones(4).contiguous().realize()
    a = Tensor.arange(4)+x
    a[0] = 6
    np.testing.assert_equal(a.numpy(), [6., 2., 3., 4.])

  @unittest.skipUnless(Device.DEFAULT in view_supported_devices, "need view")
  def test_arange_view_op(self):
    a = Tensor.arange(12).reshape(4, 3).shrink(((1, 2), (1, 3))).contiguous()
    assert isinstance(a.lazydata, LazyBuffer)
    self.assertIs(a.lazydata.base.op, MetaOps.VIEW)
    self.check_schedule(a, 1)
    np.testing.assert_equal(a.numpy(), [[4, 5]])

  @unittest.skipIf(Device.DEFAULT == "CLANG", "tests copy from ext device")
  def test_arange_shrink_copy(self):
    a = Tensor.arange(12).reshape(4, 3).shrink(((1, 2), (1, 3))).to("CLANG")
    assert isinstance(a.lazydata, LazyBuffer)
    self.assertIs(a.lazydata.base.op, MetaOps.COPY)
    self.check_schedule(a, 1)
    np.testing.assert_equal(a.numpy(), [[4, 5]])

  @unittest.skipIf(Device.DEFAULT == "CLANG", "tests copy from ext device")
  def test_arange_expand_copy(self):
    a = Tensor.arange(4).reshape(2, 2, 1).expand(2, 2, 2).to("CLANG")
    assert isinstance(a.lazydata, LazyBuffer)
    self.assertIs(a.lazydata.base.op, MetaOps.COPY)
    self.assertIs(a.lazydata.base.srcs[0].base.op, BinaryOps.ADD)
    self.check_schedule(a, 1)
    np.testing.assert_equal(a.numpy(), [[[0, 0], [1, 1]], [[2, 2], [3, 3]]])

  @unittest.skipUnless(is_dtype_supported(dtypes.half), "need half")
  def test_precompute_freqs_cis(self):
    args = {"dim":32 if CI else 128, "end":2048 if CI else 8192, "theta":10000, "dtype":dtypes.half}
    fused = precompute_freqs_cis(**args)
    self.check_schedule(fused, 1)
    if getenv("CHECK", 1):
      ref = precompute_freqs_cis(**args)
      run_schedule(check_schedule(ref, 3))
      np.testing.assert_equal(fused.numpy(), ref.numpy())

  def test_fuse_assign_contiguous(self):
    x = Tensor.zeros(4, 4, dtype=dtypes.int).contiguous().realize()
    a = Tensor.arange(8).reshape(4, 2)
    self.check_schedule(x.shrink((None, (0, 2))).assign(a.contiguous()), 2)
    np.testing.assert_equal(x.numpy(), [[0, 1, 0, 0], [2, 3, 0, 0], [4, 5, 0, 0], [6, 7, 0, 0]])

  def test_assign_non_contiguous(self):
    x = Tensor.zeros(4, 4, dtype=dtypes.int).contiguous().realize()
    y = Tensor.randint(4, 2)
    a = Tensor.arange(8).reshape(4, 2)+y
    x.shrink((None, (0, 2))).assign(a).realize()
    xref = np.zeros((4, 4), dtype=int)
    xref[:, :2] = np.arange(8).reshape(4, 2)+y.numpy()
    np.testing.assert_equal(x.numpy(), xref)

  def test_sparse_categorical_crossentropy_simple(self):
    X = Tensor([[0, 2, 3], [1, 2, 3]]).realize()
    Y = Tensor([1, 2]).realize()
    loss = X.sparse_categorical_crossentropy(Y)
    self.check_schedule(loss, 5)
    np.testing.assert_allclose(loss.item(), 0.878309, atol=1e-5, rtol=1e-6)

  def test_mnist_val(self):
    from tinygrad.nn.datasets import mnist
    import torch
    _, Y_train, _, _ = mnist()
    samples = Tensor.randint(BS:=getenv("BS", 512), high=cast(int,Y_train.shape[-1]))
    yt = Tensor.randn(BS, 10)
    with Context(SPLIT_REDUCEOP=0):
      loss = yt.sparse_categorical_crossentropy(Y_train[samples])
      self.check_schedule(loss, 6)
      loss_fused = loss.numpy()
    loss_ref = torch.nn.CrossEntropyLoss()(torch.tensor(yt.numpy()), torch.tensor(Y_train.numpy())[torch.tensor(samples.numpy())])
    np.testing.assert_allclose(loss_fused, loss_ref.numpy(), atol=1e-6, rtol=1e-6)

  def test_arange_fuse_grouped_children(self):
    X = Tensor.randn(4, 4).realize()
    r = (X+Tensor.arange(16).reshape(4, 4)).sum()
    out0 = r+2
    out1 = r+3
    self.check_schedule([out0, out1], 1)
    r_ref = (X.numpy()+np.arange(16).reshape(4, 4)).sum()
    np.testing.assert_allclose(out0.numpy(), r_ref+2, rtol=2e-7)
    np.testing.assert_allclose(out1.numpy(), r_ref+3, rtol=2e-7)

  @unittest.expectedFailure
  def test_fold_arange_view(self):
    X = Tensor.randn(4, 4).realize()
    r = (X+Tensor.arange(16).reshape(4, 4).contiguous()).sum(1, keepdim=True)
    self.check_schedule([r], 1)
    np.testing.assert_allclose(r.numpy(), (X.numpy()+np.arange(16).reshape(4, 4)).sum(1, keepdims=True))

  @unittest.expectedFailure
  def test_multiview_arange_children(self):
    X = Tensor.randn(2,3,4,4).numpy()
    with Context(FUSE_ARANGE=1):
      compare = Tensor(X).interpolate(size=(2, 2), mode="linear").numpy()
    with Context(FUSE_ARANGE=0, GRAPH=0, SAVE_SCHEDULE=1):
      ref = Tensor(X).interpolate(size=(2, 2), mode="linear").numpy()
    np.testing.assert_allclose(ref, compare, atol=1e-5, rtol=1e-6)

class TestScheduleRewrite(unittest.TestCase):
  def setUp(self):
    self.old_val = AST_REWRITE.value
    AST_REWRITE.value = 1
  def tearDown(self): AST_REWRITE.value = self.old_val

  def test_recursive_st_fixup(self):
    a = Tensor([1,2,3,4]).realize()
    for _ in range(24): a = a + a
    ast = a.schedule()[0].ast
    new_uop, et = timeit(st_fixup, ast.src[0].src[2], lambda st:st.reshape((4, 1)), {}, {})
    self.assertEqual(get_output_st(new_uop, {}), ShapeTracker.from_shape((4,)).reshape((4, 1)))
    self.assertLess(et, 1e3)

  def test_no_rewrite_elementwise(self):
    bufs = [UOp(UOps.DEFINE_GLOBAL, PtrDType(dtypes.int), (), i) for i in range(3)]
    ld1 = UOp(UOps.LOAD, dtypes.int, (bufs[1], ShapeTracker.from_shape((32, 32)).to_uop()))
    ld2 = UOp(UOps.LOAD, dtypes.int, (bufs[2], ShapeTracker.from_shape((32, 32)).to_uop()))
    sink = UOp(UOps.SINK, None, (UOp(UOps.STORE, None, (bufs[0], ShapeTracker.from_shape((32, 32)).to_uop(), ld1+ld2)),))
    rsink = graph_rewrite(sink, reduceop_fusor)
    self.assertEqual(rsink.key, sink.key)

  def test_simple_store_reshape(self):
    bufs = [UOp(UOps.DEFINE_GLOBAL, PtrDType(dtypes.int), (), i) for i in range(2)]
    ld = UOp(UOps.LOAD, dtypes.int, (bufs[1], ShapeTracker.from_shape((32, 32)).to_uop()))
    r = UOp(UOps.REDUCE_AXIS, dtypes.int, (ld,), (BinaryOps.ADD, (0, 1)))
    r = r + UOp(UOps.CONST, dtypes.int, (ShapeTracker.from_shape(()).to_uop(),), 2)
    sink = UOp(UOps.SINK, None, (UOp(UOps.STORE, None, (bufs[0], ShapeTracker.from_shape(()).to_uop(), r)),))
    rsink = graph_rewrite(sink, reduceop_fusor)
    with self.assertRaisesRegex(AssertionError, "implicit reshape"): verify_ast(sink)
    verify_ast(rsink)

  def test_no_reshape_reduceop(self):
    bufs = [UOp(UOps.DEFINE_GLOBAL, PtrDType(dtypes.int), (), i) for i in range(2)]
    ld = UOp(UOps.LOAD, dtypes.int, (bufs[1], ShapeTracker.from_shape((32, 32)).to_uop()))
    r = UOp(UOps.REDUCE_AXIS, dtypes.int, (ld,), (BinaryOps.ADD, (0, 1)))
    sink = UOp(UOps.SINK, None, (UOp(UOps.STORE, None, (bufs[0], ShapeTracker.from_shape((1, 1)).to_uop(), r)),))
    rsink = graph_rewrite(sink, reduceop_fusor)
    verify_ast(sink)
    self.assertEqual(sink.key, rsink.key)

  def test_reshape_many(self):
    bufs = [UOp(UOps.DEFINE_GLOBAL, PtrDType(dtypes.int), (), i) for i in range(2)]
    ld = UOp(UOps.LOAD, dtypes.int, (bufs[1], ShapeTracker.from_shape((32, 32)).to_uop()))
    r = UOp(UOps.REDUCE_AXIS, dtypes.int, (ld,), (BinaryOps.ADD, (0, 1)))
    for _ in range(24): r = r + UOp(UOps.CONST, dtypes.int, (ShapeTracker.from_shape(()).to_uop(),), 2)
    sink = UOp(UOps.SINK, None, (UOp(UOps.STORE, None, (bufs[0], ShapeTracker.from_shape(()).to_uop(), r)),))
    rsink, et = timeit(graph_rewrite, sink, reduceop_fusor)
    with self.assertRaisesRegex(AssertionError, "implicit reshape"): verify_ast(sink)
    verify_ast(rsink)
    self.assertLessEqual(et, 1e3)

  @unittest.expectedFailure
  def test_complexity(self):
    SZ = 30 if getenv("BIG") else 10
    sizes = [10*(i+1) for i in range(SZ)]
    tms: List[float] = []
    for sz in sizes:
      bufs = [UOp(UOps.DEFINE_GLOBAL, PtrDType(dtypes.int), (), i) for i in range(2)]
      ld = UOp(UOps.LOAD, dtypes.int, (bufs[1], ShapeTracker.from_shape((32, 32)).to_uop()))
      r = UOp(UOps.REDUCE_AXIS, dtypes.int, (ld,), (BinaryOps.ADD, (0, 1)))
      for _ in range(sz): r = r + UOp(UOps.CONST, dtypes.int, (ShapeTracker.from_shape(()).to_uop(),), 2)
      sink = UOp(UOps.SINK, None, (UOp(UOps.STORE, None, (bufs[0], ShapeTracker.from_shape(()).to_uop(), r)),))
      rsink, et = timeit(graph_rewrite, sink, reduceop_fusor)
      with self.assertRaisesRegex(AssertionError, "implicit reshape"): verify_ast(sink)
      verify_ast(rsink)
      tms.append(et)
    if getenv("GRAPH_TIMING"):
      import plotly.express as px
      fig = px.line(x=sizes, y=tms, title="graph_rewrite time as ast grows")
      fig.update_layout(paper_bgcolor="black", plot_bgcolor="black", font={"color":"white"},
                        yaxis={"gridcolor":"rgba(255, 255, 255, 0.3)"}, xaxis={"gridcolor":"rgba(255, 255, 255, 0.3)"})
      fig.show()
    change = tms[-1] / tms[0]
    assert change <= SZ, f"bad complexity, time increased by {change:4.2f}x while input only grew {SZ}x"

  def test_swizzle_rewrite(self):
    # graph rewrite
    sink = UOp(UOps.SINK, None, arg=None, src=(
      UOp(UOps.STORE, None, arg=None, src=(
        UOp(UOps.DEFINE_GLOBAL, PtrDType(dtypes.int), arg=0, src=()),
        UOp(UOps.SHAPETRACKER, None, arg=ShapeTracker(views=(View(shape=(1, 1), strides=(0, 0), offset=0, mask=None, contiguous=True),)), src=()),
        UOp(UOps.REDUCE_AXIS, dtypes.int, arg=(BinaryOps.ADD, (0, 1)), src=(
          UOp(UOps.ALU, dtypes.int, arg=BinaryOps.ADD, src=(
            UOp(UOps.SWIZZLE, dtypes.int, arg=ShapeTracker(views=(View(shape=(32, 32), strides=(0, 0), offset=0, mask=None, contiguous=False),)), src=( # noqa E501
              UOp(UOps.REDUCE_AXIS, dtypes.int, arg=(BinaryOps.ADD, (0, 1)), src=(
                UOp(UOps.LOAD, dtypes.int, arg=None, src=(
                  x8:=UOp(UOps.DEFINE_GLOBAL, PtrDType(dtypes.int), arg=1, src=()),
                  UOp(UOps.SHAPETRACKER, None, arg=ShapeTracker(views=(View(shape=(32, 32), strides=(32, 1), offset=0, mask=None, contiguous=True),)), src=()),)),)),)), # noqa E501
            UOp(UOps.LOAD, dtypes.int, arg=None, src=(
               x8,
              UOp(UOps.SHAPETRACKER, None, arg=ShapeTracker(views=(View(shape=(32, 32), strides=(32, 1), offset=0, mask=None, contiguous=True),)), src=()),)),)),)),)),)) # noqa E501
    sink = graph_rewrite(sink, reduceop_fusor)
    # verify output
    k = Kernel(sink)
    p = k.to_program()
    a = Tensor.randint(32, 32).realize()
    b = Tensor.empty((), dtype=dtypes.int).realize()
    CompiledRunner(p).exec([b.lazydata.buffer, a.lazydata.buffer])
    expected_out = (a.numpy() + a.numpy().sum()).sum()
    np.testing.assert_equal(b.numpy(), expected_out)

if __name__ == '__main__':
  unittest.main(verbosity=2)
