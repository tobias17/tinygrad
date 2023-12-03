from tinygrad.tensor import Tensor
from tinygrad.nn import LayerNorm, Linear, Conv2d, Embedding
from collections import OrderedDict
from typing import List, Callable, Optional, Tuple

CLIP_CONFIGS = {
  "ViT-H-14.json": {
    "embed_dim":  1024,
    "vision_cfg": { "image_size": 224, "layers": 32, "width": 1280, "head_width": 80, "patch_size": 14 },
    "text_cfg":   { "context_length": 77, "vocab_size": 49408, "width": 1024, "heads": 16, "layers": 24 }
  },
  "ViT-bigG-14.json": {
    "embed_dim":  1280,
    "vision_cfg": { "image_size": 224, "layers": 48, "width": 1664, "head_width": 104, "patch_size": 14, "mlp_ratio": 4.9231 },
    "text_cfg":   { "context_length": 77, "vocab_size": 49408, "width": 1280, "heads": 20, "layers": 32 }
  },
}

def identity(x:Tensor) -> Tensor: return x

class MultiheadAttention:
  def __init__(self, embed_dim:int, n_heads:int, is_causal:bool):
    self.to_q = Linear(embed_dim, embed_dim, bias=False)
    self.to_k = Linear(embed_dim, embed_dim, bias=False)
    self.to_v = Linear(embed_dim, embed_dim, bias=False)
    self.n_heads = n_heads
    self.d_head  = embed_dim // n_heads
    assert embed_dim == n_heads * self.d_head
    self.to_out: List[Callable[[Tensor],Tensor]] = [Linear(n_heads*self.d_head, embed_dim)]
    self.is_causal = is_causal
  def __call__(self, x:Tensor) -> Tensor:
    q,k,v = self.to_q(x), self.to_v(x), self.to_v(x)
    q,k,v = [y.reshape(x.shape[0], -1, self.n_heads, self.d_head).transpose(-3,-2) for y in (q,k,v)]
    attention = Tensor.scaled_dot_product_attention(q, k, v, is_causal=self.is_causal).transpose(-3,-2)
    return attention.reshape(shape=(*x.shape[0:-2], -1, self.n_heads * self.d_head)).sequential(self.to_out)

class ResidualAttentionBlock:
  def __init__(self, dims:int, n_heads:int, mlp_ratio:float, is_causal:bool):
    self.ln_1 = LayerNorm(dims)
    self.attn = MultiheadAttention(dims, n_heads, is_causal)
    self.ls_1 = identity
    self.ln_2 = LayerNorm(dims)
    mlp_width = int(dims * mlp_ratio)
    self.mlp = OrderedDict([
      ("c_fc", Linear(dims, mlp_width)),
      ("gelu", lambda x: Tensor.gelu(x)),
      ("c_proj", Linear(mlp_width, dims)),
    ])
    self.ls_2 = identity
  def __call__(self, x:Tensor) -> Tensor:
    h = self.ls_1(self.attn(self.ln_1(x)))
    h = self.ls_2(self.ln_2(h).sequential(self.mlp))
    return x + h

class Transformer:
  def __init__(self, dims:int, layers:int, n_heads:int, mlp_ratio:float=4.0, is_causal:bool=False):
    self.resblocks: List[Callable[[Tensor],Tensor]] = [ResidualAttentionBlock(dims, n_heads, mlp_ratio, is_causal=is_causal) for _ in range(layers)]
  def __call__(self, x:Tensor) -> Tensor:
    return x.sequential(self.resblocks)

class VisionTransformer:
  def __init__(self, output_dim:int, image_size:int, layers:int, width:int, head_width:int, patch_size:int, mlp_ratio:float=4.0):
    grid_size = image_size // patch_size
    self.conv1 = Conv2d(in_channels=3, out_channels=output_dim, kernel_size=patch_size, stride=patch_size, bias=False)
    self.class_embedding = Tensor.zeros(width)
    self.positional_embedding = Tensor.zeros(grid_size*grid_size+1, width)
    self.patch_dropout = identity
    self.ln_pre = LayerNorm(width)
    self.transformer = Transformer(width, layers, width//head_width, mlp_ratio)
    self.ln_post = LayerNorm(width)
    self.proj = Tensor.zeros(width, output_dim)
  def __call__(self, x:Tensor) -> Tensor:
    x = self.conv1(x)
    x = x.reshape((*x.shape[:2],-1))
    x = x.permute((0,2,1))
    x = self.class_embedding.reshape((1,1,-1)).expand((x.shape[0],-1,-1)).cast(x.dtype).cat(x,dim=1)
    x = x + self.positional_embedding.cast(x.dtype)
    x = self.ln_pre(self.patch_dropout(x))
    x = self.transformer(x.permute((1,0,2))).permute((1,0,2))
    x = self.ln_post(x)
    return x[:,0] @ self.proj

class TextTransformer:
  output_tokens = False
  def __init__(self, output_dim:int, context_length:int, vocab_size:int, width:int, heads:int, layers:int):
    self.token_embedding = Embedding(vocab_size, width)
    self.positional_embedding = Tensor.zeros(context_length, width)
    self.transformer = Transformer(width, layers, heads, is_causal=True)
    self.ln_final = LayerNorm(width)
    self.text_projection = Tensor.zeros(width, output_dim)
  def __call__(self, text:Tensor) -> Tensor:
    x = self.token_embedding(text)
    x = x + self.positional_embedding[:text.shape[1]] # type: ignore
    x = self.transformer(x.permute((1,0,2))).permute((1,0,2))
    x = self.ln_final(x)
    return x[Tensor.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection # type: ignore

class OpenCLIP:
  def __init__(self, embed_dim, vision_cfg, text_cfg):
    self.visual = VisionTransformer(embed_dim, **vision_cfg)
    text = TextTransformer(embed_dim, **text_cfg)
    self.transformer = text.transformer
    self.token_embedding = text.token_embedding
    self.positional_embedding = text.positional_embedding
    self.ln_final = text.ln_final
    self.text_projection = text.text_projection
    self.logit_scale = Tensor.zeros(0)

  def encode_image(self, image:Tensor, normalize:bool=False):
    features = self.visual(image)
    return features.normalize(dim=-1) if normalize else features

  def encode_text(self, text:Tensor, normalize:bool=False):
    x = self.token_embedding(text)
    x = x + self.positional_embedding
    x = self.transformer(x.permute((1,0,2))).permute((1,0,2))
    x = self.ln_final(x)
    x = x[Tensor.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection # type: ignore
    return x.normalize(dim=-1) if normalize else x

  def __call__(self, image:Optional[Tensor]=None, text:Optional[Tensor]=None) -> Tuple[Tensor,Tensor,Tensor]:
    image_features = self.encode_image(image, normalize=True) if image else None
    text_features  = self.encode_text (text,  normalize=True) if text  else None
    return image_features, text_features, self.logit_scale.exp()
