"""EVA attention의 동일 가중치·동일 계산을 3/4D 위주로 표현하는 RKNN 비교 후보."""
import types
import torch
from torch.nn import functional as F

def forward_layout(self,x):
    if self.qk_float:raise RuntimeError('qk_float=True는 이 비교 후보에서 지원하지 않음')
    from detectron2.modeling.backbone.utils import get_rel_pos
    b,h,w,c=x.shape;n=h*w;heads=self.num_heads;d=c//heads
    image=x.permute(0,3,1,2)
    weights=self.qkv.weight
    if self.beit_like_qkv_bias:
        biases=[self.q_bias,None,self.v_bias]
    else:
        biases=[None,None,None] if self.qkv.bias is None else list(self.qkv.bias.chunk(3))
    q,k,v=[F.conv2d(image,weights[i*c:(i+1)*c].reshape(c,c,1,1),biases[i]).reshape(b*heads,d,n) for i in range(3)]
    scores=(q*self.scale).transpose(1,2)@k
    if self.use_rel_pos:
        rh=get_rel_pos(h,h,self.rel_pos_h,self.interp_type)
        rw=get_rel_pos(w,w,self.rel_pos_w,self.interp_type)
        rows=torch.arange(h,device=x.device).repeat_interleave(w)
        cols=torch.arange(w,device=x.device).repeat(h)
        # 각 위치의 q와 행/열 상대위치 가중치의 내적. 5D broadcast add를 피한다.
        q_by_pos=q.permute(2,0,1)
        rel_h=torch.bmm(q_by_pos,rh.index_select(0,rows).transpose(1,2)).permute(1,0,2)
        rel_w=torch.bmm(q_by_pos,rw.index_select(0,cols).transpose(1,2)).permute(1,0,2)
        scores=scores+rel_h.index_select(2,rows)+rel_w.index_select(2,cols)
    attn=scores.softmax(dim=-1)
    y=(attn@v.transpose(1,2)).transpose(1,2).reshape(b,c,h,w)
    y=F.conv2d(y,self.proj.weight.reshape(c,c,1,1),self.proj.bias)
    return y.permute(0,2,3,1)

def apply_layout(model):
    for block in model.blocks:block.attn.forward=types.MethodType(forward_layout,block.attn)
