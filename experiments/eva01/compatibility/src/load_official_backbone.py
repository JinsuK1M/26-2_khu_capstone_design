"""공식 EVA 탐지 백본 정의 격리. Detectron2 전체 탐지기 import 검증과 구분."""
import ast, math, numbers, logging
from pathlib import Path
from functools import partial
from typing import Union,List
import torch
from torch import nn,Tensor,Size
from torch.nn.parameter import Parameter
ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'vendor/EVA/EVA-01/det/detectron2/modeling/backbone'
def extract(path,names,ns):
    tree=ast.parse(path.read_text())
    nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
    if {n.name for n in nodes}!=set(names):raise RuntimeError('공식 정의 누락: '+str(path))
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),ns)
def definitions():
    ns=dict(torch=torch,nn=nn,F=torch.nn.functional,math=math,numbers=numbers,Tensor=Tensor,Size=Size,
            Parameter=Parameter,_shape_t=Union[int,List[int],Size],Backbone=nn.Module,
            logger=logging.getLogger('eva_probe'),trunc_normal_=nn.init.trunc_normal_,
            to_2tuple=lambda v:v if isinstance(v,tuple) else (v,v))
    extract(ROOT/'vendor/timm_0_6_11_mlp.py',['Mlp'],ns)
    extract(SRC/'utils.py',['window_partition','window_unpartition','get_rel_pos','add_decomposed_rel_pos','get_abs_pos','PatchEmbed'],ns)
    extract(SRC/'vit.py',['LayerNormWithForceFP32','Attention','Block','ViT'],ns)
    return ns
def build_backbone(size=640,meta=False):
    ns=definitions()
    args=dict(img_size=size,patch_size=16,embed_dim=1408,depth=40,num_heads=16,
              mlp_ratio=6144/1408,norm_layer=partial(nn.LayerNorm,eps=1e-6),use_rel_pos=True,
              window_size=16,window_block_indexes=[i for i in range(40) if i%4!=3],
              drop_path_rate=0.,use_act_checkpoint=False,beit_like_qkv_bias=True,
              beit_like_gamma=False,freeze_patch_embed=True)
    if not meta:return ns['ViT'](**args)
    original=torch.linspace
    def constant(*a,**kw):kw['device']='cpu';return original(*a,**kw)
    torch.linspace=constant
    try:
        with torch.device('meta'):return ns['ViT'](**args)
    finally:torch.linspace=original
