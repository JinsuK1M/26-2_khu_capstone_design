"""실제 EVA-01 탐지 가중치의 k4 구간을 보드 검사용 고정 ONNX로 추출."""
from pathlib import Path
from functools import partial
from datetime import datetime
import os,sys,json,hashlib,gc,time
os.environ.setdefault('WANDB_MODE','disabled')
import numpy as np
import torch
from torch import nn
import onnx,onnxruntime as ort
from load_official_backbone import ROOT
from check_mac_detector import guard_unused_soft_nms,SOURCE
OUT=ROOT.parent/'eva01_board_k4_layout_v2'
SHA='2cbbd46259851bad0052b418464cd866097e0e01bb60ae368b213ab040960650'
class OutputTensor(nn.Module):
    def __init__(self,model):super().__init__();self.model=model
    def forward(self,x):return self.model(x)['last_feat'].contiguous()
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024**2),b''):h.update(chunk)
    return h.hexdigest()
def main():
    for name in ('models','inputs','runs','configs'): (OUT/name).mkdir(parents=True,exist_ok=True)
    folder=OUT/'runs'/('export_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'));folder.mkdir()
    r=dict(state='running',scope='실제 사전학습 COCO 탐지 가중치의 PatchEmbed+위치정보+blocks0..3 구간. 실제사진1장, CPU/ONNX 검증. 보드 NPU·FCM·서버연결·mAP·정식속도 검증 아님.')
    try:
        torch.set_num_threads(4);torch.manual_seed(20261124)
        guard_unused_soft_nms();sys.path.insert(0,str(SOURCE))
        from PIL import Image
        for old,new in [('LINEAR','BILINEAR'),('CUBIC','BICUBIC'),('ANTIALIAS','LANCZOS')]:
            if not hasattr(Image,old):setattr(Image,old,getattr(Image.Resampling,new))
        from detectron2.modeling.backbone.vit import ViT
        from detectron2.structures import ImageList
        import cv2
        path=ROOT/'models/eva_coco_det.pth'
        print('공식 가중치 해시 검사',flush=True)
        if path.stat().st_size!=4295207874 or digest(path)!=SHA:raise RuntimeError('공식 가중치 불일치')
        print('실제 탐지판 앞4블록 구성·strict 로딩',flush=True)
        model=ViT(img_size=1280,patch_size=16,embed_dim=1408,depth=4,num_heads=16,
            mlp_ratio=6144/1408,norm_layer=partial(nn.LayerNorm,eps=1e-6),use_rel_pos=True,
            window_size=16,window_block_indexes=[0,1,2],drop_path_rate=0.,
            use_act_checkpoint=False,beit_like_qkv_bias=True,beit_like_gamma=False,freeze_patch_embed=True)
        loaded=torch.load(str(path),map_location='cpu',weights_only=True,mmap=True)
        keys=model.state_dict().keys()
        state={key:loaded['model']['backbone.net.'+key] for key in keys}
        model.load_state_dict(state,strict=True,assign=True)
        model.eval();del loaded,state;gc.collect()
        image_path=ROOT/'vendor/bus.jpg';image=cv2.imread(str(image_path))
        if image is None:raise RuntimeError('검증 사진 누락')
        rgb=cv2.cvtColor(cv2.resize(image,(336,336),interpolation=cv2.INTER_LINEAR),cv2.COLOR_BGR2RGB)
        pixels=torch.from_numpy(np.ascontiguousarray(rgb.transpose(2,0,1))).float()
        mean=torch.tensor([123.675,116.28,103.53]).view(3,1,1)
        std=torch.tensor([58.395,57.12,57.375]).view(3,1,1)
        image_list=ImageList.from_tensors([(pixels-mean)/std],32,padding_constraints={'square_size':336})
        x=image_list.tensor.contiguous()
        assert tuple(x.shape)==(1,3,352,352)
        with torch.inference_mode():
            start=time.perf_counter();ref=model(x)['last_feat'].contiguous();elapsed=time.perf_counter()-start
        assert tuple(ref.shape)==(1,1408,22,22) and torch.isfinite(ref).all()
        from layout_attention import apply_layout
        apply_layout(model)
        with torch.inference_mode(): changed=model(x)['last_feat'].contiguous()
        layout_rel=float(torch.linalg.vector_norm(changed-ref)/torch.linalg.vector_norm(ref))
        layout_abs=float(torch.max(torch.abs(changed-ref)))
        if layout_rel>1e-4:raise RuntimeError('layout 변경 CPU 기준 출력 불일치')
        r.update(layout_cpu_relative_l2=layout_rel,layout_cpu_max_abs=layout_abs,implementation='동일 checkpoint/attention, 3D/4D layout 후보. window/head/블록 수 불변')
        print('원본 vs layout 후보 CPU 비교 통과',layout_rel,flush=True)
        np.save(OUT/'inputs/input_nchw.npy',x.numpy());np.save(OUT/'inputs/reference_nchw.npy',ref.numpy())
        wrapper=OutputTensor(model).eval();raw=OUT/'models/eva01_k4_352_raw.onnx'
        print('실제 가중치 ONNX 내보내기',flush=True)
        torch.onnx.export(wrapper,x,str(raw),opset_version=17,input_names=['input'],output_names=['feature'],do_constant_folding=True)
        onnx.checker.check_model(str(raw))
        opt=OUT/'models/eva01_k4_352.onnx'
        options=ort.SessionOptions();options.intra_op_num_threads=4;options.inter_op_num_threads=1
        options.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        options.optimized_model_filepath=str(opt)
        session=ort.InferenceSession(str(raw),sess_options=options,providers=['CPUExecutionProvider'])
        got=session.run(None,{'input':x.numpy()})[0];ref_np=ref.numpy()
        rel=float(np.linalg.norm(got-ref_np)/np.linalg.norm(ref_np));max_abs=float(np.max(np.abs(got-ref_np)))
        if rel>1e-4 or not np.isfinite(got).all():raise RuntimeError('ONNX CPU 출력 불일치')
        del session;gc.collect()
        graph=onnx.load(str(opt));onnx.checker.check_model(graph)
        domains=sorted(set(n.domain for n in graph.graph.node))
        if any(d not in ('','ai.onnx') for d in domains):raise RuntimeError('표준 ONNX 외 연산이 생성됨: '+str(domains))
        options=ort.SessionOptions();options.intra_op_num_threads=4;options.inter_op_num_threads=1
        verified=ort.InferenceSession(str(opt),sess_options=options,providers=['CPUExecutionProvider']).run(None,{'input':x.numpy()})[0]
        np.testing.assert_allclose(verified,got,rtol=1e-5,atol=1e-5)
        contract=dict(model='EVA-01 COCO detector prefix',checkpoint_sha256=SHA,source_revision='5623d08bb08a8268c7bc18e1d9b8b4bd651fe4d9',split_k=4,
            block_indices=[0,1,2,3],attention=['window16','window16','window16','global'],
            input_shape_nchw=[1,3,352,352],input_dtype='float32',output_shape_nchw=[1,1408,22,22],
            preprocessing=dict(color='RGB',verification_resize=[336,336],resize='OpenCV INTER_LINEAR direct resize for this probe',range_before_normalization='0..255',mean=[123.675,116.28,103.53],std=[58.395,57.12,57.375],padding='after normalization: bottom16/right16 value0',source_input_hw=[336,336],canvas_hw=[352,352]),
            input_boundary='already normalized and padded buffer; RKNN mean0/std1; do not normalize twice',position_embedding='included exactly once in prefix; initialized parameter shape kept at original checkpoint',
            output_boundary='after complete fourth residual block; NCHW; server must resume blocks[4] without redoing patch/position embedding',
            frame_sha256=digest(image_path),parameters=sum(p.numel() for p in model.parameters()),
            onnx_sha256=digest(opt),input_sha256=digest(OUT/'inputs/input_nchw.npy'),reference_sha256=digest(OUT/'inputs/reference_nchw.npy'),
            implementation='layout_v2: QKV separate Conv1x1, 3D relative bias additions, same weights and attention semantics',
            note='336/352 is a provisional probe configuration, not confirmation of training preprocessing or split choice; no FCM or server changes.')
        (OUT/'configs/contract.json').write_text(json.dumps(contract,ensure_ascii=False,indent=2))
        r.update(state='completed',parameters=contract['parameters'],checkpoint_sha256=SHA,onnx_sha256=contract['onnx_sha256'],input_shape=list(x.shape),output_shape=list(ref.shape),first_cpu_call_ms=elapsed*1000,onnx_max_abs=max_abs,onnx_relative_l2=rel,onnx_ops=sorted(set(n.op_type for n in graph.graph.node)),onnx_bytes=opt.stat().st_size)
        print('검증 완료:',OUT,flush=True)
    except BaseException as e:r.update(state='failed',error=repr(e));raise
    finally:
        (folder/'report.json').write_text(json.dumps(r,ensure_ascii=False,indent=2));print(json.dumps(r,ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':main()
