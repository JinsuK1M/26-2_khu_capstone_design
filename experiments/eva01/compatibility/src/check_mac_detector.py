"""맥북 CPU에서 EVA-01 전체 검출기 실행. 분할·FCM·mAP는 평가하지 않음."""
from pathlib import Path
from datetime import datetime
import argparse,json,hashlib,platform,resource,time,sys,types,gc,os
os.environ.setdefault("WANDB_MODE","disabled")
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'vendor/EVA/EVA-01/det'
SHA='2cbbd46259851bad0052b418464cd866097e0e01bb60ae368b213ab040960650'
UNAVAILABLE_CALLS=[]
def guard_unused_soft_nms():
    # EVA의 기본 NMS 경로는 torchvision 계산을 사용한다.
    # import만 필요한 미설치 Soft-NMS를 호출하면 명시적으로 실패한다.
    ops=types.ModuleType('mmcv.ops')
    def unavailable(*args,**kwargs):
        UNAVAILABLE_CALLS.append('mmcv.ops.soft_nms')
        raise RuntimeError('Soft-NMS는 이번 CPU 기본 NMS 검사에서 지원·실행하지 않았습니다.')
    ops.soft_nms=unavailable
    sys.modules['mmcv.ops']=ops

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--size',type=int,choices=[640,1280],default=640);p.add_argument('--image',type=Path,default=ROOT/'vendor/bus.jpg');a=p.parse_args()
    folder=ROOT/'reports'/('mac_detector_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'));folder.mkdir(parents=True)
    r=dict(state='running',stage='프레임워크 import',device='cpu',dtype='float32',input_size=a.size,threads=4,
        scope='실제 사전학습 EVA-01 전체 ViT+피라미드+RPN+Cascade ROI+mask 실행. CPU 기본NMS 경로, confidence0.25, 실제사진1장. Soft-NMS·MPS·분할·FCM·mAP·정식속도평가 아님.')
    def save():
        t=folder/'report.json.tmp';t.write_text(json.dumps(r,ensure_ascii=False,indent=2));t.replace(folder/'report.json')
    def stage(s):r['stage']=s;save();print(s,flush=True)
    print('결과:',folder,flush=True);save()
    try:
        torch.set_num_threads(4);torch.manual_seed(20261123)
        guard_unused_soft_nms();sys.path.insert(0,str(SOURCE))
        from PIL import Image
        # 구형 Detectron2의 보간 이름만 같은 Pillow 연산으로 연결.
        for name,target in [('LINEAR','BILINEAR'),('CUBIC','BICUBIC'),('ANTIALIAS','LANCZOS')]:
            if not hasattr(Image,name):setattr(Image,name,getattr(Image.Resampling,target))
        import detectron2,torchvision,cv2,timm
        from detectron2.config import LazyConfig,instantiate
        # 형식만 흉내 내지 않고 실제 torchvision CPU 계산을 먼저 검사.
        from torchvision.ops import nms,roi_align
        keep=nms(torch.tensor([[0.,0.,10.,10.],[0.,0.,10.,10.]]),torch.tensor([.9,.8]),.5)
        assert keep.tolist()==[0]
        roi=roi_align(torch.ones(1,1,8,8),[torch.tensor([[1.,1.,6.,6.]])],(2,2),aligned=True)
        torch.testing.assert_close(roi,torch.ones_like(roi))
        r.update(torch=torch.__version__,torchvision=torchvision.__version__,timm=timm.__version__,detectron2=detectron2.__version__,machine=platform.machine(),cpu_nms_roi_align_check=True)
        stage('가중치 SHA256 확인')
        path=ROOT/'models/eva_coco_det.pth'
        if path.stat().st_size!=4295207874:raise RuntimeError('가중치 크기 불일치')
        h=hashlib.sha256()
        with path.open('rb') as f:
            for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
        if h.hexdigest()!=SHA:raise RuntimeError('가중치 해시 불일치')
        r['checkpoint_sha256']=SHA
        stage('공식 전체 검출기 CPU 구성')
        cfg=LazyConfig.load(str(SOURCE/'projects/ViTDet/configs/COCO/cascade_mask_rcnn_vitdet_eva.py'))
        cfg.model.backbone.square_pad=a.size
        # img_size1280은 체크포인트 상대위치 파라미터 형상 유지용. forward에서 실제크기로 처리.
        cfg.model.backbone.net.use_act_checkpoint=False
        cfg.model.backbone.net.drop_path_rate=0.
        for head in cfg.model.roi_heads.box_predictors:head.test_score_thresh=.25
        cfg.model.roi_heads.use_soft_nms=False
        model=instantiate(cfg.model);model.to('cpu');model.eval()
        stage('실제 탐지 체크포인트 전체 strict 적용')
        loaded=torch.load(str(path),map_location='cpu',weights_only=True,mmap=True)
        state=loaded['model']
        actual=model.state_dict()
        mismatched=[k for k,v in actual.items() if k in state and v.shape!=state[k].shape]
        r['state_dict_missing']=sorted(set(actual)-set(state));r['state_dict_extra']=sorted(set(state)-set(actual));r['state_dict_mismatched']=mismatched
        if r['state_dict_missing'] or r['state_dict_extra'] or mismatched:raise RuntimeError('모델과 체크포인트 규격 불일치: report 참조')
        del actual
        model.load_state_dict(state,strict=True,assign=True)
        r['parameters']=sum(x.numel() for x in model.parameters())
        del loaded,state;gc.collect()
        image=cv2.imread(str(a.image))
        if image is None:raise RuntimeError('입력 사진 읽기 실패')
        r.update(image=str(a.image),image_sha256=hashlib.sha256(a.image.read_bytes()).hexdigest(),source_hw=list(image.shape[:2]))
        # 실행 가능성 검사에 한해 직접640 리사이즈. 공식1536 mAP 조건과 다름.
        rgb=cv2.cvtColor(cv2.resize(image,(a.size,a.size)),cv2.COLOR_BGR2RGB)
        inp=torch.from_numpy(np.ascontiguousarray(rgb.transpose(2,0,1))).float()
        hooks=[];timings={};starts={}
        for name in ('backbone','proposal_generator','roi_heads'):
            module=getattr(model,name)
            def before(mod,inputs,n=name):stage('전체 검출기 계산: '+n);starts[n]=time.perf_counter()
            def after(mod,inputs,output,n=name):timings[n]=time.perf_counter()-starts[n]
            hooks += [module.register_forward_pre_hook(before),module.register_forward_hook(after)]
        stage('실제 사진 전체 객체 검출 시작')
        tick=time.perf_counter()
        with torch.inference_mode():outputs=model([{'image':inp,'height':image.shape[0],'width':image.shape[1]}])
        elapsed=time.perf_counter()-tick
        for hook in hooks:hook.remove()
        if len(outputs)!=1 or 'instances' not in outputs[0]:raise RuntimeError('탐지 결과 형식 오류')
        result=outputs[0]['instances'].to('cpu')
        if not result.has('pred_boxes') or not result.has('scores') or not result.has('pred_classes'):raise RuntimeError('최종 탐지 필드 누락')
        if not torch.isfinite(result.pred_boxes.tensor).all() or not torch.isfinite(result.scores).all():raise RuntimeError('최종 결과 비유한값')
        if not result.has('pred_masks'):raise RuntimeError('mask 경로 결과 미생성')
        if UNAVAILABLE_CALLS:raise RuntimeError('미지원 연산 호출')
        boxes=[dict(class_id=int(c),score=float(s),box=[float(v) for v in b]) for c,s,b in zip(result.pred_classes,result.scores,result.pred_boxes.tensor)]
        (folder/'detections.json').write_text(json.dumps(boxes,indent=2))
        r.update(state='completed',stage='실제 전체 객체 검출기 CPU 실행 완료',first_full_detector_s=elapsed,stage_seconds=timings,detection_count=len(result),has_masks=True,mask_shape=list(result.pred_masks.shape),soft_nms_tested=False,unavailable_calls=UNAVAILABLE_CALLS)
    except BaseException as e:r.update(state='failed',error=repr(e));raise
    finally:
        peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss;r['process_peak_rss_mib']=peak/(1024**2 if platform.system()=='Darwin' else 1024)
        save();print(json.dumps(r,ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':main()
