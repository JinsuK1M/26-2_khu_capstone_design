"""공식 EVA-01 사전학습 탐지 백본 40블록 CPU 검사. 탐지 헤드는 포함하지 않음."""
import argparse,json,hashlib,platform,resource,time,gc
from datetime import datetime
from pathlib import Path
import torch
from load_official_backbone import ROOT,build_backbone
EXPECTED='2cbbd46259851bad0052b418464cd866097e0e01bb60ae368b213ab040960650'
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--size',type=int,choices=[640,1280],default=1280);a=p.parse_args()
    out=ROOT/'reports'/('backbone_cpu_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'));out.mkdir(parents=True)
    r=dict(state='running',stage='체크포인트 검증',input_shape=[1,3,a.size,a.size],scope='공식 탐지 체크포인트의 ViT 백본 전체40블록·CPU FP32·합성입력1회. FPN/RPN/ROI 탐지부·mAP·NPU·분할 실행은 포함하지 않음.',initialization_img_size=1280,device='cpu',torch=torch.__version__,platform=platform.platform(),threads=4)
    def save():
        tmp=out/'report.json.tmp';tmp.write_text(json.dumps(r,ensure_ascii=False,indent=2));tmp.replace(out/'report.json')
    def stage(s):r['stage']=s;save();print(s,flush=True)
    print('결과 폴더:',out,flush=True);save()
    try:
        f=a.checkpoint.expanduser().resolve()
        if f.stat().st_size!=4295207874:raise RuntimeError('가중치 크기 불일치')
        h=hashlib.sha256()
        with f.open('rb') as stream:
            for chunk in iter(lambda:stream.read(8*1024**2),b''):h.update(chunk)
        if h.hexdigest()!=EXPECTED:raise RuntimeError('SHA256 불일치')
        r['checkpoint_sha256']=EXPECTED
        torch.set_num_threads(4);torch.manual_seed(20261123)
        stage('공식 전체 백본 meta 구성')
        model=build_backbone(size=1280,meta=True)
        stage('공식 탐지 가중치 mmap 및 백본 strict 적용')
        loaded=torch.load(str(f),map_location='cpu',weights_only=True,mmap=True)
        state={k.removeprefix('backbone.net.'):v for k,v in loaded['model'].items() if k.startswith('backbone.net.')}
        if any(v.dtype!=torch.float32 for v in state.values()):raise RuntimeError('FP32 체크포인트 아님')
        model.load_state_dict(state,strict=True,assign=True)
        if any(v.device.type=='meta' for v in model.parameters()):raise RuntimeError('meta 파라미터 잔존')
        model.eval();r['parameters']=sum(v.numel() for v in model.parameters())
        del loaded,state;gc.collect()
        image=torch.randn(1,3,a.size,a.size)
        stage('실제 가중치 전체 백본40블록 CPU 추론')
        start=time.perf_counter()
        with torch.inference_mode():result=model(image)['last_feat']
        elapsed=time.perf_counter()-start
        if not torch.isfinite(result).all():raise RuntimeError('비유한값 출력')
        r.update(state='completed',stage='전체 백본 CPU 실행 완료',first_backbone_inference_s=elapsed,output_shape=list(result.shape),interpretation='해당 조건에서 전체 백본은 실행 가능. 전체 객체 검출기 실행 성공 여부는 FPN/RPN/ROI까지 별도 검사 필요.')
    except BaseException as e:r.update(state='failed',error=repr(e));raise
    finally:
        peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss;r['process_peak_rss_mib']=peak/(1024**2 if platform.system()=='Darwin' else 1024)
        save();print(json.dumps(r,ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':main()
