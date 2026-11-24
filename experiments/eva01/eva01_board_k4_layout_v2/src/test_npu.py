"""EVA 앞4블록의 실제 RK3588 실행·CPU 기준 특징 비교. FCM/서버/mAP 제외."""
from pathlib import Path
from datetime import datetime
import json,hashlib,time,resource
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def output_nchw(outputs):
    if outputs is None or len(outputs)!=1:raise RuntimeError('NPU 출력 개수 오류')
    y=np.asarray(outputs[0]);layout='NCHW'
    if y.shape==(1,22,22,1408):y=y.transpose(0,3,1,2);layout='NHWC→NCHW'
    if y.shape!=(1,1408,22,22):raise RuntimeError('출력 형상 오류: '+str(y.shape))
    if not np.isfinite(y).all():raise RuntimeError('NPU 출력 비유한값')
    return np.ascontiguousarray(y),layout
def main():
    out=ROOT/'runs'/('npu_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'));out.mkdir(parents=True)
    r=dict(state='running',stage='파일 검사',scope='실제 탐지 가중치 앞4블록, 고정 사진입력, NPU CORE0. API 호출만 시간 측정; 전처리·CPU검사·FCM·압축·전송·전체 탐지·mAP 제외.',warmup=3,measured_calls=5)
    engine=None
    def save():
        p=out/'report.json.tmp';p.write_text(json.dumps(r,ensure_ascii=False,indent=2));p.replace(out/'report.json')
    save();print('결과 폴더:',out,flush=True)
    try:
        c=json.loads((ROOT/'configs/contract.json').read_text());b=json.loads((ROOT/'configs/build_result.json').read_text());path=ROOT/'models/eva01_k4_352_fp.rknn'
        if digest(path)!=b['rknn_sha256']:raise RuntimeError('RKNN 해시 불일치')
        for name,key in [('input_nchw.npy','input_sha256'),('reference_nchw.npy','reference_sha256')]:
            if digest(ROOT/'inputs'/name)!=c[key]:raise RuntimeError(name+' 해시 불일치')
        x=np.load(ROOT/'inputs/input_nchw.npy',allow_pickle=False);ref=np.load(ROOT/'inputs/reference_nchw.npy',allow_pickle=False)
        if x.shape!=(1,3,352,352) or x.dtype!=np.float32:raise RuntimeError('입력 규격 오류')
        inputs=np.ascontiguousarray(x.transpose(0,2,3,1))
        from rknnlite.api import RKNNLite
        engine=RKNNLite();r['stage']='RKNN 로딩';save()
        rc=engine.load_rknn(str(path))
        if rc!=0:raise RuntimeError('load_rknn 실패: '+str(rc))
        r['stage']='NPU CORE0 초기화';save();print(r['stage'],flush=True)
        rc=engine.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
        if rc!=0:raise RuntimeError('init_runtime 실패: '+str(rc))
        durations=[];first=None
        for i in range(8):
            r['stage']=f'NPU API 호출 {i+1}/8';save();tick=time.perf_counter()
            outputs=engine.inference(inputs=[inputs],data_format=['nhwc'])
            elapsed=(time.perf_counter()-tick)*1000
            y,layout=output_nchw(outputs)
            if i==0:first=elapsed
            if i>=3:durations.append(elapsed)
            print(f'{i+1}/8: {elapsed:.2f}ms '+('워밍업' if i<3 else '측정'),flush=True)
        a=y.astype(np.float64).ravel();z=ref.astype(np.float64).ravel()
        delta=a-z;rel=float(np.linalg.norm(delta)/max(np.linalg.norm(z),1e-12));cos=float(np.dot(a,z)/max(np.linalg.norm(a)*np.linalg.norm(z),1e-12))
        screen=bool(rel<=.02 and cos>=.999)
        np.save(out/'npu_output_nchw.npy',y)
        r.update(state='completed' if screen else 'needs_review',stage='실제 NPU 실행 완료',execution_completed=True,
            checkpoint_sha256=c['checkpoint_sha256'],onnx_sha256=c['onnx_sha256'],rknn_sha256=b['rknn_sha256'],npu_core='NPU_CORE_0',split_k=4,
            input_shape=list(x.shape),output_shape=list(y.shape),output_layout_conversion=layout,output_dtype=str(y.dtype),first_api_ms=first,
            npu_api_ms=dict(samples=durations,mean=float(np.mean(durations)),median=float(np.median(durations)),p95=float(np.percentile(durations,95))),
            feature_comparison=dict(max_abs=float(np.max(np.abs(delta))),mean_abs=float(np.mean(np.abs(delta))),relative_l2=rel,cosine=cos),
            provisional_numeric_screen_pass=screen,numeric_screen_note='relativeL2<=0.02/cosine>=0.999는 특징 수치의 잠정 점검 기준이며 최종 탐지 정확도 합격 기준이 아님. 사진1장만 검사.')
    except BaseException as e:r.update(state='failed',error=repr(e));raise
    finally:
        if engine is not None:engine.release()
        r['process_peak_rss_mib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
        save();print(json.dumps(r,ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':main()
