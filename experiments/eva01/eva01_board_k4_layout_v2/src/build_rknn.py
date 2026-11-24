"""보드에서 EVA 실제 k4 ONNX를 RKNN으로 빌드. 빌드 성공은 NPU 실행 성공이 아님."""
from pathlib import Path
from datetime import datetime
import json,hashlib,time,resource
ROOT=Path(__file__).resolve().parents[1]
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def check(rc,name):
    if rc not in (None,0):raise RuntimeError(f'{name} 실패: {rc}')
def main():
    out=ROOT/'runs'/('build_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'));out.mkdir(parents=True)
    r=dict(state='running',stage='ONNX 무결성 검사',scope='실제 EVA 탐지 가중치의 앞4블록 RKNN 빌드 검사. NPU 실행·전체 모델·FCM·mAP 아님.')
    engine=None
    def save():
        p=out/'report.json.tmp';p.write_text(json.dumps(r,ensure_ascii=False,indent=2));p.replace(out/'report.json')
    def stage(s):r['stage']=s;save();print(s,flush=True)
    save();print('결과 폴더:',out,flush=True)
    try:
        c=json.loads((ROOT/'configs/contract.json').read_text());source=ROOT/'models/eva01_k4_352.onnx'
        if digest(source)!=c['onnx_sha256']:raise RuntimeError('ONNX 해시 불일치')
        r.update(onnx_sha256=c['onnx_sha256'],checkpoint_sha256=c['checkpoint_sha256'],input_shape=c['input_shape_nchw'],output_shape=c['output_shape_nchw'],split_k=4)
        from rknn.api import RKNN
        engine=RKNN(verbose=True)
        stage('RK3588 설정: 이미 정규화된 입력, mean0/std1')
        check(engine.config(target_platform='rk3588',mean_values=[[0,0,0]],std_values=[[1,1,1]]),'config')
        stage('실제 k4 ONNX 로딩');check(engine.load_onnx(model=str(source)),'load_onnx')
        stage('RKNN 빌드 시작: INT8 양자화 미사용');t=time.perf_counter()
        check(engine.build(do_quantization=False),'build');r['build_seconds']=time.perf_counter()-t
        target=ROOT/'models/eva01_k4_352_fp.rknn'
        stage('RKNN 파일 저장');check(engine.export_rknn(str(target)),'export_rknn')
        r.update(state='completed',stage='RKNN 빌드 완료. 다음은 실제 NPU 실행 검사',rknn_file=str(target),rknn_sha256=digest(target),rknn_bytes=target.stat().st_size,build_mode='do_quantization=False; 실제 정밀도·실행 배치는 런타임 로그 확인 필요')
        (ROOT/'configs/build_result.json').write_text(json.dumps(r,ensure_ascii=False,indent=2))
    except BaseException as e:r.update(state='failed',error=repr(e));raise
    finally:
        if engine is not None:engine.release()
        r['process_peak_rss_mib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
        save();print(json.dumps(r,ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':main()
