from pathlib import Path
import sys,argparse,json,hashlib,time,platform
import numpy as np
import scipy
from scipy.io import loadmat
import torch
B=Path(__file__).resolve().parent
sys.path.insert(0,str(B/'source'))
from common import ST_SRI_Interpreter
from experiments.bspc_revision.collect_st_sri_curves import build_variant_model,normalized_record_window
from experiments.bspc_revision.leakage_free_db2 import build_trial_records,build_window_records
from experiments.bspc_revision.onset_protocol import build_onset_audit_records
from experiments.bspc_revision.st_sri_detector import DetectorConfig,curve_statistics
ap=argparse.ArgumentParser();ap.add_argument('--raw-mat',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ah(x):
 x=np.ascontiguousarray(x);return hashlib.sha256((str(x.dtype)+str(x.shape)).encode()+x.tobytes()).hexdigest()
torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.manual_seed(20261004);torch.use_deterministic_algorithms(True)
ck=torch.load(B/'S01_best.pth',map_location='cpu',weights_only=False);m=build_variant_model('trained',ck,0).eval();mat=loadmat(a.raw_mat,variable_names=['emg','restimulus']);data=mat['emg'];labels=mat['restimulus'].reshape(-1);mu=np.asarray(ck['training_mean'],np.float32);sd=np.asarray(ck['training_std'],np.float32)
tr=build_trial_records(labels,1,20260815,600);recs=build_onset_audit_records(tr,labels,'audit',[0,50,100,150]);r=next(r for r in recs if r.segment_index==11 and r.phase_ms==0);x=normalized_record_window(data,r,mu,sd)
rest=[r for r in build_window_records(labels,[t for t in tr if t.split=='train'],600,100) if r.modal_label==0];ix=np.random.default_rng(20261816).choice(len(rest),64,replace=False);bg=torch.from_numpy(np.stack([(np.asarray(data[rest[int(i)].raw_start:rest[int(i)].raw_end],np.float32)-mu)/sd for i in ix]));interp=ST_SRI_Interpreter(m,bg,device='cpu')
with torch.no_grad():prob=torch.softmax(m(x.unsqueeze(0)),1)[0].numpy()
target=int(prob.argmax());curves=[];seconds=[]
for repeat in range(3):
 t=time.perf_counter();lags,c,n=interp.scan_fast(x,max_lag_ms=150,stride=1,block_size=10,current_endpoint=599,target_cls=target);seconds.append(time.perf_counter()-t);curves.append(c)
st=curve_statistics(np.asarray(lags),c,DetectorConfig());np.savez_compressed(a.output/'curves.npz',lags_ms=lags,positive=curves,probabilities=prob)
s={'source_scope':'source from Zenodo deposited repository ZIP plus release code-and-manifest ZIP; checkpoint extracted from local ZIP whose SHA-256 equals GitHub API asset digest; user-provided licensed NinaPro raw MAT outside the package','checkpoint_sha256':sha(B/'S01_best.pth'),'raw_mat_sha256':sha(a.raw_mat),'source_sha256':{str(p.relative_to(B/'source')):sha(p) for p in (B/'source').rglob('*.py')},'runner_sha256':sha(__file__),'window':'S01_onset_seg011_ph0','target':target,'target_probability':float(prob[target]),'statistics':st,'support':bool(st['positive_mass']>.4249633979052305 and st['peak_height']>.09494568914491236),'background_ids':ix.tolist(),'template_sha256':ah(bg.mean(0).numpy()),'window_sha256':ah(x.numpy()),'repeat_max_difference':float(np.max(np.ptp(curves,axis=0))),'scan_seconds':seconds,'python':sys.version,'torch':torch.__version__,'numpy':np.__version__,'scipy':scipy.__version__,'platform':platform.platform(),'cpu_threads':2,'command':sys.argv}
(a.output/'receipt.json').write_text(json.dumps(s,indent=2));print(json.dumps({k:s[k] for k in ['target_probability','statistics','support','repeat_max_difference','scan_seconds']},indent=2))
