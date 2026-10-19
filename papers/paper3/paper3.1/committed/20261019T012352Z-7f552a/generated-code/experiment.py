import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
import argparse,json,time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.linalg import lstsq

def load(dataset,protocol):
    p=json.load(open(protocol))
    a=pd.read_csv(dataset)[p['features']].to_numpy(dtype=np.float64)
    assert np.isfinite(a).all()
    assert len(a)>=p['split']['test_end']
    return p,a

def windows(z,starts,L,H,targets=True):
    x=np.stack([z[s-L:s].T for s in starts])
    y=np.stack([z[s:s+H].T for s in starts]) if targets else None
    return x,y

def design(x,model):
    if model=='OLS':
        return np.concatenate([x,np.ones((len(x),1))],axis=1),0.
    mean=x.mean(axis=1,keepdims=True)
    return np.concatenate([x-mean,x.std(axis=1,ddof=0,keepdims=True)],axis=1),mean

def predict(z,p,ck,start,end):
    L,H=p['lookback'],p['horizon']; out=[]
    for s in range(start,end-H+1,p['training']['batch_size']):
        starts=range(s,min(s+p['training']['batch_size'],end-H+1))
        x,_=windows(z,starts,L,H,False)
        n,c,_=x.shape
        d,m=design(x.reshape(-1,L),str(ck['model']))
        out.append((d@ck['coef']+m).reshape(n,c,H).transpose(0,2,1))
    return np.concatenate(out)

def metric(pred,z,start,H):
    target=np.stack([z[s:s+H] for s in range(start,start+len(pred))])
    err=pred-target
    return {'mse':float(np.mean(err**2)),'mae':float(np.mean(abs(err)))}

def train(args,p,a,model):
    np.random.seed(p['seed']); t=time.perf_counter()
    assert p['training']['optimizer']=='OLS'
    L,H=p['lookback'],p['horizon']; sp=p['split']; te=sp['train_end']
    mean=a[:te].mean(0);std=a[:te].std(0,ddof=0)
    assert (std>0).all()
    z=(a-mean)/std
    starts=np.arange(L,te-H+1)
    if p['training']['training_shuffle']: np.random.shuffle(starts)
    x,y=windows(z,starts,L,H)
    d,m=design(x.reshape(-1,L),model)
    y=y.reshape(-1,H)-m
    coef,res,rank,singular=lstsq(d,y,cond=max(d.shape)*np.finfo(np.float64).eps,lapack_driver='gelsd',overwrite_a=True,overwrite_b=True)
    fit_seconds=time.perf_counter()-t
    ck={'model':np.array(model),'coef':coef,'mean':mean,'std':std,'lookback':np.array(L),'horizon':np.array(H)}
    val=predict(z,p,ck,te,sp['validation_end'])
    vm=metric(val,z,te,H)
    assert np.isfinite(vm['mse'])
    Path('checkpoints').mkdir(exist_ok=True)
    path=f'checkpoints/{model}.npz'
    np.savez(path,**ck)
    # Only candidate: selected by finite validation MSE before any test inference.
    with np.load(path,allow_pickle=False) as saved:
        test=predict(z,p,saved,sp['validation_end'],sp['test_end'])
    Path('output').mkdir(exist_ok=True);np.save(f'output/{model}.npy',test)
    result={'model':model,'fit_seconds':fit_seconds,'total_seconds':time.perf_counter()-t,'selected_epoch':1,'candidate_count':1,'rank':int(rank),'stored_parameters':int(coef.size),'train_windows':len(starts),'validation_windows':len(val),'test_windows':len(test),'validation':vm,'test':metric(test,z,sp['validation_end'],H)}
    Path('metadata').mkdir(exist_ok=True)
    Path(f'metadata/{model}.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['train','predict']);ap.add_argument('--model');ap.add_argument('--checkpoint');ap.add_argument('--dataset',default='/inputs/dataset.csv');ap.add_argument('--protocol',default='/inputs/protocol.json');ap.add_argument('--output');args=ap.parse_args()
    p,a=load(args.dataset,args.protocol)
    if args.mode=='train':
        for model in ([args.model] if args.model else p['models']):
            assert model in p['models'];train(args,p,a,model)
    else:
        with np.load(args.checkpoint,allow_pickle=False) as ck:
            assert str(ck['model']) in p['models']
            assert int(ck['lookback'])==p['lookback'] and int(ck['horizon'])==p['horizon']
            z=(a-ck['mean'])/ck['std'];s=p['split']
            pred=predict(z,p,ck,s['validation_end'],s['test_end'])
        assert pred.shape==(s['test_end']-s['validation_end']-p['horizon']+1,p['horizon'],len(p['features']))
        assert np.isfinite(pred).all()
        Path(args.output).parent.mkdir(parents=True,exist_ok=True);np.save(args.output,pred)
if __name__=='__main__':main()
