import argparse, json, random, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from models import FITS

class Windows(Dataset):
    def __init__(self, data, lo, hi, lookback, horizon):
        self.data=torch.from_numpy(data); self.starts=range(lo,hi-horizon+1)
        self.L=lookback; self.H=horizon
    def __len__(self): return len(self.starts)
    def __getitem__(self,i):
        s=self.starts[i]
        return self.data[s-self.L:s],self.data[s:s+self.H]

def load_data(dataset,p,mean=None,std=None):
    a=pd.read_csv(dataset)[p['features']].to_numpy(dtype=np.float64)
    assert np.isfinite(a).all() and len(a)>=p['split']['test_end']
    if mean is None:
        train=a[:p['split']['train_end']]; mean=train.mean(0); std=train.std(0,ddof=0)
        assert (std>0).all()
    return ((a-mean)/std).astype(np.float32),mean,std

def loader(a,p,split):
    s=p['split']; L=p['lookback']; H=p['horizon']
    lo,hi={'train':(L,s['train_end']),'validation':(s['train_end'],s['validation_end']),'test':(s['validation_end'],s['test_end'])}[split]
    return DataLoader(Windows(a,lo,hi,L,H),batch_size=p['training']['batch_size'],shuffle=(split=='train' and p['training']['training_shuffle']),drop_last=False,num_workers=0)

@torch.no_grad()
def evaluate(model,batches,p,save=False):
    model.eval(); se=ae=0.; n=0; predictions=[]
    for x,y in batches:
        out=model(x)[:,-p['horizon']:]
        e=out-y; se+=e.double().square().sum().item(); ae+=e.double().abs().sum().item(); n+=e.numel()
        if save: predictions.append(out.numpy())
    return {'mse':se/n,'mae':ae/n}, np.concatenate(predictions) if save else None

def predict(args,p):
    ck=torch.load(args.checkpoint,map_location='cpu')
    assert ck['protocol']==p
    a,_,_=load_data(args.dataset,p,ck['mean'],ck['std'])
    m=FITS(**ck['architecture']); m.load_state_dict(ck['state'])
    # Inference accesses only inputs, never test target values.
    starts=range(p['split']['validation_end'],p['split']['test_end']-p['horizon']+1)
    out=[]; bs=p['training']['batch_size']; L=p['lookback']
    m.eval()
    with torch.no_grad():
        for i in range(0,len(starts),bs):
            x=torch.from_numpy(np.stack([a[s-L:s] for s in starts[i:i+bs]]))
            out.append(m(x)[:,-p['horizon']:].numpy())
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    np.save(args.output,np.concatenate(out).astype(np.float32),allow_pickle=False)

def train(args,p):
    assert p['models']==['FITS']
    t=time.perf_counter(); random.seed(p['seed']); np.random.seed(p['seed']); torch.manual_seed(p['seed'])
    a,mean,std=load_data(args.dataset,p)
    arch={'lookback':p['lookback'],'horizon':p['horizon'],'cutoff':26}
    m=FITS(**arch); tr=loader(a,p,'train'); va=loader(a,p,'validation')
    controls=p['training']; assert controls['optimizer']=='Adam' and controls['schedule']=='constant'
    opt=torch.optim.Adam(m.parameters(),lr=controls['learning_rate'])
    best=float('inf'); stale=0; history=[]
    Path('checkpoints').mkdir(exist_ok=True); Path('output').mkdir(exist_ok=True)
    for epoch in range(1,controls['max_epochs']+1):
        m.train(); loss_sum=0.; count=0
        for x,y in tr:
            opt.zero_grad(); out=m(x); target=torch.cat([x,y],1)
            loss=(out-target).square().mean(); loss.backward(); opt.step()
            loss_sum+=loss.item()*target.numel(); count+=target.numel()
        metrics,_=evaluate(m,va,p); v=metrics['mse']
        row={'epoch':epoch,'train_backcast_forecast_mse':loss_sum/count,'validation_forecast_mse':v}
        history.append(row); print(json.dumps(row),flush=True)
        if v<best:
            best=v; best_epoch=epoch; stale=0
            torch.save({'state':m.state_dict(),'architecture':arch,'mean':mean,'std':std,'protocol':p,'epoch':epoch,'validation_mse':v},'checkpoints/FITS.pt')
        else: stale+=1
        if stale>=controls['patience']: break
    duration=time.perf_counter()-t
    ck=torch.load('checkpoints/FITS.pt',map_location='cpu'); m.load_state_dict(ck['state'])
    metrics,out=evaluate(m,loader(a,p,'test'),p,True)
    np.save('output/FITS.npy',out.astype(np.float32),allow_pickle=False)
    summary={'model':'FITS','training_seconds':duration,'best_epoch':best_epoch,'epochs_run':epoch,'validation_mse':best,'test':metrics,'shape':list(out.shape),'complex_parameters':sum(v.numel() for v in m.parameters()),'real_scalar_parameters':2*sum(v.numel() for v in m.parameters()),'history':history}
    Path('results.json').write_text(json.dumps(summary,indent=2)); print(json.dumps({k:v for k,v in summary.items() if k!='history'}),flush=True)

if __name__=='__main__':
    torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
    ap=argparse.ArgumentParser(); ap.add_argument('mode',choices=['train','predict']); ap.add_argument('--dataset',default='/inputs/dataset.csv'); ap.add_argument('--protocol',default='/inputs/protocol.json'); ap.add_argument('--checkpoint',default='checkpoints/FITS.pt'); ap.add_argument('--output',default='output/FITS.npy')
    args=ap.parse_args(); p=json.loads(Path(args.protocol).read_text())
    (train if args.mode=='train' else predict)(args,p)
