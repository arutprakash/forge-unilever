import argparse,json,time,random,os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset,DataLoader
from models import Forecast

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
class Windows(Dataset):
    def __init__(self,z,L,H,start,end):
        self.z=torch.from_numpy(z); self.L=L; self.H=H
        self.starts=range(start,end-H+1)
    def __len__(self): return len(self.starts)
    def __getitem__(self,i):
        s=self.starts[i]
        return self.z[s-self.L:s],self.z[s:s+self.H]
def read_data(dataset,p,mean=None,std=None):
    a=pd.read_csv(dataset).loc[:,p['features']].to_numpy(dtype=np.float64)
    assert len(a)>=p['split']['test_end'] and np.isfinite(a).all()
    if mean is None:
        train=a[:p['split']['train_end']]
        mean=train.mean(0); std=train.std(0,ddof=0)
    assert np.all(std>0)
    return ((a-mean)/std).astype(np.float32),mean,std
def loader(z,p,start,end,shuffle=False):
    return DataLoader(Windows(z,p['lookback'],p['horizon'],start,end),batch_size=p['training']['batch_size'],shuffle=shuffle,drop_last=False,num_workers=0)
@torch.no_grad()
def evaluate(m,dl,predict=False):
    m.eval(); sq=ab=0.; n=0; out=[]
    for x,y in dl:
        pred=m(x); d=(pred-y).double(); sq+=d.square().sum().item(); ab+=d.abs().sum().item(); n+=d.numel()
        if predict: out.append(pred.numpy())
    return {'mse':sq/n,'mae':ab/n}, np.concatenate(out) if predict else None

def train(args,p):
    seed=p['seed']; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    cfg=p['training']; assert cfg['optimizer']=='Adam' and cfg['schedule']=='constant'
    assert cfg['drop_last']==False and cfg['checkpoint_selection']=='lowest validation MSE'
    z,mean,std=read_data(args.dataset,p); s=p['split']
    tr=loader(z,p,p['lookback'],s['train_end'],cfg['training_shuffle'])
    va=loader(z,p,s['train_end'],s['validation_end'])
    m=Forecast(args.model,p['lookback'],p['horizon'])
    expected=p['lookback']*p['horizon']*(2 if args.model=='DLinear' else 1)
    assert sum(x.numel() for x in m.parameters())==expected
    opt=torch.optim.Adam(m.parameters(),lr=cfg['learning_rate'])
    best=float('inf'); stale=0; began=time.perf_counter(); selected=None
    os.makedirs('checkpoints',exist_ok=True); os.makedirs('output',exist_ok=True)
    path='checkpoints/'+args.model+'.pt'
    print(json.dumps({'model':args.model,'parameters':expected,'train_windows':len(tr.dataset),'validation_windows':len(va.dataset)}),flush=True)
    for epoch in range(1,cfg['max_epochs']+1):
        m.train(); total=0.; count=0
        for x,y in tr:
            opt.zero_grad(); pred=m(x); loss=(pred-y).square().mean(); loss.backward(); opt.step()
            total+=loss.item()*y.numel(); count+=y.numel()
        metrics,_=evaluate(m,va)
        print(json.dumps({'epoch':epoch,'train_mse':total/count,'validation':metrics}),flush=True)
        if metrics['mse']<best:
            best=metrics['mse']; stale=0; selected=epoch
            torch.save({'state_dict':m.state_dict(),'model':args.model,'protocol':p,'mean':mean,'std':std,'epoch':epoch,'validation_mse':best},path)
        else: stale+=1
        if stale>=cfg['patience']: break
    duration=time.perf_counter()-began
    ck=torch.load(path,map_location='cpu'); m.load_state_dict(ck['state_dict'])
    test,arr=evaluate(m,loader(z,p,s['validation_end'],s['test_end']),True)
    np.save('output/'+args.model+'.npy',arr,allow_pickle=False)
    result={'model':args.model,'duration_seconds':duration,'epochs_run':epoch,'selected_epoch':selected,'validation_mse':best,'test':test,'shape':list(arr.shape)}
    with open('output/'+args.model+'_metadata.json','w') as f: json.dump(result,f,indent=2)
    print(json.dumps(result),flush=True)
def predict(args,p):
    ck=torch.load(args.checkpoint,map_location='cpu')
    assert ck['protocol']==p
    z,_,_=read_data(args.dataset,p,ck['mean'],ck['std'])
    m=Forecast(ck['model'],p['lookback'],p['horizon']); m.load_state_dict(ck['state_dict']); m.eval()
    # Replay accesses inputs only; no target values or metrics are needed.
    out=[]; s=p['split']; L=p['lookback']; H=p['horizon']; bs=p['training']['batch_size']
    with torch.no_grad():
        for first in range(s['validation_end'],s['test_end']-H+1,bs):
            starts=range(first,min(first+bs,s['test_end']-H+1))
            x=torch.from_numpy(np.stack([z[t-L:t] for t in starts]))
            out.append(m(x).numpy())
    os.makedirs(os.path.dirname(os.path.abspath(args.output)),exist_ok=True)
    np.save(args.output,np.concatenate(out),allow_pickle=False)
if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('mode',choices=['train','predict']); ap.add_argument('--model'); ap.add_argument('--checkpoint'); ap.add_argument('--output'); ap.add_argument('--dataset',default='/inputs/dataset.csv'); ap.add_argument('--protocol',default='/inputs/protocol.json'); args=ap.parse_args()
    with open(args.protocol) as f: p=json.load(f)
    if args.mode=='train':
        assert args.model in p['models']; train(args,p)
    else: predict(args,p)
