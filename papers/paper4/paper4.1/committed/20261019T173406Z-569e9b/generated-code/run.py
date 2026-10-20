import argparse, json, random, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from models import TimeBase

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

class Windows(Dataset):
    def __init__(self, data, lookback, horizon, start, end):
        self.data, self.lookback, self.horizon = data, lookback, horizon
        self.starts = range(max(lookback,start),end-horizon+1)
    def __len__(self): return len(self.starts)
    def __getitem__(self,i):
        s=self.starts[i]
        return self.data[s-self.lookback:s],self.data[s:s+self.horizon]

def data_load(path,p,stats=None):
    a=pd.read_csv(path)[p['features']].to_numpy(dtype=np.float64)
    assert len(a)>=p['split']['test_end'] and np.isfinite(a).all()
    if stats is None:
        train=a[:p['split']['train_end']]
        mean,std=train.mean(0),train.std(0,ddof=0)
        std=np.where(std==0,1.,std)
    else: mean,std=np.array(stats['mean']),np.array(stats['std'])
    return torch.from_numpy(((a-mean)/std).astype(np.float32)), {'mean':mean.tolist(),'std':std.tolist()}

def loader(a,p,start,end,shuffle=False):
    return DataLoader(Windows(a,p['lookback'],p['horizon'],start,end),batch_size=p['training']['batch_size'],shuffle=shuffle,drop_last=False,num_workers=0)

def evaluate(model,dl,predict=False):
    model.eval(); se=ae=n=0; out=[]
    with torch.no_grad():
        for x,y in dl:
            z=model(x)
            if predict: out.append(z.numpy())
            d=(z-y).double(); se+=d.square().sum().item(); ae+=d.abs().sum().item(); n+=d.numel()
    return {'mse':se/n,'mae':ae/n}, np.concatenate(out) if predict else None

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--mode',choices=['train','predict'],default='train'); ap.add_argument('--dataset',default='/inputs/dataset.csv'); ap.add_argument('--protocol',default='/inputs/protocol.json'); ap.add_argument('--checkpoint',default='checkpoints/TimeBase.pt'); ap.add_argument('--output',default='output/TimeBase.npy'); args=ap.parse_args()
    p=json.load(open(args.protocol)); assert p['models']==['TimeBase']
    t=p['training']; random.seed(p['seed']); np.random.seed(p['seed']); torch.manual_seed(p['seed']); torch.use_deterministic_algorithms(True)
    if args.mode=='train':
        a,stats=data_load(args.dataset,p); model=TimeBase(p['lookback'],p['horizon'])
        train=loader(a,p,0,p['split']['train_end'],t['training_shuffle']); val=loader(a,p,p['split']['train_end'],p['split']['validation_end'])
        opt=torch.optim.Adam(model.parameters(),lr=t['learning_rate'])
        best=float('inf'); bad=0; history=[]; begin=time.perf_counter()
        print(json.dumps({'parameters':sum(v.numel() for v in model.parameters()),'train_windows':len(train.dataset),'validation_windows':len(val.dataset),'torch':torch.__version__}),flush=True)
        for epoch in range(1,t['max_epochs']+1):
            schedule=t['schedule']; lr=t['learning_rate']
            if isinstance(schedule,dict):
                assert schedule['type']=='exponential'
                lr*=schedule['factor']**max(0,epoch-schedule['start_epoch']+1)
            else: assert schedule=='constant'
            for group in opt.param_groups: group['lr']=lr
            model.train(); total=n=0; epstart=time.perf_counter()
            for x,y in train:
                opt.zero_grad(); z,orth=model(x,True); loss=(z-y).square().mean()+0.04*orth
                assert torch.isfinite(loss)
                loss.backward(); opt.step(); total+=loss.item()*len(x); n+=len(x)
            metrics,_=evaluate(model,val)
            row={'epoch':epoch,'lr':lr,'training_objective':total/n,'validation_mse':metrics['mse'],'epoch_seconds':time.perf_counter()-epstart}
            history.append(row); print(json.dumps(row),flush=True)
            if metrics['mse']<best:
                best=metrics['mse']; bad=0
                Path(args.checkpoint).parent.mkdir(parents=True,exist_ok=True)
                torch.save({'state_dict':model.state_dict(),'stats':stats,'protocol':p,'architecture':{'lookback':p['lookback'],'horizon':p['horizon'],'period':24,'rank':6},'epoch':epoch,'validation_mse':best},args.checkpoint)
            else: bad+=1
            if bad>=t['patience']: break
        duration=time.perf_counter()-begin
    ck=torch.load(args.checkpoint,map_location='cpu'); assert ck['protocol']==p
    model=TimeBase(**ck['architecture']); model.load_state_dict(ck['state_dict'])
    a,_=data_load(args.dataset,p,ck['stats']); test=loader(a,p,p['split']['validation_end'],p['split']['test_end'])
    metrics,preds=evaluate(model,test,True)
    assert preds.shape==(p['split']['test_end']-p['split']['validation_end']-p['horizon']+1,p['horizon'],len(p['features'])) and np.isfinite(preds).all()
    Path(args.output).parent.mkdir(parents=True,exist_ok=True); np.save(args.output,preds,allow_pickle=False)
    result={'selected_epoch':ck['epoch'],'validation_mse':ck['validation_mse'],'test':metrics,'shape':list(preds.shape)}
    if args.mode=='train':
        result.update(training_seconds=duration,history=history)
        Path('results.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='history'}),flush=True)
if __name__=='__main__': main()
