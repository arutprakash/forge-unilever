import argparse,json,time,random,os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset,DataLoader
from models import CrossGNN
class Windows(Dataset):
    def __init__(self,a,L,H,start,end):
        self.a=torch.tensor(a,dtype=torch.float32);self.L=L;self.H=H;self.starts=range(max(start,L),end-H+1)
    def __len__(self):return len(self.starts)
    def __getitem__(self,i):
        t=self.starts[i];return self.a[t-self.L:t],self.a[t:t+self.H]
def setup(dataset,protocol,stats=None):
    p=json.load(open(protocol)); a=pd.read_csv(dataset)[p['features']].to_numpy(dtype=np.float64)
    assert np.isfinite(a).all()
    tr=p['split']['train_end']
    mean=a[:tr].mean(0) if stats is None else np.array(stats['mean'])
    std=a[:tr].std(0,ddof=0) if stats is None else np.array(stats['std'])
    assert (std>0).all()
    return p,((a-mean)/std).astype('float32'),{'mean':mean.tolist(),'std':std.tolist()}
def evaluate(m,loader,collect=False):
    m.eval();ss=sa=count=0;out=[]
    with torch.no_grad():
        for x,y in loader:
            z=m(x);ss+=(z-y).square().sum().item();sa+=(z-y).abs().sum().item();count+=y.numel()
            if collect:out.append(z.numpy())
    return ss/count,sa/count,np.concatenate(out) if collect else None
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',default='/inputs/dataset.csv');ap.add_argument('--protocol',default='/inputs/protocol.json');ap.add_argument('--checkpoint',default='best_CrossGNN.pt');ap.add_argument('--output',default='output/CrossGNN.npy');ap.add_argument('--predict',action='store_true');args=ap.parse_args()
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    if args.predict:
        ck=torch.load(args.checkpoint,map_location='cpu');p,a,stats=setup(args.dataset,args.protocol,ck['stats'])
    else:p,a,stats=setup(args.dataset,args.protocol)
    seed=p['seed'];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    assert p['models']==['CrossGNN']
    L=p['lookback'];H=p['horizon'];s=p['split'];t=p['training'];m=CrossGNN(L,H,len(p['features']))
    def loader(start,end,shuffle=False):return DataLoader(Windows(a,L,H,start,end),batch_size=t['batch_size'],shuffle=shuffle,drop_last=False,num_workers=0)
    if not args.predict:
        train=loader(L,s['train_end'],t['training_shuffle']);val=loader(s['train_end'],s['validation_end'])
        opt=torch.optim.Adam(m.parameters(),lr=t['learning_rate']);best=float('inf');wait=0;begin=time.perf_counter();history=[]
        print('parameters',sum(v.numel() for v in m.parameters()),'train_windows',len(train.dataset),'validation_windows',len(val.dataset),flush=True)
        for ep in range(1,t['max_epochs']+1):
            start=time.perf_counter();m.train();total=count=0
            for x,y in train:
                opt.zero_grad();z=m(x);loss=(z-y).square().mean();loss.backward();opt.step();total+=loss.item()*y.numel();count+=y.numel()
            v,_,_=evaluate(m,val);row={'epoch':ep,'train_mse':total/count,'validation_mse':v,'seconds':time.perf_counter()-start};history.append(row);print(json.dumps(row),flush=True)
            if v<best:
                best=v;wait=0;torch.save({'state_dict':m.state_dict(),'stats':stats,'protocol':p,'epoch':ep,'validation_mse':v},args.checkpoint)
            else:wait+=1
            if wait>=t['patience']:break
        ck=torch.load(args.checkpoint,map_location='cpu')
        meta={'training_seconds':time.perf_counter()-begin,'selected_epoch':ck['epoch'],'validation_mse':best,'parameters':sum(v.numel() for v in m.parameters()),'history':history}
        json.dump(meta,open('training_metadata.json','w'),indent=2)
    m.load_state_dict(ck['state_dict']);test=loader(s['validation_end'],s['test_end']);mse,mae,pred=evaluate(m,test,True)
    os.makedirs(os.path.dirname(args.output) or '.',exist_ok=True);np.save(args.output,pred.astype(np.float32))
    print(json.dumps({'test_mse':mse,'test_mae':mae,'shape':list(pred.shape),'first_target':test.dataset.starts[0],'last_target':test.dataset.starts[-1]}),flush=True)
if __name__=='__main__':main()
