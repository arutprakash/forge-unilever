import torch
from torch import nn
from torch.nn import functional as F

class CrossGNN(nn.Module):
    def __init__(self,lookback,horizon,features):
        super().__init__()
        self.L=lookback; self.S=5; self.K=10; self.C=8
        self.maxnodes=self.S*(lookback//2)
        self.embed=nn.Linear(1,8)
        self.t1=nn.Parameter(torch.randn(self.maxnodes,8)*0.1)
        self.t2=nn.Parameter(torch.randn(8,self.maxnodes)*0.1)
        self.v1=nn.Parameter(torch.randn(features,8)*0.1)
        self.v2=nn.Parameter(torch.randn(8,features)*0.1)
        self.time=nn.Linear(16,8); self.var=nn.Linear(16,8)
        self.channel=nn.Linear(8,1); self.decoder=nn.Linear(self.maxnodes,horizon)
    def forward(self,x):
        b,l,d=x.shape
        amp=torch.fft.rfft(x,dim=1).abs().mean(-1)
        amp[:,0]=-torch.inf
        freq=amp.topk(self.S,dim=1).indices
        periods=(l//freq).clamp(min=2)
        lengths=l//periods
        ends=lengths.cumsum(1); starts=ends-lengths
        n=int(ends[:,-1].max())
        idx=torch.arange(n,device=x.device)[None,:].expand(b,-1)
        scale=(idx[:,:,None]>=ends[:,None,:]).sum(-1).clamp(max=self.S-1)
        valid=idx<ends[:,-1:]
        p=periods.gather(1,scale); offset=idx-starts.gather(1,scale)
        left=(offset*p).clamp(0,l); right=((offset+1)*p).clamp(0,l)
        cs=F.pad(x.cumsum(1),(0,0,1,0))
        z=(cs.gather(1,right[:,:,None].expand(-1,-1,d))-cs.gather(1,left[:,:,None].expand(-1,-1,d)))/p[:,:,None]
        h=self.embed(z[:,:,:,None])*valid[:,:,None,None]
        raw=F.relu(self.t1[:n]@self.t2[:,:n])
        scores=raw[None].expand(b,-1,-1).masked_fill(~valid[:,None,:],-torch.inf).softmax(-1)
        mask=torch.zeros((b,n,n),dtype=torch.bool,device=x.device)
        for s in range(self.S):
            candidate=(scale==s)&valid
            ranked=scores.masked_fill(~candidate[:,None,:],-torch.inf).argsort(-1,descending=True)
            ranks=ranked.argsort(-1)
            k=((self.K+periods[:,s]-1)//periods[:,s]).minimum(lengths[:,s])
            mask|=(ranks<k[:,None,None])&candidate[:,None,:]
        trend=(idx[:,:,None]-idx[:,None,:]).abs()<=1
        trend &= scale[:,:,None]==scale[:,None,:]
        mask|=trend&valid[:,None,:]
        a=scores*mask; a=a/a.sum(-1,keepdim=True).clamp_min(1e-12)
        agg=torch.bmm(a,h.reshape(b,n,d*8)).reshape(b,n,d,8)
        h=F.normalize(F.relu(self.time(torch.cat([agg,h],-1))),dim=-1)*valid[:,:,None,None]
        v=F.relu(self.v1@self.v2).softmax(-1)
        k=min(3,d//2)
        pos=v.topk(k,dim=-1).indices; neg=v.topk(k,dim=-1,largest=False).indices
        # Disjoint sets: stable ties may otherwise overlap.
        neg=v.masked_fill(torch.zeros_like(v,dtype=torch.bool).scatter(1,pos,True),torch.inf).topk(k,dim=-1,largest=False).indices
        ap=torch.zeros_like(v).scatter(1,pos,v.gather(1,pos)); ap=ap/ap.sum(-1,keepdim=True)
        inv=1/v.gather(1,neg).clamp_min(1e-12)
        an=torch.zeros_like(v).scatter(1,neg,inv/inv.sum(-1,keepdim=True))
        agg=torch.einsum('ij,btjc->btic',ap-an,h)
        h=F.normalize(F.relu(self.var(torch.cat([agg,h],-1))),dim=-1)*valid[:,:,None,None]
        y=self.channel(h).squeeze(-1)*valid[:,:,None]
        y=F.pad(y.transpose(1,2),(0,self.maxnodes-n))
        return self.decoder(y).transpose(1,2)
