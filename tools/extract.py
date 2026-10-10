# usage: extract.py name src fmin fmax mult cyl t0 t1 rpm1,rpm2,...  -> writes /workspace/bank/<name>_<rpm>.wav, prints json
import numpy as np, soundfile as sf, sys, json, os
from scipy.signal import medfilt
sys.path.insert(0,os.path.dirname(__file__)); from track import load, track
name,src=sys.argv[1],sys.argv[2]; fmin,fmax,mult,cyl,t0,t1=map(float,sys.argv[3:9]); rpms=[int(r) for r in sys.argv[9].split(",")]
OUT=float(os.environ.get("LEN","0.9"))
x,sr=load(src); t,f0,s=track(x,sr,fmin,fmax)
f0=medfilt(f0,9); rpm=f0*mult*120/cyl
os.makedirs("/workspace/bank",exist_ok=True); res=[]
win=(t>=t0)&(t<=t1)
for R in rpms:
    # score frames: near target, strong, smooth (small local derivative variance)
    d=np.abs(np.log(rpm/R)); good=win&(d<0.04)
    if not good.any(): print("no frames for",R,file=sys.stderr); continue
    sc=np.where(good,s-30*d,-1e9); j=int(sc.argmax()); tc=t[j]
    # warp: build f0 at sample rate around tc
    ts=np.arange(len(x))/sr; F=np.interp(ts,t,rpm)
    pos=tc*sr; need=int(OUT*sr); out=np.zeros(need,np.float32)
    # go backwards half, forwards half for centered
    p=pos; seq=[]
    for _ in range(need//2):
        p-=R/F[int(min(max(p,0),len(x)-1))]
    p=max(p,0)
    for i in range(need):
        k=int(p); 
        if k+1>=len(x): break
        out[i]=x[k]*(1-(p-k))+x[k+1]*(p-k); p+=R/F[k]
    out/=np.max(np.abs(out))+1e-9
    fn=f"/workspace/bank/{name}_{R}.wav"; sf.write(fn,out,sr); res.append({"rpm":R,"t":round(float(tc),2),"file":fn})
print(json.dumps(res))
