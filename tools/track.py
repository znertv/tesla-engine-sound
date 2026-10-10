import numpy as np, soundfile as sf, subprocess, sys, json
from scipy.signal import stft, medfilt
def load(path, sr=44100):
    raw=subprocess.run(["ffmpeg","-loglevel","error","-i",path,"-ac","1","-ar",str(sr),"-f","f32le","-"],capture_output=True).stdout
    return np.frombuffer(raw,np.float32).copy(), sr
def track(x,sr,fmin,fmax,hop=256,n=4096):
    f,t,Z=stft(x,sr,nperseg=n,noverlap=n-hop); M=np.log1p(np.abs(Z)*50)
    cands=np.geomspace(fmin,fmax,300); S=np.zeros((len(cands),M.shape[1]))
    df=f[1]
    for i,c in enumerate(cands):
        for h in range(1,7):
            k=c*h/df
            if k>=len(f)-1: break
            S[i]+=np.interp(k,np.arange(len(f)),np.arange(len(f)))*0+M[int(round(k))]*(1/h**0.3)
        # penalize subharmonic: subtract energy at half-harmonics
        for h in range(1,6):
            k=int(round(c*(h-.5)/df));S[i]-=0.6*M[k]/h**0.3
    # DP continuity
    T=S.shape[1]; L=len(cands); acc=S[:,0].copy(); bp=np.zeros((L,T),int); jump=6
    for j in range(1,T):
        best=np.full(L,-1e9); arg=np.zeros(L,int)
        for d in range(-jump,jump+1):
            sh=np.roll(acc,d)-0.3*abs(d)
            if d>0: sh[:d]=-1e9
            if d<0: sh[d:]=-1e9
            m=sh>best; best[m]=sh[m]; arg[m]=(np.arange(L)-d)[m]
        acc=best+S[:,j]; bp[:,j]=arg
    p=np.zeros(T,int); p[-1]=acc.argmax()
    for j in range(T-1,0,-1): p[j-1]=bp[p[j],j]
    return t, cands[p], S.max(0)
if __name__=="__main__":
    path,fmin,fmax=sys.argv[1],float(sys.argv[2]),float(sys.argv[3])
    x,sr=load(path); t,f0,s=track(x,sr,fmin,fmax)
    for i in range(0,len(t),17): print(f"{t[i]:6.2f} {f0[i]:7.1f} {s[i]:6.1f}")
