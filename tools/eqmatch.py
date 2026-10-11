# eqmatch.py low.wav lowRpm ref.wav refRpm [out.wav] : print 1/3-oct spectral diff at handover; if out given, write EQ-matched low loop
import numpy as np, soundfile as sf, sys
def spec(x,sr,rate):
    n=int(len(x)/rate); y=np.interp(np.arange(n)*rate,np.arange(len(x)),x)  # playbackRate resample
    w=4096; S=np.zeros(w//2+1); k=0
    for i in range(0,len(y)-w,w//4): S+=np.abs(np.fft.rfft(y[i:i+w]*np.hanning(w)))**2; k+=1
    return S/k
def bands(S,sr):
    f=np.fft.rfftfreq(4096,1/sr); c=60*2**(np.arange(0,27)/3); out=[]
    for fc in c: m=(f>=fc/2**(1/6))&(f<fc*2**(1/6)); out.append(10*np.log10(S[m].mean()+1e-12))
    return c,np.array(out)
lo,lr,ref,rr=sys.argv[1],float(sys.argv[2]),sys.argv[3],float(sys.argv[4])
a,sr=sf.read(lo); b,_=sf.read(ref); m=np.sqrt(lr*rr)
c,A=bands(spec(a,sr,m/lr),sr); _,B=bands(spec(b,sr,m/rr),sr)
d=(B-B.mean())-(A-A.mean()); print("handover %d rpm: mean |diff| %.1f dB; max %.1f"%(m,np.abs(d).mean(),np.abs(d).max()))
if len(sys.argv)>5:
    d=np.clip(np.convolve(np.pad(d,1,mode='edge'),[.25,.5,.25],'valid'),-9,9)
    # correction defined in played-back frequency; in the file domain frequencies are /(m/lr)
    F=np.fft.rfftfreq(len(a),1/sr)*(m/lr); g=10**(np.interp(np.log(np.maximum(F,1)),np.log(c),d)/20)
    y=np.fft.irfft(np.fft.rfft(a)*g,len(a)); y/=np.abs(y).max(); sf.write(sys.argv[5],y,sr)
    A2=bands(spec(y,sr,m/lr),sr)[1]; d2=(B-B.mean())-(A2-A2.mean()); print("  after EQ: mean |diff| %.1f dB"%np.abs(d2).mean())
