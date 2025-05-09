import time,matplotlib.pyplot as plt,numpy as np
from calibration_core import create_calibration_curve
from hardware import initialize_spectrometer,initialize_pumps,adjust_pumps
from controllers import ProportionalController,PIDController,BayesianController,NeuralController

def choose(mode):
    if mode=='P':return ProportionalController(0.5)
    if mode=='PID':return PIDController(0.6,0.05,0.05)
    if mode=='BAYES':return BayesianController()
    if mode=='NN':return NeuralController()
    raise ValueError

def main():
    target=7.0
    w=[415,445,480,515,555,590,630,680]
    p=[4,5,6,7,8,9,10,11]
    f=create_calibration_curve(w,p)
    spectro=initialize_spectrometer()
    pumps=initialize_pumps()
    machine=pumps['a']
    mode=input("Select controller P PID BAYES NN: ").strip().upper()
    ctrl=choose(mode)
    plt.ion()
    t0=time.time()
    ts=[]
    ps=[]
    fig,ax=plt.subplots()
    while True:
        _,peak=spectro.read_peak()
        current=float(f(peak))
        acid,base=ctrl(current,target)
        adjust_pumps(machine,acid,base)
        ts.append(time.time()-t0)
        ps.append(current)
        ax.cla()
        ax.plot(ts,ps)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("pH")
        ax.set_title("System pH")
        plt.pause(0.1)

if __name__=='__main__':
    main()
