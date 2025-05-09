import numpy as np

WAVELENGTHS=[415,445,480,515,555,590,630,680]

class Spectrometer:
    def __init__(self):pass
    def read_peak(self):
        try:
            from adafruit import readAll
            intensities=readAll()
        except Exception:
            intensities=np.random.randint(10,50,len(WAVELENGTHS))
        peak=int(WAVELENGTHS[int(np.argmax(intensities))])
        return dict(zip(WAVELENGTHS,intensities)),peak

def initialize_spectrometer(integration_time=2000,gain=1.0):
    return Spectrometer()

class MockPump:
    def __init__(self):self.state={1:0.0,2:0.0}
    def stop(self,n):self.state[n]=0.0
    def changePump(self,n):self.p=n
    def rate(self,r):self.state[self.p]=r
    def run(self,n):self.state[n]=self.state.get(n,0.0)

def initialize_pumps():
    return {'a':MockPump()}

def adjust_pumps(machine,acid,base):
    machine.stop(1)
    machine.stop(2)
    if acid>0.0:
        machine.changePump(1)
        machine.rate(acid)
        machine.run(1)
    if base>0.0:
        machine.changePump(2)
        machine.rate(base)
        machine.run(2)
