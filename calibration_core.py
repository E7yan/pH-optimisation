import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

def create_calibration_curve(w,p):
    f=interp1d(w,p,kind='linear',fill_value='extrapolate')
    plt.figure(figsize=(6,4))
    x=np.linspace(min(w),max(w),400)
    plt.plot(x,f(x))
    plt.scatter(w,p)
    plt.xlabel("Wavelength")
    plt.ylabel("pH")
    plt.title("Calibration")
    plt.tight_layout()
    plt.show()
    return f
