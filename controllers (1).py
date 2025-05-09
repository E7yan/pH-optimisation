import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel,RBF

def expected_improvement(x,gp,y_best,xi):
    mu,sigma=gp.predict(x,return_std=True)
    sigma=np.maximum(sigma,1e-9)
    z=(mu-y_best-xi)/sigma
    return (mu-y_best-xi)*norm.cdf(z)+sigma*norm.pdf(z)

def optimize_flow_rate(pH,target,gp_a,gp_b,xi=0.01):
    X=np.linspace(0.0,4.0,100).reshape(-1,1)
    ei_a=expected_improvement(X,gp_a,target if pH>target else pH,xi)
    ei_b=expected_improvement(X,gp_b,target if pH<target else pH,xi)
    acid=X[ei_a.argmax()][0] if pH>target else 0.0
    base=X[ei_b.argmax()][0] if pH<target else 0.0
    if abs(pH-target)<=0.1:
        acid=0.0
        base=0.0
    return acid,base

class ProportionalController:
    def __init__(self,kp):self.kp=kp
    def __call__(self,current,target):
        d=current-target
        return self.kp*max(d,0.0),self.kp*max(-d,0.0)

class PIDController:
    def __init__(self,kp,ki,kd):
        self.kp=kp;self.ki=ki;self.kd=kd;self.i=0.0;self.prev=0.0
    def __call__(self,current,target):
        e=target-current
        self.i+=e
        d=e-self.prev
        self.prev=e
        out=self.kp*e+self.ki*self.i+self.kd*d
        return max(-out,0.0),max(out,0.0)

class BayesianController:
    def __init__(self):
        k=ConstantKernel(1.0)*RBF(1.0)
        self.gp_a=GaussianProcessRegressor(kernel=k,alpha=1e-10)
        self.gp_b=GaussianProcessRegressor(kernel=k,alpha=1e-10)
    def __call__(self,current,target):
        return optimize_flow_rate(current,target,self.gp_a,self.gp_b)

class NeuralController:
    def __init__(self):
        from sklearn.linear_model import LinearRegression
        self.m=LinearRegression()
        self.m.coef_=np.array([[1.0,-1.0]])
        self.m.intercept_=np.array([0.0,0.0])
    def __call__(self,current,target):
        y=self.m.predict(np.array([[current,target]]))
        return max(y[0,0],0.0),max(y[0,1],0.0)
