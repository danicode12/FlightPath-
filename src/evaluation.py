import numpy as np
def rmse(a,b): return float(np.sqrt(np.mean((a-b)**2)))
def ade(p,t):  return float(np.mean(np.linalg.norm(p-t,axis=1)))
def fde(p,t):  return float(np.linalg.norm(p[-1]-t[-1]))
