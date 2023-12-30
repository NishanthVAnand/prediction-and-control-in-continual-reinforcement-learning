import numpy as np
import scipy.signal as signal

def compute_return(rew_list, gamma):
	r = rew_list[::-1] # can be numpy array too
	a = [1, -gamma]
	b = [1]
	y = signal.lfilter(b, a, x=r)
	return y[::-1]