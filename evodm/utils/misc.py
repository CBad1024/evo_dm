import numpy as np
import matplotlib.pyplot as plt




def f(x: np.ndarray, N : int) -> np.ndarray:
    return (1- x**2)/(1 - x**(2*N))





x = np.arange(-2, 2, 0.01)
plt.plot(x, f(x, 1), color='r', label='N=1')
plt.plot(x, f(x, 2), color='g', label='N=2')
plt.plot(x, f(x, 3), color='b', label='N=3')
plt.plot(x, f(x, 4), color='c', label='N=4')
plt.plot(x, f(x, 5), color='m', label='N=5')

plt.title("Fixation Probability for Wright-Fisher Model (x = f_i/f_j)")
plt.legend()
plt.tight_layout()
plt.show()