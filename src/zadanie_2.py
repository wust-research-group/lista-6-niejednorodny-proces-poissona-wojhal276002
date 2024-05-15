import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.interpolate import interp1d

lambda1 = lambda t: 5 * np.sin(t) ** 2
T = 2 * np.pi
lambda2 = scipy.integrate.quad(lambda1, 0, T)[0]
n = np.random.poisson(lambda2)
xs = np.linspace(0, T, n)
m = [scipy.integrate.quad(lambda1, 0, x)[0] for x in xs]
odwr_dyst = interp1d(m, xs)
t = np.zeros(2 * n)
N_t = np.zeros(2 * n)
i = 0
I = 0
while i < 2 * n - 2:
    U = np.random.uniform(0, 1)
    t[i] = odwr_dyst(lambda2 * U)
    t[i + 1] = odwr_dyst(lambda2 * U)
    N_t[i], N_t[i + 1] = I, I
    I += 1
    i += 2
t[i] = T
N_t[i], N_t[i + 1] = I, I
t.sort()
xs_1 = np.linspace(0, T, 1000)
plt.plot(t, N_t, label="niejednorodny proces (odwr dyst)")
plt.plot(xs_1, lambda1(xs_1), label="λ(t)")
plt.legend()
plt.show()

N_tt = np.zeros(1000)
i_1 = 0
for x in range(1000):
    lambda1 = lambda t: 5 * np.sin(t) ** 2
    xs_2 = np.linspace(0, T, 1000)
    lambda2 = scipy.integrate.quad(lambda1, 0, T)[0]
    n = np.random.poisson(lambda2)
    N_tt[i_1] = n
    i_1 += 1
expected_lambda_t = scipy.integrate.quad(lambda1, 0, T)[0]
plt.hist(N_tt, bins=int(max(N_tt) - 5), density=True, label="Empiryczny")
xs_3 = np.arange(0, int(max(N_tt)))
poiss = scipy.stats.poisson.pmf(xs_3, mu=expected_lambda_t)
plt.plot(xs_3, poiss, "ro-", label="Teoretyczny (Poisson)")
plt.legend()
plt.show()
