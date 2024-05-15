import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from scipy.interpolate import interp1d

# pierwsza
lambda1_1 = lambda t: 5 * np.sin(t) ** 2
T_1 = 2 * np.pi
lambda2_1 = scipy.integrate.quad(lambda1_1, 0, T_1)[0]
n_1 = np.random.poisson(lambda2_1)
xs_1 = np.linspace(0, T_1, n_1)
m_1 = [scipy.integrate.quad(lambda1_1, 0, x)[0] for x in xs_1]
odwr_dyst_1 = interp1d(m_1, xs_1)
t_1 = np.zeros(2 * n_1)
N_t_1 = np.zeros(2 * n_1)
i_1 = 0
I_1 = 0
while i_1 < 2 * n_1 - 2:
    U = np.random.uniform(0, 1)
    t_1[i_1] = odwr_dyst_1(lambda2_1 * U)
    t_1[i_1 + 1] = odwr_dyst_1(lambda2_1 * U)
    N_t_1[i_1], N_t_1[i_1 + 1] = I_1, I_1
    I_1 += 1
    i_1 += 2
t_1[i_1] = T_1
N_t_1[i_1], N_t_1[i_1 + 1] = I_1, I_1
t_1.sort()
# druga
lambda1_2 = lambda t: np.cos(t) ** 2
T_2 = 2 * np.pi
lambda2_2 = scipy.integrate.quad(lambda1_2, 0, T_2)[0]
n_2 = np.random.poisson(lambda2_2)
xs_2 = np.linspace(0, T_2, n_2)
m_2 = [scipy.integrate.quad(lambda1_2, 0, x)[0] for x in xs_2]
odwr_dyst_2 = interp1d(m_2, xs_2)
t_2 = np.zeros(2 * n_2)
N_t_2 = np.zeros(2 * n_2)
i_2 = 0
I_2 = 0
while i_2 < 2 * n_2 - 2:
    U = np.random.uniform(0, 1)
    t_2[i_2] = odwr_dyst_2(lambda2_2 * U)
    t_2[i_2 + 1] = odwr_dyst_2(lambda2_2 * U)
    N_t_2[i_2], N_t_2[i_2 + 1] = I_2, I_2
    I_2 += 1
    i_2 += 2
t_2[i_2] = T_2
N_t_2[i_2], N_t_2[i_2 + 1] = I_2, I_2
t_2.sort()
# suma
print(N_t_1, N_t_2)
merged_events = sorted(list(t_1) + list(t_2))[1 : len(list(t_1) + list(t_2)) - 1]
nowe_N_t_2 = N_t_2[2:] + len(N_t_1) / 2
merged_times = list(N_t_1) + list(nowe_N_t_2)
xs = np.linspace(0, max(T_1, T_2), 1000)
lambda_12 = lambda t: lambda1_1(t) + lambda1_2(t)
plt.plot(merged_events, merged_times, label="niejednorodny proces (łączenie)")
plt.plot(xs, lambda_12(xs), label="λ(t)")
plt.legend()
plt.show()

N_tt = np.zeros(1000)
i = 0
T = 2 * np.pi
for x in range(1000):
    lambda11 = lambda t: 5 * np.sin(t) ** 2
    xs1 = np.linspace(0, T, 1000)
    lambda21 = scipy.integrate.quad(lambda11, 0, T)[0]
    n1 = np.random.poisson(lambda21)
    lambda12 = lambda t: np.cos(t) ** 2
    lambda22 = scipy.integrate.quad(lambda12, 0, T)[0]
    n2 = np.random.poisson(lambda2_2)
    N_tt[i] = n1 + n2
    i += 1
expected_lambda_t = scipy.integrate.quad(lambda_12, 0, T)[0]
plt.hist(N_tt, bins=int(max(N_tt) - 5), density=True, label="Empiryczny")
xs_3 = np.arange(0, int(max(N_tt)))
poiss = scipy.stats.poisson.pmf(xs_3, mu=expected_lambda_t)
plt.plot(xs_3, poiss, "ro-", label="Teoretyczny (Poisson)")
plt.legend()
plt.show()
