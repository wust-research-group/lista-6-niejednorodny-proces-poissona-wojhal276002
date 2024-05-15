import matplotlib.pyplot as plt
import numpy as np
import scipy

lambda1 = lambda t: 5 * np.sin(t) ** 2
T = 2 * np.pi
xs = np.linspace(0, T, 1000)
lambd_max = max(lambda1(xs))
lambda2 = scipy.integrate.quad(lambda1, 0, T)[0]
N_t = [0]
skoki = [0]
t = 0
I = 0
while t <= T:
    U = np.random.uniform(0, 1)
    t = t - 1 / lambd_max * np.log(U)
    if t > T:
        N_t.append(I)
        skoki.append(T)
    else:
        U_1 = np.random.uniform(0, 1)
        if U_1 <= lambda1(t) / lambd_max:
            N_t.append(I)
            I += 1
            skoki.append(t)
            N_t.append(I)
            skoki.append(t)
plt.plot(skoki, N_t, label="niejednorodny proces (rozrzedzanie)")
plt.plot(xs, lambda1(xs), label="λ(t)")
plt.legend()
plt.show()

N_t = np.zeros(1000)
i = 0
for x in range(1000):
    lambda1 = lambda t: 5 * np.sin(t) ** 2
    xs = np.linspace(0, T, 1000)
    lambd_max = max(lambda1(xs))
    I = 0
    T = 2 * np.pi
    T_i = 0
    while T_i <= T:
        T_i = T_i - 1 / lambd_max * np.log(np.random.uniform(0, 1))
        U_1 = np.random.uniform(0, 1)
        if U_1 <= lambda1(T_i) / lambd_max:
            I += 1
    N_t[i] = I
    i += 1
expected_lambda_t = scipy.integrate.quad(lambda1, 0, T)[0]
plt.hist(N_t, bins=int(max(N_t) - 5), density=True, label="Empiryczny")
xs_1 = np.arange(0, int(max(N_t)))
poiss = scipy.stats.poisson.pmf(xs_1, mu=expected_lambda_t)
plt.plot(xs_1, poiss, "ro-", label="Teoretyczny (Poisson)")
plt.legend()
plt.show()
