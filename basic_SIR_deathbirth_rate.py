import numpy as np
import matplotlib.pyplot as plt

# simulation constants
DT = 60  # timestep of sim in seconds
starting_N = 7.8e9
death_rate_per_second = 231 / 60
birth_rate_per_second = 267 / 60
sick_death_rate_ratio = 0.01
vaccine_effect_time = 10 * 24 * 3600  # s
vaccination_rate = 100000 / 24 / 3600  # 1 / s

GAMMA = 1 / 8 / 24 / 3600  # recovery rate (1 / s)
BETA = 1 / 2 / 24 / 3600  # infection rate (1 / s)

# miscellaneous
S, I, R, V, M = range(5)  # index for vector of Susceptible Infected and Recovered

"""
derivative of SIR calculator for rk4 progression
"""


def f(SIR, N, vaccinate, V_array):
    if vaccinate and len(V_array) > vaccine_effect_time // DT:
        vacc_to_be_immune = (V_array[-vaccine_effect_time // DT] - V_array[-vaccine_effect_time // DT - 1]) / DT
        return np.array([(-BETA / N) * SIR[I] * SIR[S] + birth_rate_per_second - death_rate_per_second / 3 - vaccination_rate,
                         -GAMMA * SIR[I] + (BETA / N) * SIR[I] * SIR[S] - death_rate_per_second / 3,
                         GAMMA * SIR[I] - death_rate_per_second / 3,
                         vaccination_rate - vacc_to_be_immune,
                         vacc_to_be_immune])
    else:
        return np.array([(-BETA / N) * SIR[I] * SIR[S] + birth_rate_per_second - death_rate_per_second / 3,
                         -GAMMA * SIR[I] + (BETA / N) * SIR[I] * SIR[S] - death_rate_per_second / 3,
                         GAMMA * SIR[I] - death_rate_per_second / 3,
                         0,
                         0])


def rk4(SIR, N, vaccinate, V_array):
    k_1 = DT * f(SIR, N, vaccinate, V_array)
    k_2 = DT * f(SIR + k_1 / 2, N, vaccinate, V_array)
    k_3 = DT * f(SIR + k_2 / 2, N, vaccinate, V_array)
    k_4 = DT * f(SIR + k_3, N, vaccinate, V_array)

    return SIR + (1 / 6) * (k_1 + 2 * k_2 + 2 * k_3 + k_4)


def run_simulation():
    vaccinate = False
    N = starting_N
    SIRVM = np.array([N - 1000000, 1000000, 0, 0, 0])
    S_array, I_array, R_array, V_array, M_array, time = [], [], [], [], [], []
    t = 0
    step_count = 0
    cur_SIRVM = SIRVM

    while cur_SIRVM[I] >= 1:
        if cur_SIRVM[I] > 1000:
            vaccinate = True

        cur_SIRVM = rk4(cur_SIRVM, N, vaccinate, V_array)
        N += (birth_rate_per_second - death_rate_per_second) * DT

        S_array.append(cur_SIRVM[S])
        I_array.append(cur_SIRVM[I])
        R_array.append(cur_SIRVM[R])
        V_array.append(cur_SIRVM[V])
        M_array.append(cur_SIRVM[M])
        time.append(t)

        t += DT
        step_count += 1

    plt.plot(np.array(time), np.array(S_array), label="S")
    plt.plot(np.array(time), np.array(I_array), label="I")
    plt.plot(np.array(time), np.array(R_array), label="R")
    plt.legend()
    plt.show()


run_simulation()
