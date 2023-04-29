import numpy as np
import matplotlib.pyplot as plt

# simulation constants
DT = 60  # timestep of sim in seconds
starting_N = 7.8e9
death_rate_per_minute_ratio = 231 / 7.8e9
birth_rate_per_minute_ratio = 267 / 7.8e9
sick_death_rate_ratio = 0.01

GAMMA = 1 / 4 / 24 / 3600  # recovery rate (1 / s)
BETA = 1 / 2 / 24 / 3600  # infection rate (1 / s)

# miscellaneous
S, I, R = 0, 1, 2  # index for vector of Susceptible Infected and Recovered

"""
derivative of SIR calculator for rk4 progression
"""


def f(SIR, N):
    return np.array([(-BETA / N) * SIR[I] * SIR[S],
                     -GAMMA * SIR[I] + (BETA / N) * SIR[I] * SIR[S],
                     GAMMA * SIR[I]])


def rk4(SIR, N):
    k_1 = DT * f(SIR, N)
    k_2 = DT * f(SIR + k_1 / 2, N)
    k_3 = DT * f(SIR + k_2 / 2, N)
    k_4 = DT * f(SIR + k_3, N)

    return SIR + (1 / 6) * (k_1 + 2 * k_2 + 2 * k_3 + k_4)


def run_simulation():
    N = starting_N
    SIR = np.array([N - 1000, 1000, 0])
    S_array, I_array, R_array, time = [], [], [], []
    t = 0
    step_count = 0
    cur_SIR = SIR

    while cur_SIR[I] >= 1:
        if step_count % (60 // DT) == 0:
            N += (-int(death_rate_per_minute_ratio * N) + int(birth_rate_per_minute_ratio * N) - int(sick_death_rate_ratio * SIR[I]))
            SIR[S] += int(-death_rate_per_minute_ratio * N / 3 + birth_rate_per_minute_ratio * N)
            SIR[I] += (-int(death_rate_per_minute_ratio * N / 3) - int(sick_death_rate_ratio * SIR[I]))
            SIR[R] += (-int(death_rate_per_minute_ratio * N / 3))

        cur_SIR = rk4(cur_SIR, N)

        S_array.append(cur_SIR[S])
        I_array.append(cur_SIR[I])
        R_array.append(cur_SIR[R])
        time.append(t)

        t += DT
        step_count += 1

    plt.plot(np.array(time), np.array(S_array), label="S")
    plt.plot(np.array(time), np.array(I_array), label="I")
    plt.plot(np.array(time), np.array(R_array), label="R")
    plt.legend()
    plt.show()


run_simulation()
