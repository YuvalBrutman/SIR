import numpy as np
import matplotlib.pyplot as plt

d_t = 60
S, I, R = 0, 1, 2
starting_N = 7.8e9
death_rate_per_minute_ratio = 231/7.8e9
birth_rate_per_minute_ratio = 267/7.8e9
sick_death_rate_ratio = 0.01
def f(SIR, beta, N, gama):
    return np.array([(-beta / N) * SIR[I] * SIR[S],
                     -gama * SIR[I] + (beta / N) * SIR[I] * SIR[S],
                     gama * SIR[I]])


def run_simulation(d_t, SIR, N, death_rate_per_minute_ratio, birth_rate_per_minute_ratio, sick_death_rate_ratio, gama=(1 / 4)/(24*3600), beta=(1 / 2)/(24*3600)):
    S_array, I_array, R_array, time = [], [], [], []
    t = 0
    step_count = 0
    cur_SIR = SIR
    while cur_SIR[I] >= 1:
        if step_count%(60//d_t) == 0:
            N += (-int(death_rate_per_minute_ratio*N)+int(birth_rate_per_minute_ratio*N)-int(sick_death_rate_ratio*SIR[I]))
            SIR[S] += int(-death_rate_per_minute_ratio*N/3+birth_rate_per_minute_ratio*N)
            SIR[I] += (-int(death_rate_per_minute_ratio*N/3)-int(sick_death_rate_ratio*SIR[I]))
            SIR[R] += (-int(death_rate_per_minute_ratio*N/3))
        k_1 = d_t * f(cur_SIR, beta, N, gama)
        k_2 = d_t * f(cur_SIR + k_1 / 2, beta, N, gama)
        k_3 = d_t * f(cur_SIR + k_2 / 2, beta, N, gama)
        k_4 = d_t * f(cur_SIR + k_3, beta, N, gama)
        next_SIR = cur_SIR + (1 / 6) * (k_1 + 2 * k_2 + 2 * k_3 + k_4)
        cur_SIR = next_SIR
        S_array.append(cur_SIR[S])
        I_array.append(cur_SIR[I])
        R_array.append(cur_SIR[R])
        time.append(t)
        t += d_t
        step_count += 1
    plt.plot(np.array(time), np.array(S_array), label="S")
    plt.plot(np.array(time), np.array(I_array), label="I")
    plt.plot(np.array(time), np.array(R_array), label="R")
    plt.legend()
    plt.show()


run_simulation(d_t, np.array([starting_N - 1000, 1000, 0]), starting_N, death_rate_per_minute_ratio, birth_rate_per_minute_ratio,sick_death_rate_ratio)
