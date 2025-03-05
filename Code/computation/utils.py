import numpy as np


# pythran export auto_correlation(uint64[], uint64[], bool) -> (float[], float[])
# pythran export auto_correlation(float[], float[], bool) -> (float[], float[])
def auto_correlation(x, y, fast):
    n = len(x)
    # if n % 2 == 0:
    #     x = x[:-1]
    #     y = y[:-1]
    dis = x[-1] - x[0]
    # tau = np.linspace(-dis / 2, dis / 2, n)
    tau = np.linspace(0, dis, n)

    if not fast:
        result = np.array([np.mean(y * np.roll(y, i)) for i in range(n)])
        # result = np.roll(result / result[0], n // 2)

        return tau, result
    else:
        corr = np.fft.ifft(np.fft.ifft(y) * np.fft.fft(y)).real
        # return tau, np.roll(corr / corr[0], n//2)
        return tau, corr


# pythran export get_total_dissipation_rate(uint8[] list, uint64) -> uint[]
def get_total_dissipation_rate(dissipation_rates, max_time):
    # n = np.ceil(len(dissipation_rates) / perturbation_rate).astype(np.uint64)

    j = np.zeros(max_time, dtype=np.uint64)
    starting_points = np.random.randint(0, max_time, size=len(dissipation_rates))

    for ind, dis in zip(starting_points, dissipation_rates):
        # for k in range(ind, n):
        #     j[k % n] += dis[k - i]
        for k, d in enumerate(dis):
            j[(k + ind) % max_time] += d

    return j
