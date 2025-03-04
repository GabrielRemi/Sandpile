import numpy as np


# pythran export auto_correlation(int[], int[]) -> (float[], float[])
# pythran export auto_correlation(uint64[], uint64[]) -> (float[], float[])
# pythran export auto_correlation(float[], float[]) -> (float[], float[])
def auto_correlation(x, y):
    n = len(x)
    if n % 2 == 0:
        x = x[:-1]
        y = y[:-1]
    result = [np.mean(y * np.roll(y, i)) for i in range((n))]
    result = np.roll(result, int(n / 2))

    dis = x[-1] - x[0]
    tau = np.linspace(-dis / 2, dis / 2, n)

    return tau, result


# def get_total_dissipation_rate(dissipation_rates, time_steps):
#     diff = np.diff(time_steps)
#     # return dissipation_rates[0]
#     total_dissipation_rate = []
#     total_dissipation_rate.extend([0] * time_steps[0])
#     total_dissipation_rate.extend(dissipation_rates[0])
#
#
#     for d, r in zip(diff, dissipation_rates[1:]):
#         total_dissipation_rate.extend([0] * int(d))
#         total_dissipation_rate.extend(r)
#
#     x = np.array(range(len(total_dissipation_rate)), dtype=np.uint64)
#     y = np.array(total_dissipation_rate, dtype=np.uint64)
#
#     return x, y

# pythran export get_total_dissipation_rate(uint8[] list, float) -> int[], int
def get_total_dissipation_rate(dissipation_rates, perturbation_rate):
    n = np.ceil(len(dissipation_rates) / perturbation_rate).astype(np.uint64)

    j = np.zeros(n, dtype=np.uint64)

    for dis in dissipation_rates:
        i = np.random.randint(0, n - len(dis))
        for k in range(i, n):
            j[k] += dis[k - i]

    return j, int(n)
