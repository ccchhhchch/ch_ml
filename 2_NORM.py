import numpy as np
import matplotlib.pyplot as plt
import random

# нахождение коэффициентов нормального распределения
def predict(N, x, y):

    sigm = 1
    m = 0
    d = 1

    xi1 = N / (2.0 * sigm) - 1 / (2.0 * d * d)

    sum = 0
    for i in range(N):
        sum += x[i] * x[i]
    xi2 = 1.0 / (2.0 * sigm) * sum - 1.0 / (2.0 * d * d)

    sum = 0
    for i in range(N):
        sum += x[i]
    xi3 = 1.0 / sigm * sum

    sum = 0
    for i in range(N):
        sum += y[i]
    xi4 = 1.0 / sigm * sum + m / (d * d)

    sum = 0
    for i in range(N):
        sum += x[i] * y[i]
    xi5 = 1.0 / sigm * sum + m / (d * d)

    rho = xi3 / (2 * np.sqrt(xi1 * xi2))
    sigma1 = -2.0 * xi1 * (1 - rho * rho)
    sigma2 = -2.0 * xi2 * (1 - rho * rho)

    a = -1.0 / (2 * (1 - rho * rho)) * sigma1 * sigma1
    b = -1.0 / (2 * (1 - rho * rho)) * sigma2 * sigma2
    c = 1.0 / (2 * (1 - rho * rho)) * 2.0 * rho * sigma1 * sigma2

    M = np.zeros(shape=(2, 2))
    M[0][0] = -2.0 * a
    M[0][1] = -c
    M[1][0] = -c
    M[1][1] = -2.0 * b
    mu = np.linalg.solve(M, [xi4, xi5])

    mean = [mu[0], mu[1]]
    cov = [[1.0 / (sigma1 * sigma1), rho], [rho, 1.0 / (sigma2 * sigma2)]]
    x, y = np.random.multivariate_normal(mean, cov, N).T

    i = random.randint(0, N - 1)
    return [x[i], y[i]]


def main():
    N = 10
    n = 10

    x = []
    y_true = []
    M = np.zeros(shape=(n, 2))

    for i in range(n):
        N *= 2

        x = np.linspace(0, 7, N)
        y_true = 1.0 + 5.0 * x
        y = y_true + np.random.normal(0, 1, N)

        res = predict(N, x, y)
        M[i][0] = res[0]
        M[i][1] = res[1]

    color = 0.0
    for i in range(n):
        y_pred = M[i][0] + M[i][1] * x
        plt.plot(x, y_pred, c=str(color))
        color += 1.0 / n

    plt.scatter(x, y, c='pink')
    plt.show()


if __name__ == '__main__':
    main()
