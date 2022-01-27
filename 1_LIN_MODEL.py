#Линейная Регрессия
import numpy as np
import matplotlib as plt

np.set_printoptions(precision=3)


# Нормальное уравнение линейной регрессии
def normal_eq(X, y):
    XT = np.transpose(X)
    XTX = np.matmul(XT, X)
    XTX_inv = np.linalg.inv(XTX)
    XTX_invXT = np.matmul(XTX_inv, XT)
    return np.matmul(XTX_invXT, y)


# Стохастический вектор градиент
def stoh_grad_f(X, y, w):
    N = len(y)
    m = len(X[0])
    d = 5
    batch_idxs = random.sample(range(0, N), d)
    return grad_f(X[batch_idxs], y[batch_idxs], w)


# Производная одного элемента суммы функции MSE
def part_dMSEdw(X, y, w, i, k):
    Sum = 0
    for j in range(len(w)):
        Sum += w[j] * X[i][j]

    sub = (y[i] - Sum) * X[i][k]
    return sub


# Значение производной функции MSE(w0, w1, ..., wm) по аргументу wk в точке w = (w0, w1, ..., wm)
def dMSEdw(X, y, w, k):
    N = len(y)
    der = 0

    for i in range(N):
        der += part_dMSEdw(X, y, w, i, k)

    der *= (-2.0 / N)
    return der


# Вектор-градиент функции MSE
def grad_f(X, y, w):
    m = len(w)
    grad = np.zeros(shape=(m, 1))
    for i in range(m):
        grad[i] = dMSEdw(X, y, w, i)

    return grad


# Условие для выхода из цикла
def condition(P, P_pred, eps):
    fl = 1
    for i in range(len(P)):
        if abs(P[i] - P_pred[i]) >= eps:
            fl *= 0
    return fl


# Метод градиентного спуска
def gradient_desc(X, y, w, k, func):
    eps = 0.00001
    a = 1 / 1000

    P = np.copy(w)
    P_pred = np.copy(P)
    P -= a * func(X, y, w)

    while k > 0:
        if condition(P, P_pred, eps):
            break
        P_pred = np.copy(P)
        P -= a * func(X, y, w)
        k -= 1

    return P


# Средне-квадратическая ошибка
def MSE(y_true, y):
    Q = 0
    for i in range(len(y)):
        Q += (y_true[i] - y[i]) * (y_true[i] - y[i])
    return Q / len(y)


def main():
    N = 10
    D = 2
    # Матрица объекты-признаки
    X = np.random.uniform(0, 1, (N, D))

    # Добавление единичного признака для w_0
    D += 1
    x_ones = np.ones((N, 1))
    X = np.hstack((x_ones, X))

    # Вектор коэффициентов
    w_true = np.random.uniform(-5, 10, (D, 1))

    # Целевая переменная
    y = X @ w_true

    # Задача: получить w_true, зная только X и y
    # используя нормальное уравение линейной регрессии
    # Модель мы строим сами

    w = normal_eq(X, y)
    print(w)
    print('===================')

    # Шум
    P = w + np.random.normal(0, len(w))

    # Нахождение w с помощью метода градиентного спуска
    # Наилучшее приближение будет при 20 < k < 40
    print(gradient_desc(X, y, P, 25, grad_f))

    max_iter = 200

    k = np.zeros(shape=(max_iter, 1))
    Qk = np.zeros(shape=(max_iter, 1))

    t = 0
    while t < max_iter:
        k[t] = t
        Qk[t] = MSE(w_true, gradient_desc(X, y, P, t, grad_f))
        t += 1

    # Метод стохастического градиентного спуска

    print('===================')
    print(gradient_desc(X, y, P, 25, stoh_grad_f))

    k_stoh = np.zeros(shape=(max_iter, 1))
    Qk_stoh = np.zeros(shape=(max_iter, 1))

    t = 0
    while t < max_iter:
        k_stoh[t] = t
        Qk_stoh[t] = MSE(w_true, gradient_desc(X, y, P, t, stoh_grad_f))
        t += 1

    plt.plot(k_stoh, Qk_stoh, c='red')
    plt.plot(k, Qk, c='blue')
    plt.show()

if __name__ == '__main__':
    main()