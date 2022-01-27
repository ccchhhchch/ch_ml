import numpy as np
import matplotlib as plt

#Град.спуск
np.set_printoptions(precision=3)


# Функция Химмельблау
def f(x, y):
    return (x * x + y - 11) * (x * x + y - 11) + (x + y * y - 7) * (x + y * y - 7)


def dfdx(x, y):
    return 4 * x * x * x + 4 * x * y - 42 * x + 2 * y * y - 14


def dfdy(x, y):
    return -26 * y + 2 * x * x - 22 + 4 * y * y * y + 4 * x * y


def grad_f(P):
    x = dfdx(P[0], P[1])
    y = dfdy(P[0], P[1])
    return np.array([x, y])


def main():
    # P - исходная точка
    P = np.array([6.0, 5.0])
    P_pred = np.copy(P)

    # a - скорость обучения
    a = 1 / 1000
    P -= a * grad_f(P)

    # Эпсилон окрестность
    eps = 0.000001

    # Максиимальное число итераций
    k = 1000

    # Метод градиентного спуска
    while k > 0:
        if abs(P[0] - P_pred[0]) < eps:
            break
        P_pred = np.copy(P)
        P -= a * grad_f(P)
        k -= 1

    print(P)


if __name__ == '__main__':
    main()
