import numpy as np
import matplotlib as plt
from sklearn.model_selection import train_test_split

np.set_printoptions(precision=3)

# Средне квадратическая ошибка
def MSE(y_true, y):
    Q = 0
    for i in range(len(y)):
        Q += (y_true[i] - y[i]) * (y_true[i] - y[i])
    return Q / len(y)


def middle(arr):
    arr_sum = 0
    for i in range(len(arr)):
        arr_sum += arr[i]
    return arr_sum / len(arr)


def denumenatorR2(y):
    sumDenR2 = 0
    y_mid = middle(y)
    for i in range(len(y)):
        sumDenR2 += (y[i] - y_mid) * (y[i] - y_mid)
    return sumDenR2


def R2(y_true, y):
    num = MSE(y_true, y) * len(y)
    den = denumenatorR2(y_true)
    return 1 - num / den


def graph(x, y, x_graph, y_graph_model, y_graph_sin):
    plt.scatter(x, y, c='blue')
    plt.plot(x_graph, y_graph_model, c='red')
    plt.plot(x_graph, y_graph_sin, c='green')
    plt.show()


def main():
    n = 10
    d = 3

    x = np.linspace(0, 7, n)
    y_true = 2 * np.sin(x)

    y = y_true + np.random.normal(0, 1, len(y_true))
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.90, random_state=42)

    print(x)
    print(y)

    model = np.poly1d(np.polyfit(x_train, y_train, d))

    y_pred_train = model(x_train)
    y_pred_test = model(x_test)

    print(x_train, y_pred_train)
    print(x_test, y_pred_test)

    y_train_true = 2 * np.sin(x_train)
    y_test_true = 2 * np.sin(x_test)

    Q_train = MSE(y_train_true, y_pred_train)
    Q_test = MSE(y_test_true, y_pred_test)

    print('Q_train = ' + str(Q_train))
    print('Q_test = ' + str(Q_test))

if __name__ == '__main__':
    main()
