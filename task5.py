import numpy as np
from autograd import grad


def gradient_descent(f, x0, epsilon=1e-3, alpha=0.01):
    x = np.array(x0, dtype=float)
    f_x = f(x)
    grad_f = grad(f)

    for _ in range(10000):

        gradient = grad_f(x)
        grad_norm = np.linalg.norm(gradient)
        print(grad_norm)
        # Шаг 2: Проверка условия остановки
        if grad_norm < epsilon:
            break

        # Шаг 3: Попытка сделать шаг
        y = x - alpha * gradient
        f_y = f(y)

        if f_y < f_x:
            x = y
            f_x = f_y
        else:
            # Шаг 4: Уменьшение шага
            alpha /= 2
        print(_)

    return x, f_x


def f(x):
    return 96 * (x[0] ** 2) + 191 * x[0] * x[1] + 96 * (x[1] ** 2) - 91 * x[0] - 12 * x[1] + 86


print(gradient_descent(f, [0, 0]))
