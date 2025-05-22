import numpy as np
from scipy.optimize import minimize_scalar





def one_dimensional_minimization(x, direction):
    result = minimize_scalar(lambda alpha: f(x + alpha * direction))
    return result.x, result.fun


def cyclic_coordinate_descent(x, e=1e-5):
    n = len(x)
    x = np.array(x)

    # Базисные векторы (матрица с единицами на главной диагонали)
    base_vectors = np.eye(n)

    i = 0
    while True:
        if i < n:
            new_alpha, f_new = one_dimensional_minimization(x, base_vectors[i])
            x_ = x + new_alpha * base_vectors[i]

        if i < n:
            x = x_
            i += 1
        else:
            # Проверка условия остановки
            if np.abs(f(x_) - f(x)) < e:
                x_min = x_
                f_min = f(x_)
                break
            else:
                x = x_
                i = 0


    return x_min, f_min, i



def coordinate_search(x, delta):
    n = len(x)
    base_vectors = np.eye(n)
    x_ = x.copy()

    for j in range(n):
        # Попытка сдлетаь шаг в отрицательном направлении от базиса
        y = x_ - delta[j] * base_vectors[j]

        # Если на прошлом шаге функция меньше, пробуем сделать шаг в положительном направлении
        if f(x_) < f(y):
            y = x_ + delta[j] * base_vectors[j]

        # Если был сделан шаг в сторону минимума, то присваиваем
        if f(x_) > f(y):
            x_ = y

    return x_

def hooke_jeeves_vethod(x, delta=0.5, gamma=2.0, e=1e-5):
    n = len(x)
    delta = np.full(n, delta)
    x = np.array(x, dtype=float)

    j = 0
    for i in range(10000):
        j += 1
        # Проводим покоординатный поиск
        x_ = coordinate_search(x, delta)

        if np.array_equal(x_, x):

            # Если x и x_ равны, но проверяем норму приращений на окончание поиска
            if np.linalg.norm(delta) < e:
                break
            else:
                # Если поиск не окончен, то уменьшаем приращения
                delta = delta / gamma
                continue
        # Если x и x_ не равны, то осуществляем перемещение в направлении убывания
        x = x + 2 * (x_ - x)

        # Выбираем лучшую точку между x_, x
        if f(x_) < f(x):
            x = x_

    return x, f(x), j

def random_search(x, e=1e-5, step=0.5, gamma=1.5, max_tries=100):
    n = len(x)
    x = np.array(x, dtype=float)
    x_min = np.array(x, dtype=float)
    # Счетчик числа неудачных попыток
    j = 0

    for i in range(10000):
        # Получаем случайный вектор
        Xi = np.random.uniform(low=-1.0, high=1.0, size=n)

        # Пробная точка
        y = x + step * (Xi / np.linalg.norm(Xi))
        # Если f(y) < f(x), то идем в правильном направлении к минимуму
        if f(y) < f(x):
            x = y
            j = 0
            continue

        # Если нет, то проверяем на окончание поиска и уменьшаем шаг
        j += 1
        if j > max_tries:
            if step < e:
                x_min = x
                break
            step = step / gamma
            j = 0

    return x_min, f(x_min), i

def f(x):
    return x[0] ** 3 + 2 * x[1] ** 2 - 3 * x[0] + x[1] + 7

result1 = cyclic_coordinate_descent([0, 0])
print(f"Метод циклического координатного спуска: x = {result1[0]}, f(x) = {result1[1]}, количество итераций = {result1[2]}")

result2 = hooke_jeeves_vethod([0, 0])
print(f"Метод Хука-Дживса: x = {result2[0]}, f(x) = {result2[1]}, количество итераций = {result2[2]}")

result3 = random_search([0, 0])
print(f"Метод случайного поиска: x = {result3[0]}, f(x) = {result3[1]}, количество итераций = {result3[2]}")
