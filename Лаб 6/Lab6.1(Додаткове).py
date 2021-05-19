import random
import sklearn.linear_model as lm
from pyDOE2 import ccdesign
import numpy as np
import scipy.stats
from functools import partial
import sys
import time


average_time = []
for i in range(10):

    x_range = [(-7, 10), (-4, 6), (-5, 3)]

    def regression(x, b):
        return sum([x[i] * b[i] for i in range(len(x))])


    def s_kv(y, y_aver, n, m):
        res = []
        for i in range(n):
            s = sum([(y_aver[i] - y[i][j]) ** 2 for j in range(m)]) / m
            res.append(round(s, 3))
        return res


    def plan_matrix5(n, m):
        y = np.zeros(shape=(n, m))
        for i in range(n):
            for j in range(m):
                y[i][j] = random.randint(y_min, y_max)

        if n > 14:
            no = n - 14
        else:
            no = 1
        x_norm = ccdesign(3, center=(0, no))
        x_norm = np.insert(x_norm, 0, 1, axis=1)

        for i in range(4, 11):
            x_norm = np.insert(x_norm, i, 0, axis=1)

        l = 1.215

        for i in range(len(x_norm)):
            for j in range(len(x_norm[i])):
                if x_norm[i][j] < -1 or x_norm[i][j] > 1:
                    if x_norm[i][j] < 0:
                        x_norm[i][j] = -l
                    else:
                        x_norm[i][j] = l

        def add_sq_nums(x):
            for i in range(len(x)):
                x[i][4] = x[i][1] * x[i][2]
                x[i][5] = x[i][1] * x[i][3]
                x[i][6] = x[i][2] * x[i][3]
                x[i][7] = x[i][1] * x[i][3] * x[i][2]
                x[i][8] = x[i][1] ** 2
                x[i][9] = x[i][2] ** 2
                x[i][10] = x[i][3] ** 2
            return x

        x_norm = add_sq_nums(x_norm)

        x = np.ones(shape=(len(x_norm), len(x_norm[0])), dtype=np.int64)
        for i in range(8):
            for j in range(1, 4):
                if x_norm[i][j] == -1:
                    x[i][j] = x_range[j - 1][0]
                else:
                    x[i][j] = x_range[j - 1][1]

        for i in range(8, len(x)):
            for j in range(1, 3):
                x[i][j] = (x_range[j - 1][0] + x_range[j - 1][1]) / 2

        dx = [x_range[i][1] - (x_range[i][0] + x_range[i][1]) / 2 for i in range(3)]

        x[8][1] = l * dx[0] + x[9][1]
        x[9][1] = -l * dx[0] + x[9][1]
        x[10][2] = l * dx[1] + x[9][2]
        x[11][2] = -l * dx[1] + x[9][2]
        x[12][3] = l * dx[2] + x[9][3]
        x[13][3] = -l * dx[2] + x[9][3]

        x = add_sq_nums(x)

        print('\nМатриця X:\n')
        for i in x:
            for j in i:
                print("{0:5}".format(j), end='|')
            print()
        print('\nX нормоване:\n')
        for i in x_norm:
            for j in i:
                print("{0:5.3}".format(j), end='|')
            print()
        print('Y')
        for i in y:
            for j in i:
                print("{0:5.8}".format(j), end='|')
            print()

        return x, y, x_norm


    def find_coef(X, Y, norm=False):
        skm = lm.LinearRegression(fit_intercept=False)
        skm.fit(X, Y)
        B = skm.coef_

        if norm == 1:
            print('\nКоефіцієнти рівняння регресії з нормованими X:')
        else:
            print('\nКоефіцієнти рівняння регресії:')
        B = [round(i, 3) for i in B]
        print(B)
        print('\nРезультат рівняння зі знайденими коефіцієнтами:\n', np.dot(X, B))
        return B


    def kriteriy_cochrana(y, y_aver, n, m):
        f1 = m - 1
        f2 = n
        q = 0.05
        S_kv = s_kv(y, y_aver, n, m)
        Gp = max(S_kv) / sum(S_kv)
        print('\nПеревірка за критерієм Кохрена')
        return Gp


    def kohren(f1, f2, q=0.05):
        q1 = q / f1
        fisher_value = scipy.stats.f.ppf(q=1 - q1, dfn=f2, dfd=(f1 - 1) * f2)
        return fisher_value / (fisher_value + f1 - 1)


    def bs(x, y_aver, n):
        res = [sum(1 * y for y in y_aver) / n]

        for i in range(len(x[0])):
            b = sum(j[0] * j[1] for j in zip(x[:, i], y_aver)) / n
            res.append(b)
        return res


    def kriteriy_studenta(x, y, y_aver, n, m):
        S_kv = s_kv(y, y_aver, n, m)
        s_kv_aver = sum(S_kv) / n

        s_Bs = (s_kv_aver / n / m) ** 0.5
        Bs = bs(x, y_aver, n)
        ts = [round(abs(B) / s_Bs, 3) for B in Bs]

        return ts


    def kriteriy_fishera(y, y_aver, y_new, n, m, d):
        S_ad = m / (n - d) * sum([(y_new[i] - y_aver[i]) ** 2 for i in range(len(y))])
        S_kv = s_kv(y, y_aver, n, m)
        S_kv_aver = sum(S_kv) / n

        return S_ad / S_kv_aver


    x_aver_max = sum([x[1] for x in x_range]) / 3
    x_aver_min = sum([x[0] for x in x_range]) / 3

    y_max = 200 + int(x_aver_max)
    y_min = 200 + int(x_aver_min)

    X5, Y5, X5_norm = plan_matrix5(15, 6)

    y5_aver = [round(sum(i) / len(i), 3) for i in Y5]
    B5 = find_coef(X5, y5_aver)

    print('\n\tПеревірка рівняння:')
    f1 = 5
    f2 = 15
    f3 = f1 * f2
    q = 0.05

    student = partial(scipy.stats.t.ppf, q=1 - q)
    t_student = student(df=f3)

    G_kr = kohren(f1, f2)

    y_aver = [round(sum(i) / len(i), 3) for i in Y5]
    print('\nСереднє значення y:', y_aver)

    disp = s_kv(Y5, y_aver, f2, 6)
    print('Дисперсія y:', disp)

    Gp = kriteriy_cochrana(Y5, y_aver, f2, 6)
    print(f'Gp = {Gp}')
    if Gp < G_kr:
        print(f'З ймовірністю', 1 - q, 'дисперсії однорідні.')
    else:
        print("Необхідно збільшити кількість дослідів")

    ts = kriteriy_studenta(X5_norm[:, 1:], Y5, y_aver, f2, 6)
    print('\nКритерій Стьюдента:\n', ts)
    res = [t for t in ts if t > t_student]
    time_start = time.time()
    final_k = [B5[i] for i in range(len(ts)) if ts[i] in res]
    print('\nКоефіцієнти {} статистично незначущі, тому ми виключаємо їх з рівняння.'.format(
        [round(i, 3) for i in B5 if i not in final_k]))
    average_time.append(-(time_start - time.time()) if time_start - time.time() != 0 else 0.005)

    y_new = []
    for j in range(f2):
        y_new.append(regression([X5_norm[j][i] for i in range(len(ts)) if ts[i] in res], final_k))

    print(f'\nЗначення "y" з коефіцієнтами {final_k}')
    print(y_new)

    d = len(res)
    if d >= f2:
        print('\nF4 <= 0')
        print('')
        sys.exit()

    f4 = f2 - d

    F_p = kriteriy_fishera(Y5, y_aver, y_new, f2, 6, d)

    fisher = partial(scipy.stats.f.ppf, q=0.95)
    f_t = fisher(dfn=f4, dfd=f3)
    print('\nПеревірка адекватності за критерієм Фішера')
    print('Fp =', F_p)
    print('F_t =', f_t)
    if F_p < f_t:
        print('Математична модель є адекватною')
    else:
        print('Математична модель не є адекватною')
print("-------------------------------------------------------------------------------------------------------")
print("Додаткове задання")
print(average_time)
print(sum(average_time) / len(average_time), "середній час обрахунку кожної статистичної перевірки, після 10-та ітерацій")
print("-------------------------------------------------------------------------------------------------------")