from random import *
from math import *


m = 5
while True:

    y_MAX = (30 - 20) * 10  # 100
    y_MIN = (20 - 20) * 10  # 0

    x1_MIN = -30
    x1_MAX = 20
    x2_MIN = -30
    x2_MAX = 45


    x_arr = [[-1, 1, -1], [-1, -1, 1]]
    y_arr = [[randrange(y_MIN, y_MAX) for i in range(m)] for i in range(3)]


    Y_AVG = [sum(i) / len(i) for i in y_arr]

    sigma = [sum([(y_arr[i][j] - Y_AVG[i]) ** 2 for j in range(m)]) / m for i in range(3)]
    # ОСНОВНЕ ВІДХИЛЕННЯ
    sig_main = sqrt(2 * (2 * m - 2) / (m * (m - 4)))

    fuv = [max(sigma[int((i + 3) / 2)], sigma[int(i / 2)]) / min(sigma[int((i + 3) / 2)], sigma[int(i / 2)]) for i in
           range(3)]
    teta = [i * (m - 2) / m for i in fuv]
    ruv = [fabs(i - 1) / sig_main for i in teta]

    m_X1 = sum(x_arr[0]) / 3
    m_X2 = sum(x_arr[1]) / 3
    M = sum(Y_AVG) / len(Y_AVG)

    a = [(x_arr[0][0] ** 2 + x_arr[0][1] ** 2 + x_arr[0][2] ** 2) / 3,
         (x_arr[0][0] * x_arr[1][0] + x_arr[0][1] * x_arr[1][1] + x_arr[0][2] * x_arr[1][2]) / 3,
         (x_arr[1][0] ** 2 + x_arr[1][1] ** 2 + x_arr[1][2] ** 2) / 3]
    aij = [sum([x_arr[j][i] * Y_AVG[i] for i in range(3)]) / 3 for j in range(2)]

    matr = lambda matrix: matrix[0][0] * matrix[1][1] * matrix[2][2] + \
                          matrix[0][1] * matrix[1][2] * matrix[2][0] + \
                          matrix[1][0] * matrix[2][1] * matrix[0][2] - \
                          matrix[0][2] * matrix[1][1] * matrix[2][0] - \
                          matrix[0][1] * matrix[1][0] * matrix[2][2] - \
                          matrix[0][0] * matrix[1][2] * matrix[2][1]

    znamennik = matr([
        [1, m_X1, m_X2],
        [m_X1, a[0], a[1]],
        [m_X2, a[1], a[2]]
    ])

    b0 = matr([
        [M, m_X1, m_X2],
        [aij[0], a[0], a[1]],
        [aij[1], a[1], a[2]]
    ]) / znamennik

    b1 = matr([
        [1, M, m_X2],
        [m_X1, aij[0], a[1]],
        [m_X2, aij[1], a[2]]
    ]) / znamennik

    b2 = matr([
        [1, m_X1, M],
        [m_X1, a[0], aij[0]],
        [m_X2, a[1], aij[1]]
    ]) / znamennik

    delta_X1 = fabs(x1_MAX - x1_MIN) / 2
    delta_X2 = fabs(x2_MAX - x2_MIN) / 2

    x10 = (x1_MAX + x1_MIN) / 2
    x20 = (x2_MAX + x2_MIN) / 2

    a0 = b0 - b1 * x10 / delta_X1 - b2 * x20 / delta_X2
    a1 = b1 / delta_X1
    a2 = b2 / delta_X2

    b11 = b0 - b1 - b2
    b22 = b0 + b1 - b2
    b33 = b0 - b1 + b2

    a11 = a0 + a1 * x1_MIN + a2 * x2_MIN
    a22 = a0 + a1 * x1_MAX + a2 * x2_MIN
    a33 = a0 + a1 * x1_MIN + a2 * x2_MAX

    '''Додав перевірку критерію Романовського'''
    #################################################################
    t = 0
    Rkr = {2: [1.73, 1.72, 1.71, 1.69],
           6: [2.16, 2.13, 2.10, 2],
           8: [2.43, 2.37, 2.27, 2.17],
           10: [2.62, 2.54, 2.41, 2.29],
           12: [2.75, 2.66, 2.52, 2.39],
           15: [2.9, 2.8, 2.64, 2.49],
           20: [3.08, 2.96, 2.78, 2.62]}

    nearest_val = 20
    for i in Rkr:
        if abs(i - m) < abs(nearest_val - m):
            nearest_val = i

    if False not in [i < Rkr[nearest_val][t] for i in ruv]:
        break
    else:
        print("\nm =", m)
        print("Ruv =", ruv)
        print("Rкр =", Rkr[nearest_val][t])
        print("\nГіпотеза про однорідність дисперсій не підтверджується! (Ruv > Rkr)\nПотрібно збільшити к-сть дослідів.\n")
        m += 1
        [i.append(randrange(y_MIN, y_MAX)) for i in y_arr]
    ######################################################################

print("\nY Максимальне = %s, Y Мінімальне = %s" % (y_MAX, y_MIN))
print("m = ", m)
print("Y:", y_arr)
print("Y сер.:", Y_AVG)
print("-" * 61)

print("\nσ²:", [round(i, 2) for i in sigma])
print("-" * 61)

print("\nFuv: ", fuv)
print("Ouv: ", teta)
print("Ruv: ", ruv)
print("Rкр = ", Rkr[nearest_val][t])
print("-" * 61)

print("\nНормування: \nmx1 = %.2f, mx2 = %.2f" % (m_X1, m_X2))
print("a:   ", [round(i, 1) for i in a])
print("aij: ", [round(i, 2) for i in aij])
print("-" * 61)

print("\nb0 = %.2f, b1 = %.2f, b2 = %.2f" % (b0, b1, b2))
print("Нормоване рівняння регресії виглядає так: y = %.1f + %.1f * x1 + %.1f * x2" % (b0, b1, b2))
print("-" * 61)

print("\nПеревірка отриманих результатів:")
print("%.1f - %.1f - %.1f = %.1f" % (b0, b1, b2, b11))
print("%.1f + %.1f - %.1f = %.1f" % (b0, b1, b2, b22))
print("%.1f - %.1f + %.1f = %.1f" % (b0, b1, b2, b33))
print("-" * 61)

print("\na0 = %.2f, a1 = %.2f, a2 = %.2f" % (a0, a1, a2))
print("Натуралізоване рівняння регресії: y = %.1f + %.1f * x1 + %.1f * x2" % (a0, a1, a2))
print("%.1f + %.1f * %.f + %.1f * %.f = %.1f" % (a0, a1, x1_MIN, a2, x2_MIN, a11))
print("%.1f + %.1f * %.f + %.1f * %.f = %.1f" % (a0, a1, x1_MAX, a2, x2_MIN, a22))
print("%.1f + %.1f * %.f + %.1f * %.f = %.1f" % (a0, a1, x1_MIN, a2, x2_MAX, a33))