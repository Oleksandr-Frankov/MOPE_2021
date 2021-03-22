from random import *

#довільно вибрані коефіцієнти
a0, a1, a2, a3 = 4, 7, 5, 10

#значення факторів у точках експерименту
X1 = [randint(1,20) for i in range(8)]
X2 = [randint(1,20) for i in range(8)]
X3 = [randint(1,20) for i in range(8)]

#знаходимо рівняння регресії
Y = [a0 + a1*X1[i] + a2*X2[i] + a3*X3[i] for i in range(8)]

#обчислюємо значення x0 та обчислюємо інтервал зміни фактора
X_01 = (max(X1)+min(X1))/2
dX1 = X_01-min(X1)

X_02 = (max(X2)+min(X2))/2
dX2 = X_02-min(X2)

X_03 = (max(X3)+min(X3))/2
dX3 = X_03-min(X3)

#знаходимо нормоване значення ХН для кожного фактора
XH1 = [round(((X1[i] - X_01)/dX1), 3) for i in range(8)]
XH2 = [round(((X2[i] - X_02)/dX2), 3) for i in range(8)]
XH3 = [round(((X3[i] - X_03)/dX3), 3) for i in range(8)]

#знаходження Y еталонне
Y_et = a0 + a1*X_01 + a2*X_02 + a3*X_03

#знаходження самої ф-ції за варіантом
F = [((Y[i] - Y_et)**2) for i in range(8)]
answer = max(F)
print("Коефіцієнти:\na0 = %s, a1 = %s, a2 = %s, a3 = %s" % (a0, a1, a2, a3))
print("-" * 61)
print("№ | X1 X2 X3 | Y3 | XH1 XH2 XH3 | (Y-Yет)^2 |")
print("-" * 61)
for i in range(8):
    print(f"{i + 1:^1} |{X1[i]:^4} {X2[i]:^4} {X3[i]:^4} |"
          f" {Y[i]:^5} | {'%.2f' % XH1[i]:^5} {'%.2f' % XH2[i]:^5} {'%.2f' % XH3[i]:^5} | {'%.2f' % F[i]:^8} |")
print(f"X0| {X_01:^4} {X_02:^4} {X_03:^4}|")
print(f"dx| {dX1:^4} {dX2:^4} {dX3:^4}|")
print('-' * 78 + '\nВІДПОВІДЬ:')
print("Yет = %s" % Y_et)
print("(Y-Yет)^2 = %s" % F)
print("max(Y-Yет)^2 = %s" % answer)
print('-' * 78)