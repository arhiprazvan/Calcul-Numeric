import math
import numpy as np
from scipy.linalg import lu, lu_solve, inv


def get_machine_precision():
    u = 0.1
    m = 0
    while 1 + u != 1:
        u = u / 10
        m += 1

    return m

e = 10**(-get_machine_precision()) # prezcizia masina

def determinant(A):
    det = 1
    for i in range(len(A)):
        det *= A[i][i]
    return det

def decompose_lu(A):
    if not(len(A.shape) == 2 and A.shape[0] == A.shape[1]):
        return "Matricea nu este patratica!"

    if determinant(A) <= e:
        return "Determinantul este 0!"

    n = len(A)

    for p in range(n):
        for i in range(p, n):
            sum_L = sum(A[i][k] * A[k][p] for k in range(p))
            A[i][p] = A[i][p] - sum_L

        for j in range(p + 1, n):
            sum_U = sum(A[p][k] * A[k][j] for k in range(p))
            if abs(A[p][p]) <= e:
                raise ZeroDivisionError(f"Descompunerea LU nu poate fi calculată, A[{p}][{p}] = 0")
            A[p][j] = (A[p][j] - sum_U) / A[p][p]

    return A


def forward_substitution(L, b):
    n = len(b)
    y = np.zeros(n)

    for i in range(n):
        if L[i][i] <= e:
            raise ValueError(f"Elementul diagonal L[{i}][{i}] este zero, sistemul nu poate fi rezolvat.")

        sum_Ly = sum(L[i][j] * y[j] for j in range(i))
        y[i] = (b[i] - sum_Ly) / L[i][i]

    return y


def backward_substitution(U, y):
    n = len(y)
    x = np.zeros(n)

    for i in range(n - 1, -1, -1):
        sum_Ux = sum(U[i][j] * x[j] for j in range(i + 1, n))
        x[i] = y[i] - sum_Ux

    return x


# A = np.random.rand(100, 100)
# b = np.random.rand(100)
A = np.array([[2.5, 2, 2], [5, 6, 5], [5, 6, 6.5]], dtype=float)
A_init = np.copy(A)
b = np.array([2, 2, 2], dtype=float)

print("Matricea A: \n", A_init)
decompose_lu(A)
print("Matricea LU: \n", A)

# substitutia directa
y = forward_substitution(A, b)

# substitutia inversa
x = backward_substitution(A, y)

# norma
norm = np.sqrt(np.sum((np.dot(A_init, x) - b)**2))


print("Determinantul: ", determinant(A))
print("Solutia y:", y)
print("Soluția x:", x)
print("Verificarea normei soluției ||Ax - b||_2:", norm)


# Solutia Ax=b folosind libraria
x_lib = np.linalg.solve(A_init, b)

# Inversa
A_inv_lib = inv(A_init)

# Normele
norm_xLU_xlib = np.linalg.norm(x - x_lib, 2)
norm_xLU_Ainvb = np.linalg.norm(x - np.dot(A_inv_lib, b), 2)


print("\nSoluția sistemului folosind numpy.linalg.solve (x_lib):")
print(x_lib)

print("\nInversa matricei A (A_inv_lib):")
print(A_inv_lib)

print("\nNorma ||x_LU - x_lib||_2:")
print(norm_xLU_xlib)

print("\nNorma ||x_LU - A_inv_lib * b||_2:")
print(norm_xLU_Ainvb)

# bonus
print("---------BONUS-----------")

def decompose_lu_bonus(A):
    if not(len(A.shape) == 2 and A.shape[0] == A.shape[1]):
        return "Matricea nu este patratica!"

    if determinant(A) <= e:
        return "Determinantul este 0!"

    n = len(A)
    size = n * (n + 1) // 2

    lower_vec = np.zeros(size)
    upper_vec = np.ones(size)

    lower_index = 0
    upper_index = 0

    def get_index(i, j):
        if i >= j:
            return i * (i + 1) // 2 + j
        else:
            return j * (j + 1) // 2 + i

    for p in range(n):
        # elementele coloanei p din L
        for i in range(p, n):
            sum_LU = sum(lower_vec[get_index(i, k)] * upper_vec[get_index(k, p)] for k in range(p))
            lower_vec[get_index(i, p)] = A[i, p] - sum_LU
        # elementele liniei p din U
        for j in range(p + 1, n):
            sum_LU = sum(lower_vec[get_index(p, k)] * upper_vec[get_index(k, j)] for k in range(p))
            if abs(lower_vec[get_index(p, p)]) <= e:
                raise ZeroDivisionError(f"Descompunerea LU_bonus nu poate fi calculată, lower_vec[{p}][{p}] = 0")
            upper_vec[get_index(p, j)] = (A[p, j] - sum_LU) / lower_vec[get_index(p, p)]

    return lower_vec, upper_vec

print("Bonus:", decompose_lu_bonus(A_init))

lower_vec, upper_vec = decompose_lu_bonus(A_init)


def forward_substitution(lower_vec, b):
    n = len(b)
    y = np.zeros(n)

    for i in range(n):
        if lower_vec[i * (i + 1) // 2 + i] <= e:  # elementul de pe diagonală din vectorul lower_vec
            raise ValueError(f"Elementul diagonal L[{i}][{i}] este zero, sistemul nu poate fi rezolvat.")

        sum_Ly = sum(lower_vec[i * (i + 1) // 2 + j] * y[j] for j in range(i))
        y[i] = (b[i] - sum_Ly) / lower_vec[i * (i + 1) // 2 + i]

    return y


def backward_substitution(upper_vec, y):
    n = len(y)
    x = np.zeros(n)

    for i in range(n - 1, -1, -1):
        sum_Ux = sum(upper_vec[i * (i + 1) // 2 + j] * x[j] for j in
                     range(i + 1, n))  #  elementele din vectorul upper_vec
        x[i] = (y[i] - sum_Ux) / upper_vec[i * (i + 1) // 2 + i]

    return x



y = forward_substitution(lower_vec, b)
# print("Soluția y pentru L * y = b:")
# print(y)


x = backward_substitution(upper_vec, y)
print("Soluția x pentru U * x = y:")
print(x)