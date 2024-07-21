import sys
import numpy as np


n = np.random.randint(2, 5)
A = np.random.uniform(-1000, 1000, size=(n, n))
s = np.random.uniform(-1000, 1000, size=n)


print("matricea initiala: ", A)

epsilon = 1e-9
I = np.identity(n)
Q = I
Ainit = np.copy(A)
def calcul_vector_b(A, s):
    return np.dot(A, s)

def householder(A):
    determinant = np.linalg.det(A)

    if determinant <= epsilon:
        print("Matrice singulara!")
        sys.exit(0)

    for r in range(n-1):
        sigma = 0.0
        for i in range(r, n):
            sigma += A[i, r]**2

        if np.abs(sigma) <= epsilon:
            print("Matrice singulara!")
            sys.exit(0)

        k = np.sqrt(sigma)
        if A[r,r] >= epsilon:
            k = -k

        Beta = sigma - k * A[r, r]
        U = np.zeros(n)
        U[r] = A[r, r] - k
        for i in range(r + 1, n):
            U[i] = A[i,r]

        for j in range(r + 1, n):           # transformarea coloanelor
            Gama = 0.0

            for i in range(r, n):
                Gama =  Gama + U[i] * A[i,j]

            Gama = Gama / Beta
            for i in range(r, n):
                A[i, j] = A[i, j] - Gama * U[i]

        A[r, r] = k
        for i in range(r + 1, n):
            A[i, r] = 0.0

        for i in range(r, n):
            Gama = Gama + U[i] * B[i]
        Gama = Gama / Beta

        for i in range(r, n):
            B[i] = B[i] - Gama * U[i]

        for j in range(n):
            Gama = 0
            for i in range(r, n):
                Gama = Gama + U[i] * Q[i, j]
            Gama = Gama / Beta

            for i in range(r, n):
                Q[i, j] = Q[i, j] - Gama * U[i]


def rezolvare_sistem(A, B):
    X = np.zeros(n)

    for i in range(n - 1, -1, -1):
        suma = sum(A[i, j] * X[j] for j in range(i + 1, n))
        if np.abs(A[i, i]) <= epsilon:
            print("Error")
            sys.exit(0)
        X[i] = (B[i] - suma) / A[i, i]

    return X

def calculul_inversei(A, Q):
    for i in range(n):
        if(np.abs(A[i, i]) <= epsilon):
            break
    matrice_rezultat = np.zeros((n, n))
    for j in range(n):
        b = np.zeros(n)
        for i in range(n):
            b[i] = Q[i, j]

        coloana = rezolvare_sistem(A, b)
        for i in range(n):
            matrice_rezultat[i, j] = coloana[i]

    return matrice_rezultat



B = calcul_vector_b(A, s)
Binit = calcul_vector_b(Ainit, s)

# Ex 3
print("----------------Ex3-----------------")
householder(A)
print("\nQ / A householder: \n", A)
# B = calcul_vector_b(A, s)

Xhouseholder = rezolvare_sistem(A, B)
Qb, Rb = np.linalg.qr(Ainit)

Ibibl = np.eye(A.shape[0])
X = np.linalg.solve(Rb, np.dot(Qb.T, Ibibl))

B = calcul_vector_b(Rb, s)
Xbiblioteca = rezolvare_sistem(Rb, B)

print("Solutia X-householder: ", Xhouseholder)
print("Solutia X-biblioteca: ", Xbiblioteca)

rezultat1 = np.zeros(n)
norma = 0
for i in range(n):
    rezultat1[i] = Xbiblioteca[i] - Xhouseholder[i]
for i in range(n):
    norma = norma + rezultat1[i]**2
norma = np.sqrt(norma)
print("Norma Xqr - Xhouseholder = ", norma)

# Ex 4
print("----------------Ex4-----------------")
norma2 = 0
norma3 = 0
rezultat2 = np.dot(Ainit, Xhouseholder) - Binit    # Householder
for i in range(n):
    norma2 = norma2 + rezultat2[i]**2
norma2 = np.sqrt(norma2)
print("Prima norma: ", norma2)

print("matricea initiala dupa norma:", Ainit)

rezultat3 = np.dot(Ainit, Xbiblioteca) - Binit    # QR
for i in range(n):
    norma3 = norma3 + rezultat3[i]**2
norma3 = np.sqrt(norma3)
print("A 2 a norma: ", norma3)
print("Prima eroare = ", np.abs(norma2 - norma3))

norma4 = 0    # Xhs - S
for i in range(n):
    norma4 = norma4 + (Xhouseholder[i] - s[i])**2
norma4 = np.sqrt(norma4)
print("A 3 a norma: ", norma4)

normaS = 0
for i in range(n):
    normaS = normaS + s[i]**2
normaS = np.sqrt(normaS)
print("Norma S: ", normaS)
norma4fin = norma4/normaS

norma5 = 0    # Xqr - S
for i in range(n):
    norma5 = norma5 + (Xbiblioteca[i] - s[i])**2
norma5 = np.sqrt(norma5)
print("A 5 a norma: ", norma5)
norma5fin = norma5 / normaS

print("A doua eroare = ", np.abs(norma4fin - norma5fin))

# Ex 5
print("----------------Ex5-----------------")
print("Inversa matricei: ", calculul_inversei(A, Q))

inversaHouseholder = calculul_inversei(A, Q)


Q, R = np.linalg.qr(Ainit)
b = calcul_vector_b(Ainit, s)

Xsolutie = rezolvare_sistem(R, b)

A_inv = np.zeros_like(A)


for i in range(A.shape[0]):
    b = np.eye(A.shape[0])[:, i]
    x = rezolvare_sistem(R, Q.T @ b)
    A_inv[:, i] = x

inversaBiblioteca = A_inv

rezultat = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        rezultat[i,j] = inversaHouseholder[i, j] - inversaBiblioteca[i, j]

suma = 0
for i in range(n):
    for j in range(n):
        suma = suma + rezultat[i, j]**2
normaInversa = np.sqrt(suma)
print("Norma inversa = ", normaInversa)