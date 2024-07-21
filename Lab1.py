import numpy as np
import math
import matplotlib.pyplot as plt
np.random.seed(42)

print("------------------ex1--------------------------")


def Exercitiul1():
   for m in range(1, 100):
      result = 1 + 1/(10**m)
      print(f" m = {m} --> rezultat = {result}")
      if result == 1.0 and m > 1:
         return 1 + 1/(10**(m-1)), (m-1), 1/(10**(m - 1))


prop, m, u = Exercitiul1()
print(f"1 + u = {prop}, unde u este {u}, iar m este {m}")

print("--------------------------ex2---------------------------\n")
x = 1.0
y = u/10  # 1.0000000000000001e-16
z = u/10

if abs((x + y) + z) != abs(x + (y + z)):
    print(f"Adunarea efectuata de calculator este NEASOCIATIVA dupa relatia abs((x + y) + z) != abs(x + (y + z))")
else:
    print("Adunarea efectuata de calculator este ASOCIATIVA!")

x = 0.1  # exemplul pentru care operatia de inmultire Xc este neasociativa

if abs((x*y)*z) != abs(x*(y*z)):
    print(f"Operatia de inmultire a calculatorului este NEASOCIATIVA pentru valoarea x = {x}")
else:
    print(f"Operatia de inmultire a calculatorului este ASOCIATIVA pentru valoarea x = {x}")


print("--------------------------ex3----------------------------\n")
#ex 3

def tan4(a):
   return (105*a - 10*a**3)/(105-45*a**2 + a**4)
def tan5(a):
   return (945*a - 105*a**3 + a**5)/(945 - 420*a**2 + 15*a**4)
def tan6(a):
   return (10395*a-1260*a**3+21*a**5)/(10395-4725*a**2+210*a**4-a**6)
def tan7(a):
   return (135135*a-17325*a**3+378*a**5-a**7)/(135135-62370*a**2+3150*a**4-28*a**6)
def tan8(a):
   return (2027025*a-270270*a**3+6930*a**5-36*a**7)/(2027025-945945*a**2 + 51975*a**4-630*a**6+a**8)
def tan9(a):
   return (34459425*a-4729725*a**3+135135*a**5-990*a**7+a**9)/(34459425-16216200*a**2+945945*a**4-13860*a**6+45*a**8)


numere = np.random.uniform(-np.pi/2, np.pi/2, 10000)
rezultate = {
    4: [],
    5: [],
    6: [],
    7: [],
    8: [],
    9: [],
}
erori = {
    4: 0,
    5: 0,
    6: 0,
    7: 0,
    8: 0,
    9: 0,
}
rezultateValExacta = [math.tan(a) for a in numere]

for a in numere:
    rezultate[4].append(tan4(a))
    rezultate[5].append(tan5(a))
    rezultate[6].append(tan6(a))
    rezultate[7].append(tan7(a))
    rezultate[8].append(tan8(a))
    rezultate[9].append(tan9(a))

# Calculăm erorile
for key in erori.keys():
    erori[key] = sum(abs(rez - val_exacta) for rez, val_exacta in zip(rezultate[key], rezultateValExacta))

# Afișăm rezultatele
print("Aproximarile:")
for key in sorted(erori.keys(), reverse=True):
    print(f"Pentru T({key},a) = {erori[key]}")

print("------------------BONUS----------------------")

def sin4(a):
   return tan4(a)/math.sqrt(1+(tan4(a))**2)
def sin5(a):
   return tan5(a)/math.sqrt(1+(tan5(a))**2)
def sin6(a):
   return tan6(a)/math.sqrt(1+(tan6(a))**2)
def sin7(a):
   return tan7(a)/math.sqrt(1+(tan7(a))**2)
def sin8(a):
   return tan8(a)/math.sqrt(1+(tan8(a))**2)
def sin9(a):
   return tan9(a)/math.sqrt(1+(tan9(a))**2)
def cos4(a):
   return 1/math.sqrt(1+(tan4(a))**2)
def cos5(a):
   return 1/math.sqrt(1+(tan5(a))**2)
def cos6(a):
   return 1/math.sqrt(1+(tan6(a))**2)
def cos7(a):
   return 1/math.sqrt(1+(tan7(a))**2)
def cos8(a):
   return 1/math.sqrt(1+(tan8(a))**2)
def cos9(a):
   return 1/math.sqrt(1+(tan9(a))**2)

def erori_sin_cos():
    sinerr4, sinerr5, sinerr6, sinerr7, sinerr8, sinerr9, coserr4, coserr5, coserr6, coserr7, coserr8, coserr9 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    for i in numere:
        sinerr4 = sinerr4 + abs(sin4(i) - math.sin(i))
        sinerr5 = sinerr5 + abs(sin5(i) - math.sin(i))
        sinerr6 = sinerr6 + abs(sin6(i) - math.sin(i))
        sinerr7 = sinerr7 + abs(sin7(i) - math.sin(i))
        sinerr8 = sinerr8 + abs(sin8(i) - math.sin(i))
        sinerr9 = sinerr9 + abs(sin9(i) - math.sin(i))

        coserr4 = coserr4 + abs(cos4(i) - math.cos(i))
        coserr5 = coserr5 + abs(cos5(i) - math.cos(i))
        coserr6 = coserr6 + abs(cos6(i) - math.cos(i))
        coserr7 = coserr7 + abs(cos7(i) - math.cos(i))
        coserr8 = coserr8 + abs(cos8(i) - math.cos(i))
        coserr9 = coserr9 + abs(cos9(i) - math.cos(i))

    print("Erori sin si cos: ")
    print(f"Sin4 = {sinerr4}")
    print(f"Sin5 = {sinerr5}")
    print(f"Sin6 = {sinerr6}")
    print(f"Sin7 = {sinerr7}")
    print(f"Sin8 = {sinerr8}")
    print(f"Sin9 = {sinerr9}")
    print(f"Cos4 = {coserr4}")
    print(f"Cos5 = {coserr5}")
    print(f"Cos6 = {coserr6}")
    print(f"Cos7 = {coserr7}")
    print(f"Cos8 = {coserr8}")
    print(f"Cos9 = {coserr9}")


erori_sin_cos()
