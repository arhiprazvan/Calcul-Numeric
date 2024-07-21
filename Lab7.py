import tkinter as tk
from tkinter import messagebox
import numpy as np

def horner(poly, n, x):
    result = poly[0]
    for i in range(1, n + 1):
        result = result * x + poly[i]
    return result

def muller(poly, x0, x1, x2, epsilon, max_iter):
    iter_count = 0
    while iter_count < max_iter:
        h0 = x1 - x0
        h1 = x2 - x1

        if abs(h0) < epsilon or abs(h1) < epsilon:
            return None

        delta0 = (horner(poly, len(poly) - 1, x1) - horner(poly, len(poly) - 1, x0)) / h0
        delta1 = (horner(poly, len(poly) - 1, x2) - horner(poly, len(poly) - 1, x1)) / h1

        if abs(h1 + h0) < epsilon:
            return None

        a = (delta1 - delta0) / (h1 + h0)
        b = a * h1 + delta1
        c = horner(poly, len(poly) - 1, x2)
        discriminant = b ** 2 - 4 * a * c

        if discriminant < 0:
            return None

        denom = b + np.sign(b) * np.sqrt(discriminant)
        if abs(denom) < epsilon:
            return None

        delta_x = -2 * c / denom
        x3 = x2 + delta_x

        if abs(delta_x) < epsilon:
            return x3

        x0, x1, x2 = x1, x2, x3
        iter_count += 1

    return None

def find_roots(poly, epsilon, max_iter, start_points):
    roots = []
    for points in start_points:
        root = muller(poly, *points, epsilon, max_iter)
        if root is not None and all(abs(root - r) > epsilon for r in roots):
            roots.append(root)
    return roots

def calculate_roots():
    poly_str = entry_poly.get()
    epsilon = float(entry_epsilon.get())
    max_iter = int(entry_max_iter.get())
    start_points = eval(entry_start_points.get())

    poly = list(map(float, poly_str.split()))

    roots = find_roots(poly, epsilon, max_iter, start_points)
    result_text = "\n".join([str(root) for root in roots])

    result_label.config(text="Distinct roots found:\n" + result_text)

    with open("roots.txt", "w") as file:
        for root in roots:
            file.write(f"{root}\n")

app = tk.Tk()
app.title("Muller's Method Root Finder")

tk.Label(app, text="Polynomial Coefficients (space separated):").pack()
entry_poly = tk.Entry(app, width=50)
entry_poly.pack()

tk.Label(app, text="Epsilon:").pack()
entry_epsilon = tk.Entry(app, width=50)
entry_epsilon.pack()

tk.Label(app, text="Max Iterations:").pack()
entry_max_iter = tk.Entry(app, width=50)
entry_max_iter.pack()

tk.Label(app, text="Starting Points (list of tuples):").pack()
entry_start_points = tk.Entry(app, width=50)
entry_start_points.pack()

calculate_button = tk.Button(app, text="Calculate Roots", command=calculate_roots)
calculate_button.pack()

result_label = tk.Label(app, text="")
result_label.pack()

app.mainloop()
