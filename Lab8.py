import tkinter as tk
import numpy as np


def gradient_analytic(F, grad_F, x, y):
    return grad_F(x, y)


def gradient_approx(F, x, y, h=1e-5):
    dF_dx = (3 * F(x, y) - 4 * F(x - h, y) + F(x - 2 * h, y)) / (2 * h)
    dF_dy = (3 * F(x, y) - 4 * F(x, y - h) + F(x, y - 2 * h)) / (2 * h)
    return np.array([dF_dx, dF_dy])


def gradient_descent(F, grad_F, x0, y0, epsilon, max_iter, learning_rate_type="constant", beta=0.8):
    x, y = x0, y0
    history = [(x, y)]
    for k in range(max_iter):
        grad = grad_F(F, x, y)
        if learning_rate_type == "constant":
            eta = 1e-3
        elif learning_rate_type == "backtracking":
            eta = 1
            while F(x - eta * grad[0], y - eta * grad[1]) > F(x, y) - eta / 2 * np.linalg.norm(
                    grad) ** 2 and eta > 1e-8:
                eta *= beta

        x_new, y_new = x - eta * grad[0], y - eta * grad[1]

        if np.linalg.norm([x_new - x, y_new - y]) < epsilon:
            break

        x, y = x_new, y_new
        history.append((x, y))

    return x, y, history


def F1(x, y):
    return x ** 2 + y ** 2 - 2 * x - 4 * y - 1


def grad_F1(x, y):
    return np.array([2 * x - 2, 2 * y - 4])


def F2(x, y):
    return 3 * x ** 2 - 12 * x + 2 * y ** 2 + 16 * y - 10


def grad_F2(x, y):
    return np.array([6 * x - 12, 4 * y + 16])


def F3(x, y):
    return x ** 2 - 4 * x * y + 5 * y ** 2 - 4 * y + 3


def grad_F3(x, y):
    return np.array([2 * x - 4 * y, -4 * x + 10 * y - 4])


def F4(x, y):
    return x ** 2 * y - 2 * x * y ** 2 + 3 * x * y + 4


def grad_F4(x, y):
    return np.array([2 * x * y - 2 * y ** 2 + 3 * y, x ** 2 - 4 * x * y + 3 * x])


def run_algorithm():
    func_name = func_var.get()
    method = method_var.get()
    x0 = float(entry_x0.get())
    y0 = float(entry_y0.get())
    epsilon = float(entry_epsilon.get())
    max_iter = int(entry_max_iter.get())

    if func_name == "F1":
        F, grad_F = F1, grad_F1
    elif func_name == "F2":
        F, grad_F = F2, grad_F2
    elif func_name == "F3":
        F, grad_F = F3, grad_F3
    elif func_name == "F4":
        F, grad_F = F4, grad_F4

    if method == "Analytic":
        grad_method = lambda F, x, y: gradient_analytic(F, grad_F, x, y)
    elif method == "Approximate":
        grad_method = lambda F, x, y: gradient_approx(F, x, y)

    x_min, y_min, history = gradient_descent(F, grad_method, x0, y0, epsilon, max_iter)

    result_label.config(text=f"Minim aproximat la: x={x_min}, y={y_min}")


app = tk.Tk()
app.title("Gradient Descent Minimizer")

tk.Label(app, text="Select Function:").pack()
func_var = tk.StringVar(value="F1")
tk.Radiobutton(app, text="F1", variable=func_var, value="F1").pack()
tk.Radiobutton(app, text="F2", variable=func_var, value="F2").pack()
tk.Radiobutton(app, text="F3", variable=func_var, value="F3").pack()
tk.Radiobutton(app, text="F4", variable=func_var, value="F4").pack()

tk.Label(app, text="Gradient Method:").pack()
method_var = tk.StringVar(value="Analytic")
tk.Radiobutton(app, text="Analytic", variable=method_var, value="Analytic").pack()
tk.Radiobutton(app, text="Approximate", variable=method_var, value="Approximate").pack()

tk.Label(app, text="Initial x0:").pack()
entry_x0 = tk.Entry(app)
entry_x0.pack()

tk.Label(app, text="Initial y0:").pack()
entry_y0 = tk.Entry(app)
entry_y0.pack()

tk.Label(app, text="Epsilon:").pack()
entry_epsilon = tk.Entry(app)
entry_epsilon.pack()

tk.Label(app, text="Max Iterations:").pack()
entry_max_iter = tk.Entry(app)
entry_max_iter.pack()

tk.Button(app, text="Run", command=run_algorithm).pack()

result_label = tk.Label(app, text="")
result_label.pack()

app.mainloop()
