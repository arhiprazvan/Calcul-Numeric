import tkinter as tk


def calculate():
    try:
        x = float(entry1.get())
        y = float(entry2.get())
        z = float(entry3.get())

        if abs(((x + y) + z)) == abs((x + (y + z))):
            result_sum = "Asociativ"
            sum_color = "#297D37"
        else:
            result_sum = "Neasociativ"
            sum_color = "#A51010"

        if abs(((x * y) * z)) == abs((x * (y * z))):
            result_multiply = "Asociativ"
            multiply_color = "#297D37"
        else:
            result_multiply = "Neasociativ"
            multiply_color = "#A51010"

        sum_difference = abs(((x + y) + z) - (x + (y + z)))
        multiply_difference = abs(((x * y) * z) - (x * (y * z)))

        sum_label.config(text=f"Sum: {result_sum}, difference = {sum_difference}", fg=sum_color)
        multiply_label.config(text=f"Multiplication: {result_multiply}, difference = {multiply_difference}", fg=multiply_color)

    except ValueError:
        sum_label.config(text="Eroare!Va rog introduceti un numar valid!", fg="#DBA320")


app = tk.Tk()
app.title("Bonus Tema1")
app.geometry("300x300")

label1 = tk.Label(app, text="First number:")
label1.grid(row=0, column=1,)

entry1 = tk.Entry(app, width=10)
entry1.grid(row=2, column=1)

label2 = tk.Label(app, text="Second number:")
label2.grid(row=3, column=1)

entry2 = tk.Entry(app, width=10)
entry2.grid(row=4, column=1)

label3 = tk.Label(app, text="Third number:")
label3.grid(row=5, column=1)

entry3 = tk.Entry(app, width=10)
entry3.grid(row=6, column=1)

empty_row = tk.Label(app, text="")
empty_row.grid(row=7, column=1, columnspan=6)

check_button = tk.Button(app, text="Check", command=calculate)
check_button.grid(row=8, column=1, columnspan=6)

sum_label = tk.Label(app, text="Sum: ")    # A51010   297D37
sum_label.grid(row=9, column=1, columnspan=3)
multiply_label = tk.Label(app, text="Multiplication: ")
multiply_label.grid(row=10, column=1, columnspan=3)
app.mainloop()
