import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np


def area_base_octagonal():
    try:
        lado = float(entry_lado.get())
        apotema = float(entry_apotema.get())
        area = 4 * lado * apotema
        resultado_label.config(text=f"Área: {area:.2f} unidades²")
    except ValueError:
        messagebox.showerror("Error", "Por favor ingresa valores numéricos válidos.")

def proceso_gram_schmidt():
    try:
        texto = text_vectores.get("1.0", tk.END).strip()
        if not texto:
            messagebox.showwarning("Advertencia", "Debes ingresar los vectores.")
            return
        
        lineas = texto.split("\n")
        vectores = [np.array(list(map(float, linea.split()))) for linea in lineas]
        base = gram_schmidt(vectores)
        
        resultado = "\n".join([f"u{i+1} = {np.round(v, 3)}" for i, v in enumerate(base)])
        resultado_label.config(text=resultado)
    except Exception as e:
        messagebox.showerror("Error", f"Ocurrió un problema:\n{e}")

def gram_schmidt(vectors):
    ortonormal = []
    for v in vectors:
        for u in ortonormal:
            v = v - np.dot(v, u) * u
        v = v / np.linalg.norm(v)
        ortonormal.append(v)
    return ortonormal


#interfaz

root = tk.Tk()
root.title("Calculadora de Base Octagonal y Proceso Gram-Schmidt")
root.geometry("850x500")
root.config(bg="#1E1E1E")

menu_frame = tk.Frame(root, bg="#252526", width=200)
menu_frame.pack(side="left", fill="y")

titulo = tk.Label(menu_frame, text="MENÚ", bg="#252526", fg="white", font=("Segoe UI", 16, "bold"))
titulo.pack(pady=20)

def mostrar_seccion(nombre):
    for frame in [frame_octagonal, frame_gram]:
        frame.pack_forget()
    if nombre == "octagonal":
        frame_octagonal.pack(fill="both", expand=True)
    else:
        frame_gram.pack(fill="both", expand=True)
    resultado_label.config(text="")

boton1 = tk.Button(menu_frame, text="Base Octagonal", bg="#007ACC", fg="white",
                   font=("Segoe UI", 12, "bold"), relief="flat", command=lambda: mostrar_seccion("octagonal"))
boton1.pack(pady=10, fill="x")

boton2 = tk.Button(menu_frame, text="Gram-Schmidt", bg="#007ACC", fg="white",
                   font=("Segoe UI", 12, "bold"), relief="flat", command=lambda: mostrar_seccion("gram"))
boton2.pack(pady=10, fill="x")

content_frame = tk.Frame(root, bg="#1E1E1E")
content_frame.pack(side="right", fill="both", expand=True)

frame_octagonal = tk.Frame(content_frame, bg="#1E1E1E")

titulo_oct = tk.Label(frame_octagonal, text="Cálculo de Base Octagonal", bg="#1E1E1E", fg="white",
                      font=("Segoe UI", 18, "bold"))
titulo_oct.pack(pady=15)

form_frame = tk.Frame(frame_octagonal, bg="#1E1E1E")
form_frame.pack(pady=20)

tk.Label(form_frame, text="Lado:", bg="#1E1E1E", fg="white", font=("Segoe UI", 12)).grid(row=0, column=0, padx=10, pady=10)
entry_lado = ttk.Entry(form_frame, width=10)
entry_lado.grid(row=0, column=1)

tk.Label(form_frame, text="Apotema:", bg="#1E1E1E", fg="white", font=("Segoe UI", 12)).grid(row=1, column=0, padx=10, pady=10)
entry_apotema = ttk.Entry(form_frame, width=10)
entry_apotema.grid(row=1, column=1)

btn_calcular = ttk.Button(frame_octagonal, text="Calcular Área", command=area_base_octagonal)
btn_calcular.pack(pady=10)

frame_gram = tk.Frame(content_frame, bg="#1E1E1E")

titulo_gs = tk.Label(frame_gram, text="Proceso de Gram-Schmidt", bg="#1E1E1E", fg="white",
                     font=("Segoe UI", 18, "bold"))
titulo_gs.pack(pady=15)

tk.Label(frame_gram, text="Introduce los vectores (uno por línea, separados por espacios):",
         bg="#1E1E1E", fg="white", font=("Segoe UI", 11)).pack(pady=5)

text_vectores = tk.Text(frame_gram, height=8, width=50, font=("Consolas", 11))
text_vectores.pack(pady=10)

btn_gs = ttk.Button(frame_gram, text="Ejecutar Gram-Schmidt", command=proceso_gram_schmidt)
btn_gs.pack(pady=10)

resultado_label = tk.Label(content_frame, text="", bg="#1E1E1E", fg="#00FF88",
                           font=("Consolas", 12), justify="left")
resultado_label.pack(pady=10)

# Mostrar por defecto la primera sección
mostrar_seccion("octagonal")

root.mainloop()
