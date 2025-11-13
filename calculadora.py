import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# -----------------------------
# MÉTODO DE GRAM-SCHMIDT
# -----------------------------
def gram_schmidt(vectors):
    ortonormal = []
    for v in vectors:
        w = v.copy().astype(float)
        for u in ortonormal:
            w = w - np.dot(w, u) * u
        norm = np.linalg.norm(w)
        if norm > 1e-12:
            ortonormal.append(w / norm)
    if len(ortonormal) == 0:
        return np.zeros((0, vectors.shape[1]))
    return np.vstack(ortonormal)


def proyectar_a_3d(matriz):
    m, n = matriz.shape
    if n == 3:
        return matriz.copy()
    if n < 3:
        padded = np.zeros((m, 3))
        padded[:, :n] = matriz
        return padded
    X = matriz - np.mean(matriz, axis=0)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    V3 = Vt[:3].T
    X3 = X.dot(V3)
    return X3


# -----------------------------
# APLICACIÓN PRINCIPAL
# -----------------------------
class GramSchmidtApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🔹 Visualizador 3D - Método de Gram-Schmidt")
        self.root.geometry("1150x720")
        self.root.configure(bg="#E6E6E6")

        self.matriz_guardada = None
        self.setup_ui()

    def setup_ui(self):
        estilo = ttk.Style()
        estilo.configure("TLabel", background="#E6E6E6", font=("Segoe UI", 10))
        estilo.configure("TButton", font=("Segoe UI", 10, "bold"), padding=6)
        estilo.configure("TEntry", padding=4)

        panel = ttk.Frame(self.root, padding=15, style="Card.TFrame")
        panel.pack(side="left", fill="y")

        # Encabezado lateral
        ttk.Label(panel, text="⚙️ Configuración de la Matriz", font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=(0, 5))

        ttk.Label(panel, text="Filas (vectores):").pack(anchor="w")
        self.entry_filas = ttk.Entry(panel)
        self.entry_filas.insert(0, "3")
        self.entry_filas.pack(fill="x")

        ttk.Label(panel, text="Columnas (dimensión):").pack(anchor="w", pady=(5, 0))
        self.entry_columnas = ttk.Entry(panel)
        self.entry_columnas.insert(0, "3")
        self.entry_columnas.pack(fill="x")

        ttk.Button(panel, text="➕ Crear Matriz", command=self.crear_matriz).pack(pady=10, fill="x")
        ttk.Button(panel, text="🎲 Generar Aleatoria", command=self.generar_aleatoria).pack(pady=5, fill="x")
        ttk.Button(panel, text="💾 Guardar Matriz", command=self.guardar_matriz).pack(pady=5, fill="x")
        ttk.Button(panel, text="▶ Aplicar Gram-Schmidt", command=self.mostrar_animacion).pack(pady=15, fill="x")

        ttk.Separator(panel, orient="horizontal").pack(fill="x", pady=10)
        ttk.Label(panel, text="Ejemplos Rápidos (Matrices hasta 12×12)", font=("Segoe UI", 9, "bold")).pack(anchor="w", pady=(5, 0))

        self.combo_ejemplos = ttk.Combobox(panel, values=[f"{i}x{i}" for i in range(2, 13)], state="readonly")
        self.combo_ejemplos.set("Seleccionar tamaño")
        self.combo_ejemplos.pack(fill="x", pady=5)
        ttk.Button(panel, text="📐 Crear Ejemplo", command=self.crear_ejemplo).pack(pady=5, fill="x")

        # Área gráfica
        self.fig = plt.Figure(figsize=(7, 6), facecolor="#F9F9F9")
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.ax.set_title("Gram-Schmidt: Vectores Originales y Ortogonales", fontsize=12, pad=20)
        self.ax.set_facecolor("#FAFAFA")
        self.ax.grid(True, linestyle="--", alpha=0.4)
        self.ax.set_box_aspect([1, 1, 1])
        self.ax.set_xlim(-1, 1)
        self.ax.set_ylim(-1, 1)
        self.ax.set_zlim(-1, 1)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(side="right", expand=True, fill="both")

    # -----------------------------
    # CREAR MATRIZ MANUAL
    # -----------------------------
    def crear_matriz(self):
        try:
            filas = int(self.entry_filas.get())
            columnas = int(self.entry_columnas.get())

            self.matriz_entradas = []
            ventana = tk.Toplevel(self.root)
            ventana.title("Ingresar valores de la matriz")
            ventana.geometry("420x350")
            ventana.configure(bg="#ECECEC")

            ttk.Label(ventana, text="Introduce los valores de la matriz:", font=("Segoe UI", 10, "bold")).pack(pady=5)
            frame = ttk.Frame(ventana)
            frame.pack(pady=10)

            for i in range(filas):
                fila = []
                for j in range(columnas):
                    e = ttk.Entry(frame, width=6, justify="center")
                    e.grid(row=i, column=j, padx=3, pady=3)
                    e.insert(0, np.random.randint(-3, 4))
                    fila.append(e)
                self.matriz_entradas.append(fila)

            ttk.Button(ventana, text="Guardar valores", command=lambda: self._guardar_ventana(ventana)).pack(pady=10)
        except ValueError:
            messagebox.showerror("Error", "Introduce valores numéricos válidos.")

    def _guardar_ventana(self, ventana):
        try:
            matriz = []
            for fila in self.matriz_entradas:
                matriz.append([float(e.get()) for e in fila])
            self.matriz_guardada = np.array(matriz)
            ventana.destroy()
            messagebox.showinfo("Éxito", "Matriz guardada correctamente.")
        except Exception:
            messagebox.showerror("Error", "Verifica los valores de la matriz.")

    # -----------------------------
    # GENERAR MATRIZ ALEATORIA
    # -----------------------------
    def generar_aleatoria(self):
        try:
            filas = int(self.entry_filas.get())
            columnas = int(self.entry_columnas.get())
            self.matriz_guardada = np.random.randint(-5, 6, size=(filas, columnas)).astype(float)
            messagebox.showinfo("Generada", "Matriz aleatoria generada exitosamente.")
        except ValueError:
            messagebox.showerror("Error", "Introduce valores válidos para filas y columnas.")

    # -----------------------------
    # GUARDAR MATRIZ LOCALMENTE
    # -----------------------------
    def guardar_matriz(self):
        if self.matriz_guardada is None:
            messagebox.showwarning("Atención", "No hay matriz para guardar.")
            return
        np.savetxt("matriz_guardada.txt", self.matriz_guardada, fmt="%.4f")
        messagebox.showinfo("Guardado", "Matriz guardada en 'matriz_guardada.txt'.")

    # -----------------------------
    # CREAR EJEMPLOS
    # -----------------------------
    def crear_ejemplo(self):
        seleccion = self.combo_ejemplos.get()
        if "x" not in seleccion:
            messagebox.showwarning("Atención", "Selecciona un tamaño de ejemplo válido.")
            return
        n = int(seleccion.split("x")[0])
        self.matriz_guardada = np.random.randint(-4, 5, size=(n, n)).astype(float)
        messagebox.showinfo("Ejemplo creado", f"Ejemplo de matriz {n}×{n} generado correctamente.")

    # -----------------------------
    # MOSTRAR ANIMACIÓN / VISUALIZACIÓN
    # -----------------------------
    def mostrar_animacion(self):
        try:
            if self.matriz_guardada is None:
                messagebox.showwarning("Advertencia", "Primero crea o genera una matriz.")
                return

            matriz = self.matriz_guardada
            ortonormal = gram_schmidt(matriz)

            m, n = matriz.shape
            k = ortonormal.shape[0]
            if k < min(m, n):
                messagebox.showinfo("Nota", f"Gram-Schmidt produjo {k} vectores ortonormales "
                                            f"(de {m} vectores). Algunos eran dependientes.")

            matriz_3d = proyectar_a_3d(matriz)
            ortonormal_3d = proyectar_a_3d(ortonormal)

            self.ax.clear()
            self.ax.set_box_aspect([1, 1, 1])
            self.ax.set_xlim(-1, 1)
            self.ax.set_ylim(-1, 1)
            self.ax.set_zlim(-1, 1)
            self.ax.set_title("Gram-Schmidt: Originales vs Ortogonales", fontsize=12, pad=20)
            self.ax.set_xlabel("X")
            self.ax.set_ylabel("Y")
            self.ax.set_zlabel("Z")

            colores = plt.cm.tab20(np.linspace(0, 1, max(m, k)))
            origen = np.zeros(3)

            for i, v in enumerate(matriz_3d):
                self.ax.quiver(*origen, *v, color=colores[i % len(colores)],
                               alpha=0.4, linewidth=2, label=f"O{i+1}")

            for i, v in enumerate(ortonormal_3d):
                self.ax.quiver(*origen, *v, color=colores[i % len(colores)],
                               linewidth=3, label=f"Q{i+1}")

            # Nueva ubicación y diseño de la leyenda
            box = self.ax.get_position()
            self.ax.set_position([box.x0 - 0.05, box.y0, box.width * 0.8, box.height])
            legend = self.ax.legend(loc='center left', bbox_to_anchor=(1.05, 0.5),
                                    fontsize=8, frameon=True, title="Vectores",
                                    facecolor="#FFFFFF", edgecolor="#CCCCCC", fancybox=True)
            legend.get_title().set_fontweight("bold")

            self.canvas.draw()

        except Exception as e:
            messagebox.showerror("Error", f"Verifica los datos ingresados.\n{e}")


# -----------------------------
# EJECUCIÓN
# -----------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = GramSchmidtApp(root)
    root.mainloop()
