import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.pyplot as plt
import os
import time

matplotlib.use("TkAgg")


# -----------------------------
# UTILIDADES
# -----------------------------
def safe_float(x):
    try:
        return float(x)
    except Exception:
        return None


class ToolTip:
    """Small tooltip class for Tk widgets"""
    def __init__(self, widget, text, wait=500):
        self.widget = widget
        self.text = text
        self.wait = wait
        self.tipwindow = None
        self.id = None
        widget.bind("<Enter>", self.schedule)
        widget.bind("<Leave>", self.hide)

    def schedule(self, _=None):
        self.unschedule()
        self.id = self.widget.after(self.wait, self.show)

    def unschedule(self):
        if self.id:
            self.widget.after_cancel(self.id)
            self.id = None

    def show(self):
        if self.tipwindow:
            return
        x, y, cx, cy = self.widget.bbox("insert") if self.widget.winfo_viewable() else (0,0,0,0)
        x = x + self.widget.winfo_rootx() + 25
        y = y + self.widget.winfo_rooty() + 20
        self.tipwindow = tw = tk.Toplevel(self.widget)
        tw.overrideredirect(True)
        tw.attributes("-topmost", True)
        label = tk.Label(tw, text=self.text, justify=tk.LEFT,
                         background="#ffffe0", relief=tk.SOLID, borderwidth=1,
                         font=("Segoe UI", 9))
        label.pack(ipadx=6, ipady=3)

    def hide(self, _=None):
        self.unschedule()
        if self.tipwindow:
            self.tipwindow.destroy()
            self.tipwindow = None


# -----------------------------
# MÉTODO DE GRAM-SCHMIDT (modificado)
# -----------------------------
def modified_gram_schmidt(V):
    """
    Gram-Schmidt modificado: mejora la estabilidad numérica.
    Entrada: V (m x n) -> filas = vectores (cada fila es un vector)
    Devuelve: Q (k x n) -> filas ortonormales; k = rango
    """
    if V is None or len(V) == 0:
        return np.zeros((0, V.shape[1] if V is not None else 0))
    # Convertir a array flotante
    A = np.array(V, dtype=float, copy=True)
    m, n = A.shape
    Q_list = []
    for i in range(m):
        v = A[i].copy()
        for q in Q_list:
            # proyección
            proj = np.dot(v, q) * q
            v = v - proj
        norm = np.linalg.norm(v)
        if norm > 1e-12:
            q = v / norm
            Q_list.append(q)
        # else: vector dependiente -> se ignora
    if len(Q_list) == 0:
        return np.zeros((0, n))
    return np.vstack(Q_list)


def project_to_3d(M):
    """
    Proyecta M (m x n) a 3D (m x 3).
    Si n < 3 => pad zeros. Si n == 3 => copia.
    Si n > 3 => PCA (SVD) para reducir a 3 dimensiones centradas.
    """
    M = np.array(M, dtype=float)
    m, n = M.shape
    if n == 3:
        return M.copy()
    if n < 3:
        padded = np.zeros((m, 3), dtype=float)
        padded[:, :n] = M
        return padded
    # Centrar
    X = M - np.mean(M, axis=0)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    V3 = Vt[:3].T  # columnas principales
    X3 = X.dot(V3)
    return X3


# -----------------------------
# APLICACIÓN PRINCIPAL
# -----------------------------
class GramSchmidtApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🔷 Gram-Schmidt Dashboard — Visualizador 3D")
        self.root.geometry("1200x760")
        self.root.minsize(1000, 640)
        self.style = ttk.Style()
        self._configure_style()

        # Estado
        self.matriz = None                # matriz original (m x n) filas = vectores
        self.ortonormales = None          # matriz Q (k x n)
        self.anim_running = False
        self.anim_index = 0
        self.anim_speed_ms = 600          # milisegundos entre pasos
        self.after_id = None

        # Setup UI
        self._create_layout()
        self._create_plot()
        self._bind_shortcuts()

        # Mensaje de bienvenida en barra de estado
        self.set_status("Listo — crea o carga una matriz para comenzar.")

    # -----------------------------
    # ESTILO
    # -----------------------------
    def _configure_style(self):
        # Estética general
        self.style.theme_use('clam')  # 'clam' permite más personalización
        bg_main = "#F2F4F8"
        accent = "#0B6FA4"
        panel = "#FFFFFF"
        self.root.configure(bg=bg_main)
        self.style.configure("Header.TFrame", background=accent)
        self.style.configure("Header.TLabel", background=accent, foreground="#FFFFFF",
                             font=("Segoe UI", 14, "bold"))
        self.style.configure("Sidebar.TFrame", background=panel)
        self.style.configure("Card.TFrame", background=panel, relief="flat")
        self.style.configure("TLabel", background=bg_main, font=("Segoe UI", 10))
        self.style.configure("TButton", font=("Segoe UI", 10, "bold"))
        self.style.configure("Small.TButton", font=("Segoe UI", 9))
        self.style.configure("Status.TLabel", background=bg_main, font=("Segoe UI", 9))
        # Custom ttk map for hovered buttons (subtle)
        self.style.map("TButton",
                       background=[("active", "#E6F4FF")],
                       relief=[("pressed", "sunken"), ("!pressed", "flat")])

    # -----------------------------
    # LAYOUT: header, sidebar, main, right panel, statusbar
    # -----------------------------
    def _create_layout(self):
        # Header
        header = ttk.Frame(self.root, style="Header.TFrame", padding=(12, 8))
        header.pack(side="top", fill="x")
        ttk.Label(header, text="🔷 Gram-Schmidt Dashboard", style="Header.TLabel").pack(side="left", padx=(6, 12))
        ttk.Label(header, text="Visualizador 3D · Gram–Schmidt modificado", style="Header.TLabel",
                  font=("Segoe UI", 10, "normal")).pack(side="left", padx=(6, 12))

        # Main container
        container = ttk.Frame(self.root)
        container.pack(side="top", fill="both", expand=True, padx=10, pady=(10, 6))

        # Sidebar (izquierda)
        self.sidebar = ttk.Frame(container, width=260, style="Sidebar.TFrame", padding=(10, 10))
        self.sidebar.pack(side="left", fill="y")
        self._populate_sidebar(self.sidebar)

        # Centro (plot)
        self.center = ttk.Frame(container, style="Card.TFrame", padding=8)
        self.center.pack(side="left", fill="both", expand=True, padx=(10, 6))

        # Right panel (controles y detalles)
        right_panel = ttk.Frame(container, width=320, style="Sidebar.TFrame", padding=(10, 10))
        right_panel.pack(side="right", fill="y")
        self._populate_rightpanel(right_panel)

        # Status bar
        status = ttk.Frame(self.root, padding=(8, 6))
        status.pack(side="bottom", fill="x")
        self.status_label = ttk.Label(status, text="", style="Status.TLabel")
        self.status_label.pack(side="left")

    # -----------------------------
    # SIDEBAR: creación y carga de matrices
    # -----------------------------
    def _populate_sidebar(self, parent):
        ttk.Label(parent, text="Configuración", font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=(0, 6))

        # Filas y columnas
        frame_dim = ttk.Frame(parent)
        frame_dim.pack(fill="x", pady=4)
        ttk.Label(frame_dim, text="Filas (vectores):").grid(row=0, column=0, sticky="w")
        ttk.Label(frame_dim, text="Columnas (dim):").grid(row=1, column=0, sticky="w")
        self.entry_rows = ttk.Entry(frame_dim, width=8)
        self.entry_cols = ttk.Entry(frame_dim, width=8)
        self.entry_rows.grid(row=0, column=1, sticky="w", padx=6, pady=2)
        self.entry_cols.grid(row=1, column=1, sticky="w", padx=6, pady=2)
        self.entry_rows.insert(0, "3")
        self.entry_cols.insert(0, "3")
        ToolTip(self.entry_rows, "Número de vectores (filas). Máximo recomendado: 12.")
        ToolTip(self.entry_cols, "Dimensión de cada vector (columnas).")

        ttk.Separator(parent, orient="horizontal").pack(fill="x", pady=8)

        # Botones de acciones
        ttk.Button(parent, text="➕ Crear matriz manual", command=self.open_manual_matrix).pack(fill="x", pady=6)
        ttk.Button(parent, text="🎲 Generar aleatoria", command=self.generate_random).pack(fill="x", pady=6)
        ttk.Button(parent, text="📁 Cargar desde archivo", command=self.load_matrix_from_file).pack(fill="x", pady=6)
        ttk.Button(parent, text="💾 Guardar matriz (txt)", command=self.save_matrix_to_file).pack(fill="x", pady=6)

        ttk.Separator(parent, orient="horizontal").pack(fill="x", pady=8)
        ttk.Label(parent, text="Ejemplos rápidos", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(4, 6))

        # ejemplos 2x2 a 12x12
        values = [f"{i}x{i}" for i in range(2, 13)]
        self.combo_examples = ttk.Combobox(parent, values=values, state="readonly")
        self.combo_examples.set("Seleccionar tamaño")
        self.combo_examples.pack(fill="x", pady=4)
        ttk.Button(parent, text="Crear ejemplo", command=self.create_example).pack(fill="x", pady=6)

        ttk.Separator(parent, orient="horizontal").pack(fill="x", pady=8)
        ttk.Label(parent, text="Exportar / Imprimir", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(4, 6))
        ttk.Button(parent, text="📷 Exportar gráfico (PNG)", command=self.export_png).pack(fill="x", pady=6)

    # -----------------------------
    # RIGHT PANEL: controles de animación y resultados
    # -----------------------------
    def _populate_rightpanel(self, parent):
        ttk.Label(parent, text="Controles", font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=(0, 6))

        # Animación
        frame_anim = ttk.Frame(parent)
        frame_anim.pack(fill="x", pady=6)
        self.btn_play = ttk.Button(frame_anim, text="▶ Play", width=9, command=self.toggle_animation)
        self.btn_step = ttk.Button(frame_anim, text="⮞ Step", width=9, command=self.step_animation)
        self.btn_reset = ttk.Button(frame_anim, text="↺ Reset", width=9, command=self.reset_animation)
        self.btn_play.grid(row=0, column=0, padx=4)
        self.btn_step.grid(row=0, column=1, padx=4)
        self.btn_reset.grid(row=0, column=2, padx=4)
        ToolTip(self.btn_play, "Inicia / pausa la animación paso a paso.")
        ToolTip(self.btn_step, "Avanza un paso (muestra el siguiente vector ortonormal).")
        ToolTip(self.btn_reset, "Restablece la animación al inicio.")

        # Velocidad
        ttk.Label(parent, text="Velocidad (ms):").pack(anchor="w", pady=(8, 2))
        self.scale_speed = ttk.Scale(parent, from_=100, to=1500, orient="horizontal", command=self._on_speed_change)
        self.scale_speed.set(self.anim_speed_ms)
        self.scale_speed.pack(fill="x")
        ToolTip(self.scale_speed, "Ajusta el intervalo entre pasos en milisegundos.")

        ttk.Separator(parent, orient="horizontal").pack(fill="x", pady=10)

        # Resultados numéricos
        ttk.Label(parent, text="Resultados", font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=(0, 6))
        self.text_results = tk.Text(parent, height=12, wrap="word", font=("Consolas", 9))
        self.text_results.pack(fill="both", expand=True)
        self.text_results.insert("1.0", "Aquí aparecerán los vectores ortonormales y notas.\n")
        self.text_results.config(state="disabled")

    # -----------------------------
    # PLOT: matplotlib + canvas
    # -----------------------------
    def _create_plot(self):
        self.fig = plt.Figure(figsize=(8, 6), facecolor="#FFFFFF")
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.ax.set_title("Gram-Schmidt: Originales (alpha) vs Ortogonales (beta)")
        self.ax.grid(True, linestyle="--", alpha=0.4)
        self.ax.set_box_aspect([1, 1, 1])

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.center)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill="both", expand=True)
        # Barra de navegación de matplotlib
        toolbar = NavigationToolbar2Tk(self.canvas, self.center)
        toolbar.update()
        self.canvas._tkcanvas.pack(fill="x")

    # -----------------------------
    # SHORTCUTS
    # -----------------------------
    def _bind_shortcuts(self):
        self.root.bind("<Control-o>", lambda e: self.load_matrix_from_file())
        self.root.bind("<Control-s>", lambda e: self.save_matrix_to_file())

    # -----------------------------
    # ACCIONES: crear / cargar / guardar matrices
    # -----------------------------
    def open_manual_matrix(self):
        """Abre una ventana para introducir valores manualmente"""
        try:
            rows = int(self.entry_rows.get())
            cols = int(self.entry_cols.get())
            if rows < 1 or cols < 1 or rows > 50 or cols > 50:
                raise ValueError
        except ValueError:
            messagebox.showerror("Dimensiones inválidas", "Introduce filas y columnas válidas (1-50).")
            return

        win = tk.Toplevel(self.root)
        win.title("Ingresar matriz manual")
        win.geometry("520x420")
        frame = ttk.Frame(win, padding=8)
        frame.pack(fill="both", expand=True)

        label = ttk.Label(frame, text=f"Ingrese {rows} vectores (filas) de dimensión {cols}:",
                          font=("Segoe UI", 10, "bold"))
        label.pack(anchor="w", pady=(0, 6))

        canvas = tk.Canvas(frame)
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)
        inner = ttk.Frame(canvas)
        inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        self.manual_entries = []
        for i in range(rows):
            row_frame = ttk.Frame(inner)
            row_frame.pack(anchor="w", pady=2, padx=4)
            ttk.Label(row_frame, text=f"v{i+1}:", width=4).pack(side="left")
            entries = []
            for j in range(cols):
                e = ttk.Entry(row_frame, width=6, justify="center")
                e.pack(side="left", padx=2)
                e.insert(0, "0" if j else str(np.random.randint(-3, 4)))
                entries.append(e)
            self.manual_entries.append(entries)

        def guardar_manual():
            try:
                M = []
                for r in self.manual_entries:
                    row = []
                    for e in r:
                        val = safe_float(e.get())
                        if val is None:
                            raise ValueError("Entrada no numérica")
                        row.append(val)
                    M.append(row)
                self.matriz = np.array(M, dtype=float)
                win.destroy()
                self.on_matrix_changed()
                messagebox.showinfo("Matriz guardada", "Matriz creada y guardada en memoria.")
            except Exception as ex:
                messagebox.showerror("Error", f"Revisa las entradas: {ex}")

        ttk.Button(frame, text="Guardar matriz", command=guardar_manual).pack(pady=8)

    def generate_random(self):
        try:
            rows = int(self.entry_rows.get())
            cols = int(self.entry_cols.get())
            if rows < 1 or cols < 1:
                raise ValueError
        except ValueError:
            messagebox.showerror("Error", "Filas y columnas deben ser números enteros válidos.")
            return
        self.matriz = np.random.randint(-6, 7, size=(rows, cols)).astype(float)
        self.on_matrix_changed()
        messagebox.showinfo("Generada", f"Matriz aleatoria {rows}×{cols} creada.")

    def create_example(self):
        sel = self.combo_examples.get()
        if "x" not in sel:
            messagebox.showwarning("Selecciona", "Elige un tamaño de ejemplo del desplegable.")
            return
        n = int(sel.split("x")[0])
        self.matriz = np.random.randint(-4, 5, size=(n, n)).astype(float)
        self.on_matrix_changed()
        messagebox.showinfo("Ejemplo", f"Matriz ejemplo {n}×{n} creada.")

    def save_matrix_to_file(self):
        if self.matriz is None:
            messagebox.showwarning("Nada que guardar", "No hay matriz en memoria.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text files","*.txt"), ("All files","*.*")])
        if not path:
            return
        try:
            np.savetxt(path, self.matriz, fmt="%.8f")
            messagebox.showinfo("Guardado", f"Matriz guardada en:\n{path}")
        except Exception as ex:
            messagebox.showerror("Error al guardar", str(ex))

    def load_matrix_from_file(self):
        path = filedialog.askopenfilename(filetypes=[("Text files","*.txt"), ("CSV files","*.csv"), ("All files","*.*")])
        if not path:
            return
        try:
            M = np.loadtxt(path)
            # If single row -> keep 2D
            if M.ndim == 1:
                M = M.reshape(1, -1)
            self.matriz = np.array(M, dtype=float)
            self.on_matrix_changed()
            messagebox.showinfo("Cargada", f"Matriz cargada desde:\n{path}")
        except Exception as ex:
            messagebox.showerror("Error al cargar", f"No se pudo leer el archivo:\n{ex}")

    # -----------------------------
    # CUANDO LA MATRIZ CAMBIA
    # -----------------------------
    def on_matrix_changed(self):
        # actualizar entradas de filas/cols
        if self.matriz is not None:
            self.entry_rows.delete(0, "end")
            self.entry_rows.insert(0, str(self.matriz.shape[0]))
            self.entry_cols.delete(0, "end")
            self.entry_cols.insert(0, str(self.matriz.shape[1]))
            self.set_status(f"Matriz cargada: {self.matriz.shape[0]} vectores × {self.matriz.shape[1]} dimensiones.")
        else:
            self.set_status("No hay matriz en memoria.")

        # Recalcular ortonormales
        try:
            self.ortonormales = modified_gram_schmidt(self.matriz) if self.matriz is not None else None
        except Exception as ex:
            self.ortonormales = None
            self.set_status(f"Error al calcular Gram-Schmidt: {ex}")

        # Reset anim
        self.reset_animation()
        # Mostrar en plot y en texto
        self._draw_static_plot()
        self._update_results_text()

    # -----------------------------
    # DIBUJAR: visualización estática
    # -----------------------------
    def _draw_static_plot(self):
        self.ax.clear()

        if self.matriz is None:
            self.ax.set_title("Sin matriz - crea o carga una matriz")
            self.canvas.draw()
            return

        # Proyectar
        try:
            M3 = project_to_3d(self.matriz)
        except Exception:
            M3 = np.zeros((self.matriz.shape[0], 3))
        if self.ortonormales is not None and len(self.ortonormales) > 0:
            try:
                Q3 = project_to_3d(self.ortonormales)
            except Exception:
                Q3 = np.zeros((self.ortonormales.shape[0], 3))
        else:
            Q3 = np.zeros((0, 3))

        # Escalado automatico para límites
        all_pts = np.vstack([M3, Q3]) if Q3.size else M3
        max_range = np.max(np.abs(all_pts)) if all_pts.size else 1.0
        if max_range == 0:
            max_range = 1.0
        lim = max_range * 1.2
        self.ax.set_xlim(-lim, lim)
        self.ax.set_ylim(-lim, lim)
        self.ax.set_zlim(-lim, lim)

        origen = np.zeros(3)
        # Colores
        cmap = plt.cm.get_cmap("tab10")
        m = M3.shape[0]
        k = Q3.shape[0]

        # Dibujar vectores originales (translúcidos)
        for i, v in enumerate(M3):
            c = cmap(i % 10)
            self.ax.quiver(*origen, *v, color=c, alpha=0.35, linewidth=2, length=1.0, normalize=False)
            # etiqueta pegada al final
            self.ax.text(*(v * 1.05), f"o{i+1}", fontsize=9)

        # Dibujar ortonormales (más visibles)
        for i, v in enumerate(Q3):
            c = cmap((i + 3) % 10)
            self.ax.quiver(*origen, *v, color=c, alpha=1.0, linewidth=3, length=1.0, normalize=False)
            self.ax.text(*(v * 1.05), f"q{i+1}", fontsize=9, fontweight="bold")

        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")
        self.ax.set_title("Gram-Schmidt: originales (alpha) vs ortonormales (beta)")
        self.canvas.draw()

    # -----------------------------
    # ANIMACIÓN: step / play / pause / reset
    # -----------------------------
    def toggle_animation(self):
        if self.matriz is None:
            messagebox.showwarning("Sin matriz", "Crea o carga una matriz primero.")
            return
        if self.ortonormales is None or len(self.ortonormales) == 0:
            messagebox.showinfo("Sin vectores ortonormales", "Gram-Schmidt no produjo vectores ortonormales.")
            return
        self.anim_running = not self.anim_running
        self.btn_play.config(text="⏸ Pause" if self.anim_running else "▶ Play")
        if self.anim_running:
            self._animate_step()

    def _animate_step(self):
        if not self.anim_running:
            return
        # mostrar hasta anim_index (exclusivo)
        self.ax.clear()
        M3 = project_to_3d(self.matriz)
        Q3_full = project_to_3d(self.ortonormales)
        # limites como antes
        all_pts = np.vstack([M3, Q3_full]) if Q3_full.size else M3
        max_range = np.max(np.abs(all_pts)) if all_pts.size else 1.0
        lim = (max_range if max_range != 0 else 1.0) * 1.2
        self.ax.set_xlim(-lim, lim); self.ax.set_ylim(-lim, lim); self.ax.set_zlim(-lim, lim)
        origen = np.zeros(3)
        cmap = plt.cm.get_cmap("tab10")

        # dibujar todos los originales
        for i, v in enumerate(M3):
            c = cmap(i % 10)
            self.ax.quiver(*origen, *v, color=c, alpha=0.28, linewidth=2, length=1.0, normalize=False)
            self.ax.text(*(v * 1.05), f"o{i+1}", fontsize=9)

        # dibujar ortonormales hasta anim_index
        upto = min(self.anim_index + 1, Q3_full.shape[0])
        for i in range(upto):
            v = Q3_full[i]
            c = cmap((i + 3) % 10)
            self.ax.quiver(*origen, *v, color=c, alpha=1.0, linewidth=3, length=1.0, normalize=False)
            self.ax.text(*(v * 1.05), f"q{i+1}", fontsize=10, fontweight="bold")

        self.ax.set_title(f"Animación: mostrando {upto} / {Q3_full.shape[0]} vectores ortonormales")
        self.canvas.draw()

        # avanzar índice
        if self.anim_index < Q3_full.shape[0] - 1:
            self.anim_index += 1
        else:
            # Si llegó al final, pausar automáticamente
            self.anim_running = False
            self.btn_play.config(text="▶ Play")
            self.set_status("Animación completada.")
            return

        # programar siguiente paso
        self.after_id = self.root.after(int(self.anim_speed_ms), self._animate_step)

    def step_animation(self):
        if self.matriz is None:
            messagebox.showwarning("Sin matriz", "Crea o carga una matriz primero.")
            return
        if self.ortonormales is None or len(self.ortonormales) == 0:
            messagebox.showinfo("Sin vectores ortonormales", "Gram-Schmidt no produjo vectores ortonormales.")
            return
        # Si está corriendo, hacemos pause
        if self.anim_running:
            self.anim_running = False
            if self.after_id:
                self.root.after_cancel(self.after_id)
            self.btn_play.config(text="▶ Play")
        # mostrar siguiente paso no automático
        Q3 = project_to_3d(self.ortonormales)
        if self.anim_index < Q3.shape[0] - 1:
            self.anim_index += 1
        self._animate_once()

    def _animate_once(self):
        # Similar a _animate_step pero sin programar siguiente llamado
        self.ax.clear()
        M3 = project_to_3d(self.matriz)
        Q3_full = project_to_3d(self.ortonormales)
        all_pts = np.vstack([M3, Q3_full]) if Q3_full.size else M3
        max_range = np.max(np.abs(all_pts)) if all_pts.size else 1.0
        lim = (max_range if max_range != 0 else 1.0) * 1.2
        self.ax.set_xlim(-lim, lim); self.ax.set_ylim(-lim, lim); self.ax.set_zlim(-lim, lim)
        origen = np.zeros(3)
        cmap = plt.cm.get_cmap("tab10")

        for i, v in enumerate(M3):
            c = cmap(i % 10)
            self.ax.quiver(*origen, *v, color=c, alpha=0.28, linewidth=2, length=1.0, normalize=False)
            self.ax.text(*(v * 1.05), f"o{i+1}", fontsize=9)

        upto = min(self.anim_index + 1, Q3_full.shape[0])
        for i in range(upto):
            v = Q3_full[i]
            c = cmap((i + 3) % 10)
            self.ax.quiver(*origen, *v, color=c, alpha=1.0, linewidth=3, length=1.0, normalize=False)
            self.ax.text(*(v * 1.05), f"q{i+1}", fontsize=10, fontweight="bold")

        self.ax.set_title(f"Animación: mostrando {upto} / {Q3_full.shape[0]} vectores ortonormales")
        self.canvas.draw()
        self.set_status(f"Paso {upto} / {Q3_full.shape[0]} mostrado.")

    def reset_animation(self):
        # cancelar after si existe
        if self.after_id:
            try:
                self.root.after_cancel(self.after_id)
            except Exception:
                pass
            self.after_id = None
        self.anim_running = False
        self.anim_index = 0
        self.btn_play.config(text="▶ Play")
        self._draw_static_plot()
        self.set_status("Animación reiniciada.")

    def _on_speed_change(self, val):
        try:
            v = int(float(val))
            self.anim_speed_ms = v
            self.set_status(f"Velocidad ajustada: {v} ms por paso.")
        except Exception:
            pass

    # -----------------------------
    # RESULTADOS: mostrar vectores Q en el panel derecho
    # -----------------------------
    def _update_results_text(self):
        self.text_results.config(state="normal")
        self.text_results.delete("1.0", "end")
        if self.ortonormales is None or len(self.ortonormales) == 0:
            self.text_results.insert("1.0", "Gram–Schmidt no produjo vectores ortonormales (vectores dependientes o matriz vacía).\n")
        else:
            self.text_results.insert("1.0", f"Vectores ortonormales (Q) — total: {len(self.ortonormales)}\n\n")
            for i, q in enumerate(self.ortonormales):
                q_str = ", ".join(f"{x:.6f}" for x in q)
                self.text_results.insert("end", f"q{i+1} = [{q_str}]\n")
        self.text_results.config(state="disabled")

    # -----------------------------
    # EXPORT PNG
    # -----------------------------
    def export_png(self):
        if self.matriz is None:
            messagebox.showwarning("Sin matriz", "Crea o carga una matriz antes de exportar.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG image","*.png")])
        if not path:
            return
        try:
            # Aseguramos que el plot esté actualizado
            self.canvas.draw()
            self.fig.savefig(path, dpi=200, bbox_inches="tight")
            messagebox.showinfo("Exportado", f"Gráfico exportado a:\n{path}")
        except Exception as ex:
            messagebox.showerror("Error exportando", str(ex))

    # -----------------------------
    # STATUS
    # -----------------------------
    def set_status(self, text):
        self.status_label.config(text=text)

    # -----------------------------
    # MAINLOOP helpers
    # -----------------------------
    def run(self):
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.mainloop()

    def _on_close(self):
        if self.anim_running and messagebox.askyesno("Salir", "La animación está en curso. ¿Deseas salir igualmente?"):
            self.root.destroy()
        else:
            self.root.destroy()


# -----------------------------
# EJECUCIÓN
# -----------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = GramSchmidtApp(root)
    app.run()
 