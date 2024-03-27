import tkinter as tk
from tkinter import filedialog, messagebox, Scale
from PIL import Image, ImageTk
import HoughTransform as ht  # Módulo con la función vanishing_point_detector
import cv2

class ImageProcessor:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Operaciones de Imagen")

        # Obtener las dimensiones de la ventana
        self.window_width = self.root.winfo_screenwidth() // 2
        self.window_height = self.root.winfo_screenheight() // 2

        # Crear etiquetas y botones
        self.btn_open = tk.Button(self.root, text="Abrir Imagen", command=self.open_image)
        self.btn_open.pack(pady=10)

        self.btn_apply = tk.Button(self.root, text="Buscar punto de fuga", command=self.apply_operation)
        self.btn_apply.pack(pady=5)

        # Slider para el umbral de binarización (inicialmente deshabilitado)
        self.threshold_slider = Scale(self.root, from_=0, to=255, orient="horizontal", label="Threshold", state=tk.DISABLED)
        self.threshold_slider.pack()

        self.img_label = tk.Label(self.root)
        self.img_label.pack()

        self.img = None
        self.original_img = None
        self.path = None  # Ruta de la imagen abierta

    def open_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.original_img = Image.open(file_path)
            self.path = file_path

            # Redimensionar la imagen si es necesario para que quepa en la ventana
            self.resize_image()

            self.img = self.original_img.copy()  # Crear una copia de la imagen original

            # Mostrar la imagen en el lienzo
            img_tk = ImageTk.PhotoImage(self.img)
            self.img_label.config(image=img_tk)
            self.img_label.image = img_tk

            # Habilitar el slider después de abrir la imagen
            self.threshold_slider.config(state=tk.NORMAL)
            self.threshold_slider.set(127)  # Establecer un valor inicial al abrir una nueva imagen

    def resize_image(self):
        # Redimensionar la imagen original si es necesario para que quepa en la ventana
        if self.original_img.width > self.window_width or self.original_img.height > self.window_height:
            self.original_img.thumbnail((self.window_width, self.window_height), Image.LANCZOS)
        if self.img is None:
            return
        if self.img.width > self.window_width or self.img.height > self.window_height:
            self.img.thumbnail((self.window_width, self.window_height), Image.LANCZOS)

    def apply_operation(self):
        if self.img is None:
            return

        # Obtener el valor del slider (umbral de binarización)
        threshold = self.threshold_slider.get()

        try:
            # Llamar a la función para detectar el punto de fuga
            img_cv2 = ht.vanishing_point_detector(self.path, threshold)
        except Exception as e:
            messagebox.showerror("Error", f"Error al procesar la imagen: {e}")
            return

        # Convertir la imagen procesada de OpenCV a PIL
        self.img = Image.fromarray(cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB))

        # Redimensionar la imagen procesada si es necesario para que quepa en la ventana
        self.resize_image()

        # Mostrar la imagen procesada en el lienzo
        img_tk = ImageTk.PhotoImage(self.img)
        self.img_label.config(image=img_tk)
        self.img_label.image = img_tk

    def save_image(self):
        if self.img is None:
            return

        file_path = filedialog.asksaveasfilename(defaultextension=".jpg",
                                                 filetypes=[("JPEG files", "*.jpg"), ("All files", "*.*")])
        if file_path:
            try:
                self.img.save(file_path)
                messagebox.showinfo("Guardado", "Imagen guardada exitosamente.")
            except Exception as e:
                messagebox.showerror("Error al guardar", f"No se pudo guardar la imagen: {e}")

    def run(self):
        # Crear un menú con la opción de guardar
        menubar = tk.Menu(self.root)
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Guardar Imagen", command=self.save_image)
        menubar.add_cascade(label="Archivo", menu=file_menu)
        self.root.config(menu=menubar)

        # Mantener la ventana abierta
        self.root.mainloop()

# Si este archivo se ejecuta directamente, se ejecuta la aplicación
if __name__ == "__main__":
    processor = ImageProcessor()
    processor.run()
