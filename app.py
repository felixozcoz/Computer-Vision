import cv2
import tkinter as tk
from tkinter import messagebox, ttk, filedialog
from PIL import Image, ImageTk
import filters

class SimpleImageApp:
    def __init__(self, root):
        self.root = root
        self.root.title("MyFilter App")

        # Variables
        self.cap = cv2.VideoCapture(0)  # Captura de cámara
        self.current_frame = None
        self.filtered_frame = None

        # Widgets
        self.canvas = tk.Canvas(root, width=400, height=400)
        self.canvas.pack()

        self.filters = {
            "Original": lambda img: img,
            "Contrast": lambda img, alpha, beta: cv2.convertScaleAbs(img, alpha=alpha, beta=beta),
            "Posterization": lambda img: filters.posterization_filter(img),
            "Alien": lambda img, color: filters.alien_filter(img, color),
            "Geometric distortion": lambda img, k1, k2: filters.geometric_distortion(img, k1, k2),
        }

        self.filter_var = tk.StringVar()
        self.filter_var.set("Original")

        self.filter_menu = ttk.Combobox(root, textvariable=self.filter_var, values=list(self.filters.keys()), state="readonly", width=20)
        self.filter_menu.pack()

        # Controles deslizantes para los parámetros del filtro de contraste
        self.alpha_var = tk.DoubleVar(value=1.0)
        self.beta_var = tk.DoubleVar(value=0.0)
        self.alpha_slider = tk.Scale(root, label="Alpha", from_=0.1, to=3.0, resolution=0.1, variable=self.alpha_var, orient=tk.HORIZONTAL)
        self.beta_slider = tk.Scale(root, label="Beta", from_=-100, to=100, resolution=1, variable=self.beta_var, orient=tk.HORIZONTAL)

        # Controles deslizantes para los parámetros del filtro de distorsión geométrica
        self.k1_var = tk.DoubleVar(value=0.0)
        self.k2_var = tk.DoubleVar(value=0.0)
        self.k1_slider = tk.Scale(root, label="k1", from_=-0.00001, to=0.00001, resolution=0.00000001, variable=self.k1_var, orient=tk.HORIZONTAL)
        self.k2_slider = tk.Scale(root, label="k2", from_=-0.00001, to=0.00001, resolution=0.00000001, variable=self.k2_var, orient=tk.HORIZONTAL)

        # Controles para el parámetro de color en el filtro de Alien

        self.btn_capture = tk.Button(root, text="Capture & Save", command=self.capture_and_save)
        self.btn_capture.pack()

        # Iniciar el proceso de visualización de la cámara
        self.show_camera_feed()

        # Asignar evento de cambio de filtro
        self.filter_var.trace("w", self.update_parameters_ui)

    def open_camera(self):
        self.cap = cv2.VideoCapture(0)
        self.show_camera_feed()

    def close_camera(self):
        self.cap.release()
        self.canvas.delete("all")

    def show_camera_feed(self):
        ret, frame = self.cap.read()
        if ret:
            selected_filter = self.filter_var.get()
            if selected_filter in self.filters:
                filter_function = self.filters[selected_filter]

                if selected_filter == "Contrast":
                    alpha = self.alpha_var.get()
                    beta = self.beta_var.get()
                    self.filtered_frame = filter_function(frame, alpha, beta)
                    self.update_parameters_ui()  # Mostrar controles deslizantes       
                elif selected_filter == "Alien":
                    color = (255, 0, 0)  # Color predeterminado para el filtro Alien
                    self.filtered_frame = filter_function(frame, color)
                    self.update_parameters_ui()  # Ocultar controles deslizantes
                elif selected_filter == "Geometric distortion":
                    k1 = self.k1_var.get()
                    k2 = self.k2_var.get()
                    self.filtered_frame = filter_function(frame, k1, k2)
                    self.update_parameters_ui()
                else:
                    self.filtered_frame = filter_function(frame)
                    self.hide_parameters_ui()  # Ocultar controles deslizantes
                
                self.current_frame = cv2.cvtColor(self.filtered_frame, cv2.COLOR_BGR2RGB)
            else:
                self.current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            self.display_image(self.current_frame)

        self.root.after(10, self.show_camera_feed)  # Actualizar cada 10 ms

    def update_parameters_ui(self, *args):
        self.show_parameters_ui(self.filter_var.get())

    def show_parameters_ui(self, selected_filter):
        self.hide_parameters_ui()
        if selected_filter == "Contrast":
            self.alpha_slider.pack()
            self.beta_slider.pack()
        elif selected_filter == "Geometric distortion":
            self.k1_slider.pack()
            self.k2_slider.pack()
        elif selected_filter == "Alien":
            

    def hide_parameters_ui(self):
        # ocultar controles deslizantes para el filtro de contraste
        self.alpha_slider.pack_forget()
        self.beta_slider.pack_forget()
        self.alpha_var.set(0.0)
        self.beta_var.set(0.0)
        # ocultar controles deslizantes para el filtro de distorsión geométrica
        self.k1_slider.pack_forget()
        self.k2_slider.pack_forget()
        self.k1_var.set(0.0)
        self.k2_var.set(0.0)

    def capture_and_save(self):
        if self.filtered_frame is None:
            messagebox.showerror("Error", "No filtered frame!")
            return

        file_path = tk.filedialog.asksaveasfilename(defaultextension=".jpg",
                                                    filetypes=[("JPEG files", "*.jpg"), ("PNG files", "*.png")])
        if file_path:
            cv2.imwrite(file_path, self.filtered_frame)
            messagebox.showinfo("Success", "Image saved successfully!")

    def display_image(self, image):
        image = Image.fromarray(image)
        image = ImageTk.PhotoImage(image.resize((400, 400)))
        self.canvas.image = image
        self.canvas.create_image(0, 0, anchor=tk.NW, image=image)

    def seleccionar_color():
        color = tk.askcolor(title="Seleccionar Color")
        if color:
            return color[1]


if __name__ == "__main__":
    root = tk.Tk()
    app = SimpleImageApp(root)
    root.mainloop()
