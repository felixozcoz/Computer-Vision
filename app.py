import cv2
import tkinter as tk
from tkinter import messagebox, ttk, filedialog
from PIL import Image, ImageTk

class SimpleImageApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Simple Image App")

        # Variables
        self.cap = cv2.VideoCapture(0)  # Captura de cámara
        self.current_frame = None
        self.filtered_frame = None

        # Widgets
        self.canvas = tk.Canvas(root, width=400, height=400)
        self.canvas.pack()

        self.filters = {
            "Original": lambda img: img,
            "Gray": lambda img: cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
            "Blur": lambda img: cv2.GaussianBlur(img, (15, 15), 0),
            "Canny Edge": lambda img: cv2.Canny(img, 100, 200),
        }

        self.filter_var = tk.StringVar()
        self.filter_var.set("Original")

        self.filter_menu = ttk.Combobox(root, textvariable=self.filter_var, values=list(self.filters.keys()))
        self.filter_menu.pack()

        self.btn_open_camera = tk.Button(root, text="Open Camera", command=self.open_camera)
        self.btn_open_camera.pack()

        self.btn_capture = tk.Button(root, text="Capture & Save", command=self.capture_and_save)
        self.btn_capture.pack()

        self.btn_close = tk.Button(root, text="Close Camera", command=self.close_camera)
        self.btn_close.pack()

        # Iniciar el proceso de visualización de la cámara
        self.show_camera_feed()

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
                self.filtered_frame = filter_function(frame)
                self.current_frame = cv2.cvtColor(self.filtered_frame, cv2.COLOR_BGR2RGB)
            else:
                self.current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            self.display_image(self.current_frame)

        self.root.after(10, self.show_camera_feed)  # Actualizar cada 10 ms

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


if __name__ == "__main__":
    root = tk.Tk()
    app = SimpleImageApp(root)
    root.mainloop()
