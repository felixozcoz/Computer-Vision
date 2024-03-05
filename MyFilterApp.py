# ---------------------------------------------
# Fichero: MyFilterApp.py
# ---------------------------------------------
# Escuela de Ingeniería y Arquitectura de Zaragoza
# Visión por Computador	
# 2023- 2024
#
# Félix Ozcoz Eraso             801108
# Victor Marcuello Baquero      741278
#
# Descripción:
#  Módulo que contiene la implmentación de la interfaz gráfica
#  de la aplicación de filtros de imagen 
# ---------------------------------------------

import cv2
import tkinter as tk
from tkinter import messagebox, ttk, filedialog, colorchooser
from PIL import ImageColor, ImageTk, Image
import filters

class SimpleImageApp:
    def __init__(self, root):
        self.root = root
        self.root.title("MyFilter App")

        # ------------------- Attributes -------------------

        self.cap = cv2.VideoCapture(0) 
        self.current_frame = None
        self.filtered_frame = None

        self.filter_var = tk.StringVar(value = "Original")

        # Variables for the contrast filter
        self.alpha_var = tk.DoubleVar(value=1.0)
        self.beta_var = tk.DoubleVar(value=0.0)

        # Variables for the geometric distortion filter
        self.k1_var = tk.DoubleVar(value=0.0)
        self.k2_var = tk.DoubleVar(value=0.0)

        # Filters
        self.filters = {
            "Original": lambda img: img,
            "Contrast": lambda img, alpha, beta: cv2.convertScaleAbs(img, alpha=alpha, beta=beta),
            "Posterization": lambda img: filters.posterization_filter(img),
            "Alien": lambda img, color: filters.alien_filter(img, color),
            "Geometric distortion": lambda img, k1, k2: filters.geometric_distortion(img, k1, k2),
        }

        # ------------------- Widgets -------------------

        # Canvas to display the camera feed
        self.canvas = tk.Canvas(root, width=400, height=400)
        self.canvas.pack()
        
        # Menu to select the filter
        self.filter_menu = ttk.Combobox(root, textvariable=self.filter_var, values=list(self.filters.keys()), state="readonly", width=20)
        self.filter_menu.pack()

        # Slides controls for the parameters of the contrast filter
        self.alpha_slider = tk.Scale(root, label="Alpha", from_=0.1, to=3.0, resolution=0.1, variable=self.alpha_var, orient=tk.HORIZONTAL)
        self.beta_slider = tk.Scale(root, label="Beta", from_=-100, to=100, resolution=1, variable=self.beta_var, orient=tk.HORIZONTAL)

        # Slides controls for the parameters of the geometric distortion filter
        self.k1_slider = tk.Scale(root, label="k1", from_=-0.00001, to=0.00001, resolution=0.00000001, variable=self.k1_var, orient=tk.HORIZONTAL)
        self.k2_slider = tk.Scale(root, label="k2", from_=-0.00001, to=0.00001, resolution=0.00000001, variable=self.k2_var, orient=tk.HORIZONTAL)

        # Slider control for the color of the alien filter
        self.color_button = tk.Button(root, text="Select Color", command=self.select_color)
        self.color_alien_var = tk.StringVar(value="#dc6464")

        # Capture and save button
        self.btn_capture = tk.Button(root, text="Capture & Save", command=self.capture_and_save)
        self.btn_capture.pack()

        # Initialize the camera
        self.show_camera_feed()

        # Assign the event to the filter change
        self.filter_var.trace("w", self.update_parameters_ui)

    # ------------------- Methods -------------------

    def open_camera(self):
        self.cap = cv2.VideoCapture(0)
        self.show_camera_feed()

    def close_camera(self):
        self.cap.release()
        self.canvas.delete("all")

    def show_camera_feed(self):
        '''
            Show the camera feed and apply the selected filter
        '''
        ret, frame = self.cap.read()
        if ret:
            selected_filter = self.filter_var.get()
            if selected_filter in self.filters:
                filter_function = self.filters[selected_filter]

                if selected_filter == "Contrast":
                    alpha = self.alpha_var.get()
                    beta = self.beta_var.get()
                    self.filtered_frame = filter_function(frame, alpha, beta)
                    self.update_parameters_ui()  

                elif selected_filter == "Alien":
                    color = ImageColor.getrgb( self.color_alien_var.get() ) # extract selected color
                    self.filtered_frame = filter_function(frame, color)
                    self.update_parameters_ui()  

                elif selected_filter == "Geometric distortion":
                    k1 = self.k1_var.get()
                    k2 = self.k2_var.get()
                    self.filtered_frame = filter_function(frame, k1, k2)
                    self.update_parameters_ui()

                else:
                    self.filtered_frame = filter_function(frame)
                    self.hide_parameters_ui()  # hide sliders
                
                self.current_frame = cv2.cvtColor(self.filtered_frame, cv2.COLOR_BGR2RGB)
            else:
                self.current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            self.display_image(self.current_frame)

        self.root.after(10, self.show_camera_feed)  # update the camera feed every 10 ms


    def update_parameters_ui(self, *args):
        self.show_parameters_ui(self.filter_var.get())


    def show_parameters_ui(self, selected_filter):
        '''
            Show the sliders for the selected filter
        '''
        if selected_filter == "Contrast":
            self.alpha_slider.pack()
            self.beta_slider.pack()

        elif selected_filter == "Geometric distortion":
            self.k1_slider.pack()
            self.k2_slider.pack()

        elif selected_filter == "Alien":
            self.color_button.pack()
            

    def hide_parameters_ui(self):
        '''
            Hide the sliders for the unselected filter and set the values to 0
        '''
        selected_filter = self.show_parameters_ui(self.filter_var.get())
        if selected_filter != "Contrast":
            # hide sliders for contrast filter and set the values to 0
            self.alpha_var.set(0.0)
            self.beta_var.set(0.0)
            self.alpha_slider.pack_forget()
            self.beta_slider.pack_forget()
        
        if selected_filter != "Geometric distortion":
            # hide sliders for geometric distortion filter and set the values to 0
            self.k1_var.set(0.0)
            self.k2_var.set(0.0)
            self.k1_slider.pack_forget()
            self.k2_slider.pack_forget()

        if selected_filter != "Alien":
            # hide color button and set the value to the default color value
            self.color_button.pack_forget()


    def capture_and_save(self):
        '''
            Capture the current frame and save it to a file
        '''
        if self.filtered_frame is None:
            messagebox.showerror("Error", "No filtered frame!")
            return

        file_path = tk.filedialog.asksaveasfilename(defaultextension=".jpg",
                                                    filetypes=[("JPEG files", "*.jpg"), ("PNG files", "*.png")])
        if file_path:
            cv2.imwrite(file_path, self.filtered_frame)
            messagebox.showinfo("Success", "Image saved successfully!")


    def display_image(self, image):
        '''
            Display the image in the canvas
        '''
        image = Image.fromarray(image)
        image = ImageTk.PhotoImage(image.resize((400, 400)))
        self.canvas.image = image
        self.canvas.create_image(0, 0, anchor=tk.NW, image=image)


    def select_color(self):
        '''
            Open a color picker and set the color of the alien filter
        '''
        self.color_alien_var.set( colorchooser.askcolor(title="Select Color")[1] )





# ------------------- Main -------------------


if __name__ == "__main__":
    root = tk.Tk()
    app = SimpleImageApp(root)
    root.mainloop()
