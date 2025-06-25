import tkinter as tk
from tkinter import ttk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageChops, ImageFilter
import numpy as np
import cv2

class ImageEditorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image processing (Optimized)")
        self.root.geometry("1700x950")
        self.root.configure(bg="#f0f0f0")
        self.original_image = None
        self.second_image = None
        self.edited_image = None
        self.btn_color = "#4a90e2"
        self.btn_fg = "white"
        self.font = ("Helvetica", 10, "bold")
        self.img_size = (400, 400)
        self.controls_frame_left = tk.LabelFrame(root, text="Image & Basic Ops", font=("Helvetica", 11, "bold"), padx=10, pady=10, bg="white", fg="#333", relief=tk.RIDGE, bd=1)
        self.controls_frame_left.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        self.controls_frame_right = tk.Frame(root, bg="white", relief=tk.RIDGE, bd=1)
        self.controls_frame_right.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)
        self.images_frame = tk.Frame(root, bg="#f0f0f0")
        self.images_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=10)
        self.create_widgets()
        self.create_placeholder()

    def create_widgets(self):
        def styled_btn(parent, text, command):
            return tk.Button(parent, text=text, command=command, bg=self.btn_color, fg=self.btn_fg, font=self.font, relief="raised", width=23, height=2, bd=0, padx=5)

        file_ops_frame = tk.LabelFrame(self.controls_frame_left, text="File", font=self.font, bg="white", fg="#333", padx=5, pady=5)
        file_ops_frame.pack(pady=10, fill=tk.X)
        styled_btn(file_ops_frame, "Upload Image", self.load_image).pack(pady=5)
        styled_btn(file_ops_frame, "Load Second Image", self.load_second_image).pack(pady=5)
        styled_btn(file_ops_frame, "Save Edited Image", self.save_image).pack(pady=5)


        combine_frame = tk.LabelFrame(self.controls_frame_left, text="Combine", font=self.font, bg="white", fg="#333", padx=5, pady=5)
        combine_frame.pack(pady=10, fill=tk.X)
        styled_btn(combine_frame, "Combine Images", self.combine_images).pack(pady=5)

        point_ops_frame = tk.LabelFrame(self.controls_frame_left, text="Point Operations", font=self.font, bg="white", fg="#333", padx=5, pady=5)
        point_ops_frame.pack(pady=10, fill=tk.X)
        point_val_frame = tk.Frame(point_ops_frame, bg="white")
        point_val_frame.pack()
        tk.Label(point_val_frame, text="Value:", bg="white", font=("Helvetica", 10)).pack(side=tk.LEFT, padx=2)
        self.value_entry = tk.Entry(point_val_frame, width=8)
        self.value_entry.pack(side=tk.LEFT)
        self.value_entry.insert(0, "10")
        self.operation_var = tk.StringVar(value="add")
        point_radio_frame = tk.Frame(point_ops_frame, bg="white")
        point_radio_frame.pack(pady=5)
        for op in ["add", "subtract", "multiply", "divide"]:
            tk.Radiobutton(point_radio_frame, text=op.title(), variable=self.operation_var, value=op, bg="white", font=("Helvetica", 10)).pack(side=tk.LEFT, padx=5)
        styled_btn(point_ops_frame, "Apply Point Operation", self.apply_operation).pack(pady=5)
        styled_btn(point_ops_frame, "Complement Image", self.complement_image).pack(pady=5)

        color_ops_frame = tk.LabelFrame(self.controls_frame_left, text="Color Operations", font=self.font, bg="white", fg="#333", padx=5, pady=5)
        color_ops_frame.pack(pady=10, fill=tk.X)
        remove_frame = tk.Frame(color_ops_frame, bg="white")
        remove_frame.pack(pady=5)
        tk.Label(remove_frame, text="Remove:", bg="white", font=("Helvetica", 10)).pack(side=tk.LEFT, padx=2)
        self.remove_channel_var = tk.StringVar(value="Red")
        channel_menu = tk.OptionMenu(remove_frame, self.remove_channel_var, "Red", "Green", "Blue")
        channel_menu.config(width=6, font=self.font)
        channel_menu.pack(side=tk.LEFT, padx=(0, 5))
        tk.Button(remove_frame, text="Go", command=self.remove_selected_channel, bg=self.btn_color, fg=self.btn_fg, font=self.font, width=3, height=1, bd=0).pack(side=tk.LEFT)
        swap_frame = tk.Frame(color_ops_frame, bg="white")
        swap_frame.pack(pady=5)
        tk.Label(swap_frame, text="Swap:", bg="white", font=("Helvetica", 10)).pack(side=tk.LEFT, padx=2)
        channel_options = ["Red", "Green", "Blue"]
        self.from_channel_var = tk.StringVar(value="Red")
        self.to_channel_var = tk.StringVar(value="Green")
        from_menu = tk.OptionMenu(swap_frame, self.from_channel_var, *channel_options)
        tk.Label(swap_frame, text=" <-> ", bg="white", font=("Helvetica", 10)).pack(side=tk.LEFT, padx=2)
        to_menu = tk.OptionMenu(swap_frame, self.to_channel_var, *channel_options)
        from_menu.config(width=6, font=self.font)
        to_menu.config(width=6, font=self.font)
        from_menu.pack(side=tk.LEFT)
        to_menu.pack(side=tk.LEFT)
        tk.Button(swap_frame, text="Go", command=self.swap_selected_channels, bg=self.btn_color, fg=self.btn_fg, font=self.font, width=3, height=1, bd=0).pack(side=tk.LEFT, padx=5)

        style = ttk.Style()
        style.configure('TNotebook.Tab', font=('Helvetica', '10', 'bold'))
        self.right_notebook = ttk.Notebook(self.controls_frame_right, style='TNotebook')
        tab_filters = tk.Frame(self.right_notebook, bg='white', padx=5, pady=5)
        tab_histogram = tk.Frame(self.right_notebook, bg='white', padx=5, pady=5)
        tab_features = tk.Frame(self.right_notebook, bg='white', padx=5, pady=5)
        tab_morphology = tk.Frame(self.right_notebook, bg='white', padx=5, pady=5)
        self.right_notebook.add(tab_filters, text='Filters')
        self.right_notebook.add(tab_histogram, text='Histogram')
        self.right_notebook.add(tab_features, text='Features')
        self.right_notebook.add(tab_morphology, text='Morphology')
        self.right_notebook.pack(expand=True, fill=tk.BOTH)

        def styled_right_btn(parent, text, command):
            return tk.Button(parent, text=text, command=command, bg=self.btn_color, fg=self.btn_fg, font=self.font, relief="raised", width=22, height=2, bd=0, padx=5)

        self.filter_frame = tk.LabelFrame(tab_filters, text="Spatial Filters", font=self.font, bg="white", fg="#333", padx=10, pady=10)
        self.filter_frame.pack(pady=10, padx=5, fill=tk.X)
        filters = [("Average Filter (3x3)", self.apply_average_filter), ("Laplacian Filter", self.apply_laplacian_filter), ("Minimum Filter (3x3)", self.apply_minimum_filter), ("Mode Filter (3x3)",self.apply_mode_builtin), ("Maximum Filter (3x3)", self.apply_maximum_filter), ("Median Filter (3x3)", self.apply_median_filter), ("outlier Filter",self.detect_outliers_builtin)]
        for text, command in filters: styled_right_btn(self.filter_frame, text, command).pack(padx=25,pady=5, anchor=tk.W)

        self.histogram_frame = tk.LabelFrame(tab_histogram, text="Histogram Operations", font=self.font, bg="white", fg="#333", padx=10, pady=10)
        self.histogram_frame.pack(pady=10, padx=5, fill=tk.X)
        hist_ops = [("Histogram Stretching", self.histogram_stretching), ("Histogram Equalization", self.histogram_equalization)]
        for text, command in hist_ops: styled_right_btn(self.histogram_frame, text, command).pack(pady=5,padx=25, anchor=tk.W)

        self.segmentation_frame = tk.LabelFrame(tab_features,text="Segmentation", font=self.font,bg="white",fg="#333",padx=10,pady=10)
        self.segmentation_frame.pack(pady=10, padx=5, fill=tk.X)
        styled_right_btn(self.segmentation_frame, "Segment Image (Otsu Auto)", self.segment_image).pack(pady=5)
        styled_right_btn(self.segmentation_frame, "Global Thresholding", self.global_thresholding).pack(pady=5)
        styled_right_btn(self.segmentation_frame, "Adaptive Thresholding", self.adaptive_thresholding).pack(pady=5)
        styled_right_btn(self.segmentation_frame, "Image Averaging (Blur 3x3)", self.image_averaging_builtin).pack(pady=5)

        self.edge_detection_frame = tk.LabelFrame(tab_features,text="Edge Detection", font=self.font,bg="white",fg="#333",padx=25)
        self.edge_detection_frame.pack(pady=10, padx=5, fill=tk.X)
        styled_right_btn(self.edge_detection_frame, "Edge Detect (Canny)", self.edge_detection).pack(pady=5,padx=25, anchor=tk.W)

        self.Mathematical_Morphology_frame = tk.LabelFrame(tab_morphology,text="Mathematical Morphology (5x5)", font=self.font,bg="white",fg="#333",padx=10,pady=10)
        self.Mathematical_Morphology_frame.pack(pady=10, padx=5, fill=tk.X)
        styled_right_btn(self.Mathematical_Morphology_frame, "Dilation", self.image_dilation).pack(pady=5,padx=25, anchor=tk.W)
        styled_right_btn(self.Mathematical_Morphology_frame, "Erosion", self.image_erosion).pack(pady=5,padx=25, anchor=tk.W)
        styled_right_btn(self.Mathematical_Morphology_frame, "Opening", self.image_opening).pack(pady=5,padx=25, anchor=tk.W)
        styled_right_btn(self.Mathematical_Morphology_frame, "Internal Boundary", self.extract_internal_boundary).pack(pady=5,padx=25, anchor=tk.W)
        styled_right_btn(self.Mathematical_Morphology_frame, "External Boundary", self.extract_external_boundary).pack(pady=5,padx=25, anchor=tk.W)
        styled_right_btn(self.Mathematical_Morphology_frame, "Morphological Gradient", self.calculate_morph_gradient).pack(pady=5,padx=25, anchor=tk.W)

        self.original_frame = tk.LabelFrame(self.images_frame, text="Original Image", font=self.font, bg="white", fg="#333", padx=10, pady=10)
        self.original_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        self.edited_frame = tk.LabelFrame(self.images_frame, text="Edited Image", font=self.font, bg="white", fg="#333", padx=10, pady=10)
        self.edited_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        self.images_frame.grid_rowconfigure(0, weight=1)
        self.images_frame.grid_columnconfigure(0, weight=1)
        self.images_frame.grid_columnconfigure(1, weight=1)

    def create_placeholder(self):
        placeholder = Image.new("RGB", self.img_size, color="#cccccc")
        self.placeholder_photo = ImageTk.PhotoImage(placeholder)
        for widget in self.original_frame.winfo_children(): widget.destroy()
        for widget in self.edited_frame.winfo_children(): widget.destroy()
        self.old_image_panel = tk.Label(self.original_frame, image=self.placeholder_photo, bg="white")
        self.old_image_panel.image = self.placeholder_photo
        self.old_image_panel.pack(expand=True, fill=tk.BOTH)
        self.edited_image_panel = tk.Label(self.edited_frame, image=self.placeholder_photo, bg="white")
        self.edited_image_panel.image = self.placeholder_photo
        self.edited_image_panel.pack(expand=True, fill=tk.BOTH)

    def display_image(self, img, panel):
        if not panel or not panel.winfo_exists(): return
        display_photo = self.placeholder_photo
        if img is not None:
            panel_w = panel.winfo_width()
            panel_h = panel.winfo_height()
            if panel_w <= 1: panel_w = self.root.winfo_width() // 3
            if panel_h <= 1: panel_h = self.root.winfo_height() - 50
            img_resized = img.copy()
            thumb_size = (max(10, panel_w - 20), max(10, panel_h - 20))
            img_resized.thumbnail(thumb_size, Image.Resampling.LANCZOS)
            display_photo = ImageTk.PhotoImage(img_resized)
        panel.config(image=display_photo)
        panel.image = display_photo

    def _update_edited_display(self):
        self.root.after(10, lambda: self.display_image(self.edited_image, self.edited_image_panel))

    def _ensure_original_image_loaded(self):
        if self.original_image is None:
            messagebox.showwarning("Warning", "Please load an image first.")
            return False
        return True

    def _convert_to_array(self, image_pil):
        if image_pil is None: return None
        return np.array(image_pil.convert('RGB'))[:, :, ::-1]

    def _convert_to_pil(self, image_cv):
        if image_cv is None: return None
        if image_cv.ndim == 3 and image_cv.shape[2] == 3:
            image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
            return Image.fromarray(image_rgb)
        elif image_cv.ndim == 2:
            return Image.fromarray(image_cv, mode='L')
        else:
            if image_cv.ndim == 3 and image_cv.shape[2] == 1:
                return Image.fromarray(image_cv[:,:,0], mode='L')
            else:
                return Image.fromarray(image_cv)

    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp *.gif *.tif *.tiff")])
        if path:
            self.original_image = Image.open(path).convert("RGB")
            self.edited_image = self.original_image.copy()
            self.display_image(self.original_image, self.old_image_panel)
            self._update_edited_display()
            self.root.after(50, lambda: self.display_image(self.original_image, self.old_image_panel))
            self.root.after(50, self._update_edited_display)

    def load_second_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp *.gif *.tif *.tiff")])
        if path:
            self.second_image = Image.open(path).convert("RGB")
            messagebox.showinfo("Info", "Second image loaded successfully (as RGB).")

    def apply_operation(self):
        if not self._ensure_original_image_loaded(): return
        operation = self.operation_var.get()
        value = float(self.value_entry.get())
        img = self.original_image.copy()
        img_array = np.array(img, dtype=np.float32)
        if operation == "add": img_array += value
        elif operation == "subtract": img_array -= value
        elif operation == "multiply": img_array *= value
        elif operation == "divide":
            if abs(value) < 1e-9:
                messagebox.showerror("Error", "Division by zero or very small number.")
                return
            img_array /= value
        result_array = np.clip(img_array, 0, 255).astype(np.uint8)
        self.edited_image = Image.fromarray(result_array)
        self._update_edited_display()

    def combine_images(self):
        if not self._ensure_original_image_loaded(): return
        if self.second_image is None:
            messagebox.showwarning("Warning", "Load second image first.")
            return
        img1 = self.original_image.copy()
        resized_second = self.second_image.resize(img1.size, Image.Resampling.LANCZOS)
        self.edited_image = ImageChops.add(img1, resized_second, scale=1.0, offset=0)
        self._update_edited_display()

    def complement_image(self):
        if not self._ensure_original_image_loaded(): return
        img = self.original_image.copy()
        self.edited_image = ImageChops.invert(img)
        self._update_edited_display()

    def remove_selected_channel(self):
        if not self._ensure_original_image_loaded(): return
        channel_map = {"Red": 0, "Green": 1, "Blue": 2}
        selected_channel_index = channel_map.get(self.remove_channel_var.get())
        img = self.original_image.copy()
        img_array = np.array(img)
        img_array[:, :, selected_channel_index] = 0
        self.edited_image = Image.fromarray(img_array)
        self._update_edited_display()

    def swap_selected_channels(self):
        if not self._ensure_original_image_loaded(): return
        channel_map = {"Red": 0, "Green": 1, "Blue": 2}
        from_ch = channel_map[self.from_channel_var.get()]
        to_ch = channel_map[self.to_channel_var.get()]
        if from_ch == to_ch:
            messagebox.showwarning("Warning", "Channels must be different to swap.")
            return
        img = self.original_image.copy()
        img_array = np.array(img)
        img_array[..., [from_ch, to_ch]] = img_array[..., [to_ch, from_ch]]
        self.edited_image = Image.fromarray(img_array)
        self._update_edited_display()

    def histogram_stretching(self):
        if not self._ensure_original_image_loaded(): return
        img = self.original_image.copy()
        img_array = np.array(img)
        is_rgb = img_array.ndim == 3 and img_array.shape[2] == 3
        stretched_array = np.zeros_like(img_array, dtype=np.float32)
        if is_rgb:
            for i in range(3):
                channel = img_array[..., i].astype(np.float32)
                c_min, c_max = np.min(channel), np.max(channel)
                denominator = c_max - c_min
                if denominator > 1e-9: stretched_array[..., i] = ((channel - c_min) / denominator) * 255.0
                else: stretched_array[..., i] = channel
        else:
            channel = img_array.astype(np.float32)
            if channel.ndim == 3 and channel.shape[2]==1: channel = channel[..., 0]
            elif channel.ndim != 2:
                messagebox.showerror("Error", "Unsupported image format for grayscale stretching.")
                return
            c_min, c_max = np.min(channel), np.max(channel)
            denominator = c_max - c_min
            if denominator > 1e-9: stretched_array = ((channel - c_min) / denominator) * 255.0
            else: stretched_array = channel
        final_array = np.clip(stretched_array, 0, 255).astype(np.uint8)
        self.edited_image = self._convert_to_pil(final_array)
        if self.edited_image: self._update_edited_display()
        else: messagebox.showerror("Error", "Failed to convert stretched image back to display format.")

    def histogram_equalization(self):
        if not self._ensure_original_image_loaded(): return
        img_cv = self._convert_to_array(self.original_image.copy())
        if img_cv is None: return
        if len(img_cv.shape) == 2 or img_cv.shape[2] == 1:
            if len(img_cv.shape) == 3: img_cv = img_cv[:,:,0]
            equalized_cv = cv2.equalizeHist(img_cv)
        else:
            img_ycrcb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2YCrCb)
            y, cr, cb = cv2.split(img_ycrcb)
            y_equalized = cv2.equalizeHist(y)
            img_ycrcb_equalized = cv2.merge([y_equalized, cr, cb])
            equalized_cv = cv2.cvtColor(img_ycrcb_equalized, cv2.COLOR_YCrCb2BGR)
        self.edited_image = self._convert_to_pil(equalized_cv)
        if self.edited_image: self._update_edited_display()
        else: messagebox.showerror("Error", "Failed to convert equalized image back.")

    def apply_average_filter(self):
        if not self._ensure_original_image_loaded(): return
        img = self.original_image.copy()
        self.edited_image = img.filter(ImageFilter.BoxBlur(1))
        self._update_edited_display()

    def apply_laplacian_filter(self):
        if not self._ensure_original_image_loaded(): return
        img_cv_bgr = self._convert_to_array(self.original_image.copy())
        if img_cv_bgr is None: return
        img_cv_gray = cv2.cvtColor(img_cv_bgr, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(img_cv_gray, cv2.CV_64F, ksize=3)
        laplacian_abs = cv2.convertScaleAbs(laplacian)
        self.edited_image = self._convert_to_pil(laplacian_abs)
        if self.edited_image: self._update_edited_display()
        else: messagebox.showerror("Error", "Failed to convert Laplacian result back.")

    def apply_minimum_filter(self):
        if not self._ensure_original_image_loaded(): return
        img = self.original_image.copy()
        self.edited_image = img.filter(ImageFilter.MinFilter(size=3))
        self._update_edited_display()

    def apply_maximum_filter(self):
        if not self._ensure_original_image_loaded(): return
        img = self.original_image.copy()
        self.edited_image = img.filter(ImageFilter.MaxFilter(size=3))
        self._update_edited_display()

    def apply_median_filter(self):
        if not self._ensure_original_image_loaded(): return
        img = self.original_image.copy()
        self.edited_image = img.filter(ImageFilter.MedianFilter(size=3))
        self._update_edited_display()

    def segment_image(self):
        if not self._ensure_original_image_loaded(): return
        img_cv_bgr = self._convert_to_array(self.original_image.copy())
        if img_cv_bgr is None: return
        img_cv_gray = cv2.cvtColor(img_cv_bgr, cv2.COLOR_BGR2GRAY)
        _, segmented_cv = cv2.threshold(img_cv_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        self.edited_image = self._convert_to_pil(segmented_cv)
        if self.edited_image: self._update_edited_display()
        else: messagebox.showerror("Error", "Failed to convert segmented image back.")

    def global_thresholding(self):
        if not self._ensure_original_image_loaded(): return
        img_cv_gray = cv2.cvtColor(np.array(self.original_image.copy()), cv2.COLOR_RGB2GRAY)
        T = 127
        epsilon = 1.0
        max_iterations = 100
        prev_T = 0
        iteration = 0
        while abs(T - prev_T) >= epsilon and iteration < max_iterations:
            G1 = img_cv_gray[img_cv_gray > T]
            G2 = img_cv_gray[img_cv_gray <= T]
            if len(G1) == 0 or len(G2) == 0:
                if len(G1) == 0 and len(G2) > 0: T = np.mean(G2)
                elif len(G2) == 0 and len(G1) > 0: T = np.mean(G1)
                else: T = 127
                break
            m1 = np.mean(G1)
            m2 = np.mean(G2)
            prev_T = T
            T = (m1 + m2) / 2
            iteration += 1
        _, segmented_cv = cv2.threshold(img_cv_gray, T, 255, cv2.THRESH_BINARY)
        self.edited_image = self._convert_to_pil(segmented_cv)
        if self.edited_image: self._update_edited_display()
        else: messagebox.showerror("Error", "Failed to convert global threshold result.")

    def adaptive_thresholding(self):
        if not self._ensure_original_image_loaded(): return
        img_cv_gray = cv2.cvtColor(np.array(self.original_image.copy()), cv2.COLOR_RGB2GRAY)
        segmented_cv = cv2.adaptiveThreshold(img_cv_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        self.edited_image = self._convert_to_pil(segmented_cv)
        if self.edited_image: self._update_edited_display()
        else: messagebox.showerror("Error", "Failed to convert adaptive threshold result.")

    def edge_detection(self):
        if not self._ensure_original_image_loaded(): return
        img_cv_bgr = self._convert_to_array(self.original_image.copy())
        if img_cv_bgr is None: return
        img_cv_gray = cv2.cvtColor(img_cv_bgr, cv2.COLOR_BGR2GRAY)
        edges_cv = cv2.Canny(img_cv_gray, 100, 200)
        self.edited_image = self._convert_to_pil(edges_cv)
        if self.edited_image: self._update_edited_display()
        else: messagebox.showerror("Error", "Failed to convert Canny edges.")

    def _apply_morphology_cv(self, operation_func, kernel_size=(5, 5), iterations=1):
        if not self._ensure_original_image_loaded(): return False
        img_cv = self._convert_to_array(self.original_image.copy())
        if img_cv is None: return False
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
        result_cv = operation_func(img_cv, kernel, iterations=iterations)
        self.edited_image = self._convert_to_pil(result_cv)
        if self.edited_image:
            self._update_edited_display()
            return True
        else:
            messagebox.showerror("Error", "Failed to convert morphology result back.")
            return False

    def image_dilation(self):
        self._apply_morphology_cv(cv2.dilate)

    def image_erosion(self):
        self._apply_morphology_cv(cv2.erode)

    def image_opening(self):
        op = lambda img, kernel, it: cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=it)
        self._apply_morphology_cv(op)

    def _extract_boundary_cv(self, boundary_type):
        if not self._ensure_original_image_loaded(): return
        img_cv = self._convert_to_array(self.original_image.copy())
        if img_cv is None: return
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        if boundary_type == 'internal':
            processed = cv2.erode(img_cv, kernel, iterations=1)
            boundary = cv2.subtract(img_cv, processed)
        elif boundary_type == 'external':
            processed = cv2.dilate(img_cv, kernel, iterations=1)
            boundary = cv2.subtract(processed, img_cv)
        else:
            messagebox.showerror("Error", "Invalid boundary type specified.")
            return
        self.edited_image = self._convert_to_pil(boundary)
        if self.edited_image: self._update_edited_display()
        else: messagebox.showerror("Error", "Failed to convert boundary image back.")

    def extract_internal_boundary(self):
        self._extract_boundary_cv('internal')

    def extract_external_boundary(self):
        self._extract_boundary_cv('external')

    def calculate_morph_gradient(self):
        op = lambda img, kernel, it: cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel, iterations=it)
        self._apply_morphology_cv(op)

    def save_image(self):
        if self.edited_image is None:
            messagebox.showwarning("Warning", "No edited image to save.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg;*.jpeg"), ("BMP files", "*.bmp"), ("TIFF files", "*.tif;*.tiff"), ("All files", "*.*")])
        if path:
            save_img = self.edited_image
            save_img.save(path)
            messagebox.showinfo("Success", f"Image saved successfully to\n{path}")

    def apply_mode_builtin(self):
        if not self._ensure_original_image_loaded(): return
        img_cv_gray = cv2.cvtColor(np.array(self.original_image.copy()), cv2.COLOR_RGB2GRAY)
        filtered = cv2.medianBlur(img_cv_gray, 3)
        self.edited_image = self._convert_to_pil(filtered)
        if self.edited_image: self._update_edited_display()
        else: messagebox.showerror("Error", "Failed to convert mode filter result.")

    def detect_outliers_builtin(self):
        if not self._ensure_original_image_loaded(): return
        img_cv_gray = cv2.cvtColor(np.array(self.original_image.copy()), cv2.COLOR_RGB2GRAY)
        pixels = img_cv_gray.flatten()
        q1 = np.percentile(pixels, 25)
        q3 = np.percentile(pixels, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outlier_mask = (img_cv_gray < lower_bound) | (img_cv_gray > upper_bound)
        non_outliers = pixels[~outlier_mask.flatten()]
        if len(non_outliers) > 0: mean_val = np.mean(non_outliers)
        else: mean_val = np.mean(pixels)
        filtered_img = img_cv_gray.copy()
        filtered_img[outlier_mask] = int(mean_val)
        self.edited_image = self._convert_to_pil(filtered_img)
        if self.edited_image: self._update_edited_display()
        else: messagebox.showerror("Error", "Failed to convert outlier result.")

    def image_averaging_builtin(self):
        if not self._ensure_original_image_loaded(): return
        img_cv_bgr = self._convert_to_array(self.original_image.copy())
        if img_cv_bgr is None: return
        img_blur = cv2.blur(img_cv_bgr, (3, 3))
        self.edited_image = self._convert_to_pil(img_blur)
        if self.edited_image: self._update_edited_display()
        else: messagebox.showerror("Error", "Failed to convert averaged image.")

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageEditorApp(root)
    root.mainloop()