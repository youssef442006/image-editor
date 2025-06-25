import tkinter as tk
from tkinter import ttk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageChops
import numpy as np
import cv2

def _pad_image_manual(image_array, pad_width, mode='edge', constant_values=0):
    if image_array.ndim == 3:
        h, w, c = image_array.shape
        padded_shape = (h + 2 * pad_width, w + 2 * pad_width, c)
        if mode == 'constant':
            padded_image = np.full(padded_shape, constant_values, dtype=image_array.dtype)
        else: 
            padded_image = np.zeros(padded_shape, dtype=image_array.dtype)
        
        padded_image[pad_width:h+pad_width, pad_width:w+pad_width, :] = image_array

        if mode == 'edge':
            padded_image[:pad_width, pad_width:w+pad_width, :] = image_array[0:1, :, :]
            padded_image[h+pad_width:, pad_width:w+pad_width, :] = image_array[-1:, :, :]
            for i in range(c):
                padded_image[:, :pad_width, i] = padded_image[:, pad_width:pad_width+1, i]
                padded_image[:, w+pad_width:, i] = padded_image[:, w+pad_width-1:w+pad_width, i]

    elif image_array.ndim == 2:
        h, w = image_array.shape
        padded_shape = (h + 2 * pad_width, w + 2 * pad_width)
        if mode == 'constant':
            padded_image = np.full(padded_shape, constant_values, dtype=image_array.dtype)
        else:
            padded_image = np.zeros(padded_shape, dtype=image_array.dtype)

        padded_image[pad_width:h+pad_width, pad_width:w+pad_width] = image_array

        if mode == 'edge':
            padded_image[:pad_width, pad_width:w+pad_width] = image_array[0:1, :]
            padded_image[h+pad_width:, pad_width:w+pad_width] = image_array[-1:, :]
            padded_image[:, :pad_width] = padded_image[:, pad_width:pad_width+1]
            padded_image[:, w+pad_width:] = padded_image[:, w+pad_width-1:w+pad_width]
    else:
        print("Error: Input array must be 2D or 3D.")
        return None 
    return padded_image

def _convolve_manual(image_array, kernel):
    output_array = None 
    if image_array.ndim == 3 and kernel.ndim == 2:
        h, w, c = image_array.shape
        kh, kw = kernel.shape
        pad_h, pad_w = kh // 2, kw // 2

        padded_image = _pad_image_manual(image_array, max(pad_h, pad_w), mode='edge')
        if padded_image is None: return None 
        output_array = np.zeros_like(image_array, dtype=np.float64)

        for channel in range(c):
            for i in range(h):
                for j in range(w):
                    neighborhood = padded_image[i:i+kh, j:j+kw, channel]
                    output_array[i, j, channel] = np.sum(neighborhood * kernel)

    elif image_array.ndim == 2 and kernel.ndim == 2:
        h, w = image_array.shape
        kh, kw = kernel.shape
        pad_h, pad_w = kh // 2, kw // 2

        padded_image = _pad_image_manual(image_array, max(pad_h, pad_w), mode='edge')
        if padded_image is None: return None
        output_array = np.zeros_like(image_array, dtype=np.float64)

        for i in range(h):
            for j in range(w):
                neighborhood = padded_image[i:i+kh, j:j+kw]
                output_array[i, j] = np.sum(neighborhood * kernel)
    else:
        print("Error: Unsupported image/kernel dimensions for convolution.")
        return None

    output_array = np.clip(output_array, 0, 255)
    return output_array.astype(np.uint8)


def _filter_neighborhood_manual(image_array, size, operation):
    if size % 2 == 0:
        print("Error: Filter size must be odd.")
        return None 
    pad = size // 2
    padded_image = _pad_image_manual(image_array, pad, mode='edge')
    if padded_image is None: return None 
    output_array = np.zeros_like(image_array) 

    if image_array.ndim == 3:
        h, w, c = image_array.shape
        for channel in range(c):
            for i in range(h):
                for j in range(w):
                    neighborhood = padded_image[i:i+size, j:j+size, channel]
                    if operation == 'min': output_array[i, j, channel] = np.min(neighborhood)
                    elif operation == 'max': output_array[i, j, channel] = np.max(neighborhood)
                    elif operation == 'median': output_array[i, j, channel] = np.median(neighborhood)
    elif image_array.ndim == 2:
        h, w = image_array.shape
        for i in range(h):
            for j in range(w):
                neighborhood = padded_image[i:i+size, j:j+size]
                if operation == 'min': output_array[i, j] = np.min(neighborhood)
                elif operation == 'max': output_array[i, j] = np.max(neighborhood)
                elif operation == 'median': output_array[i, j] = np.median(neighborhood)
    else:
        print("Error: Unsupported image dimensions for neighborhood filter.")
        return None

    return output_array.astype(np.uint8)

def _morphology_manual(image_array, kernel_size=(5,5), operation='dilate', iterations=1):
    if image_array.ndim == 3:
        gray_array = np.mean(image_array, axis=2).astype(np.uint8)
    elif image_array.ndim == 2:
        gray_array = image_array
    else:
        print("Error: Unsupported image dimensions for morphology.")
        return None

    threshold = 128
    binary_array = (gray_array > threshold).astype(np.uint8) * 255

    kh, kw = kernel_size
    pad_h, pad_w = kh // 2, kw // 2

    current_array = binary_array
    structuring_element = np.ones(kernel_size, dtype=bool)

    for _ in range(iterations):
        padded_image = _pad_image_manual(current_array, max(pad_h, pad_w), mode='constant', constant_values=0)
        if padded_image is None: return None 
        output_array = np.zeros_like(current_array)
        h_curr, w_curr = current_array.shape # Renamed to avoid conflict if h,w are global

        for i in range(h_curr):
            for j in range(w_curr):
                neighborhood = padded_image[i:i+kh, j:j+kw]
                relevant_pixels = neighborhood[structuring_element]

                if operation == 'dilate':
                    if np.any(relevant_pixels == 255): output_array[i, j] = 255
                elif operation == 'erode':
                    if np.all(relevant_pixels == 255): output_array[i, j] = 255
        current_array = output_array
    
    return Image.fromarray(current_array.astype(np.uint8), mode='L')


class ImageEditorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image processing (Manual Simple)")
        self.root.geometry("1700x950")
        self.root.configure(bg="#f0f0f0")

        self.original_image = None
        self.second_image = None
        self.edited_image = None

        self.btn_color = "#4a90e2"
        self.btn_fg = "white"
        self.font = ("Helvetica", 10, "bold")
        self.img_size = (400, 400)

        self.controls_frame_left = tk.LabelFrame(root, text="Image & Basic Ops", font=("Helvetica", 11, "bold"),
                                                 padx=10, pady=10, bg="white", fg="#333", relief=tk.RIDGE, bd=1)
        self.controls_frame_left.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        self.controls_frame_right = tk.Frame(root, bg="white", relief=tk.RIDGE, bd=1)
        self.controls_frame_right.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)

        self.images_frame = tk.Frame(root, bg="#f0f0f0")
        self.images_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=10)

        self.create_widgets()
        self.create_placeholder()

    def create_widgets(self):
        def styled_btn(parent, text, command):
            return tk.Button(parent, text=text, command=command,
                            bg=self.btn_color, fg=self.btn_fg, font=self.font,
                            relief="raised", width=23, height=2, bd=0, padx=5)

        file_ops_frame = tk.LabelFrame(self.controls_frame_left, text="File", font=self.font, bg="white", fg="#333", padx=5, pady=5)
        file_ops_frame.pack(pady=10, fill=tk.X)
        styled_btn(file_ops_frame, "Upload Image", self.load_image).pack(pady=5)
        styled_btn(file_ops_frame, "Load Second Image", self.load_second_image).pack(pady=5)
        styled_btn(file_ops_frame, "Save Edited Image", self.save_image).pack(pady=5)
        styled_btn(file_ops_frame, "Reset Image", self.reset_image).pack(pady=5)
        self.root.bind("<Control-z>", lambda event: self.reset_image()) 
        
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
            tk.Radiobutton(point_radio_frame, text=op.title(), variable=self.operation_var, value=op,
                             bg="white", font=("Helvetica", 10)).pack(side=tk.LEFT, padx=5)

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
        tk.Button(remove_frame, text="Go", command=self.remove_selected_channel,
                  bg=self.btn_color, fg=self.btn_fg, font=self.font, width=3, height=1, bd=0).pack(side=tk.LEFT)

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
        tk.Button(swap_frame, text="Go", command=self.swap_selected_channels,
                  bg=self.btn_color, fg=self.btn_fg, font=self.font, width=3, height=1, bd=0).pack(side=tk.LEFT, padx=5)

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
            return tk.Button(parent, text=text, command=command,
                             bg=self.btn_color, fg=self.btn_fg, font=self.font,
                             relief="raised", width=22, height=2, bd=0, padx=5)

        self.filter_frame = tk.LabelFrame(tab_filters, text="Spatial Filters", font=self.font,
                                           bg="white", fg="#333", padx=10, pady=10)
        self.filter_frame.pack(pady=10, padx=25, fill=tk.X)

        filters = [
            ("Average Filter (3x3)", self.apply_average_filter), 
            ("Laplacian Filter", self.apply_laplacian_filter),  
            ("Minimum Filter (3x3)", self.apply_minimum_filter), 
            ("Mode Filter (3x3)",self.mode_manual),
            ("Maximum Filter (3x3)", self.apply_maximum_filter), 
            ("Median Filter (3x3)", self.apply_median_filter),  
            ("outlier Filter",self.detect_outliers_manual)
        ]
        for text, command in filters:
            styled_right_btn(self.filter_frame, text, command).pack(pady=5, anchor=tk.W)

        self.histogram_frame = tk.LabelFrame(tab_histogram, text="Histogram Operations",
                                              font=self.font, bg="white", fg="#333", padx=10, pady=10)
        self.histogram_frame.pack(pady=10, padx=5, fill=tk.X)

        hist_ops = [
            ("Histogram Stretching", self.histogram_stretching),
            ("Histogram equalization", self.histogram_equalization),
        ]
        for text, command in hist_ops:
            styled_right_btn(self.histogram_frame, text, command).pack(pady=5,padx=20, anchor=tk.W)

        self.segmentation_frame = tk.LabelFrame(tab_features,text="Segmentation",
                                                 font=self.font,bg="white",fg="#333",padx=10,pady=10)
        self.segmentation_frame.pack(pady=10, padx=5, fill=tk.X)
        styled_right_btn(self.segmentation_frame, "Segment Image", self.segment_image).pack(pady=5)
        styled_right_btn(self.segmentation_frame, "Global Thresholding", self.global_thresholding).pack(pady=5) 
        styled_right_btn(self.segmentation_frame, "Automatic Thresholding (Otsu)", self.automatic_thresholding).pack(pady=5)
        styled_right_btn(self.segmentation_frame, "Adaptive Thresholding", self.adaptive_thresholding).pack(pady=5)
        styled_right_btn(self.segmentation_frame, "Image Averaging", self.image_averaging_manual).pack(pady=5)


        self.Mathematical_Morphology_frame = tk.LabelFrame(tab_morphology,text="Mathematical Morphology (5x5)",
                                                            font=self.font,bg="white",fg="#333",padx=10,pady=10)
        self.Mathematical_Morphology_frame.pack(pady=10, padx=5, fill=tk.X)

        styled_right_btn(self.Mathematical_Morphology_frame, "Dilation", self.image_dilation).pack(pady=5,padx=25, anchor=tk.W)
        styled_right_btn(self.Mathematical_Morphology_frame, "Erosion", self.image_erosion).pack(pady=5,padx=25, anchor=tk.W)

        self.original_frame = tk.LabelFrame(self.images_frame, text="Original Image", font=self.font,
                                           bg="white", fg="#333", padx=10, pady=10)
        self.original_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        self.edited_frame = tk.LabelFrame(self.images_frame, text="Edited Image", font=self.font,
                                         bg="white", fg="#333", padx=10, pady=10)
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
            display_thumb_size = (400,400) 
            img_resized = img.copy()
            img_resized.thumbnail(display_thumb_size, Image.Resampling.LANCZOS)
            display_photo = ImageTk.PhotoImage(img_resized)
        panel.config(image=display_photo)
        panel.image = display_photo

    def _update_edited_display(self):
        self.root.after(10, lambda: self.display_image(self.edited_image, self.edited_image_panel))

    def _ensure_image_loaded(self):
        if self.edited_image is None:
            messagebox.showwarning("Warning", "Please load an image first.")
            return False
        return True
    
    def _ensure_original_image_loaded(self):
        if self.original_image is None:
            messagebox.showwarning("Warning", "Please load an original image first.")
            return False
        return True

    def _convert_to_array(self, image_pil):
        if image_pil is None: return None
        return np.array(image_pil)


    def _convert_to_pil(self, image_array):
        if image_array is None: return None
        mode = 'L' 
        if image_array.ndim == 2: mode = 'L'
        elif image_array.ndim == 3:
            if image_array.shape[2] == 3: mode = 'RGB'
            elif image_array.shape[2] == 4: mode = 'RGBA'
            elif image_array.shape[2] == 1:
                mode = 'L'
                image_array = image_array[:, :, 0] 
            else:
                print(f"Error: Unsupported channel count: {image_array.shape[2]}")
                return None
        else:
            print(f"Error: Unsupported array dimension: {image_array.ndim}")
            return None 

        if image_array.dtype != np.uint8:
            image_array = np.clip(image_array, 0, 255)
            image_array = image_array.astype(np.uint8)
        try:
            return Image.fromarray(image_array, mode=mode)
        except Exception as e:
            print(f"Error converting array to PIL Image: {e}, mode: {mode}, array_shape: {image_array.shape}, dtype: {image_array.dtype}")
            return None


    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp *.gif *.tif *.tiff")])
        if path:
            try:
                self.original_image = Image.open(path)               
                self.edited_image = self.original_image.copy().convert("RGB") 
                self.display_image(self.original_image, self.old_image_panel)
                self._update_edited_display() 
                self.root.after(100, lambda: self.display_image(self.original_image, self.old_image_panel))
                self.root.after(100, self._update_edited_display)
            except Exception as e:
                messagebox.showerror("Error Loading Image", f"Could not load image: {e}")
                self.original_image = None
                self.edited_image = None
                self.create_placeholder() 


    def load_second_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp *.gif *.tif *.tiff")])
        if path:
            try:
                self.second_image = Image.open(path).convert("RGB")
                messagebox.showinfo("Info", "Second image loaded successfully (as RGB).")
            except Exception as e:
                messagebox.showerror("Error Loading Second Image", f"Could not load second image: {e}")
                self.second_image = None


    def reset_image(self):
        if not self._ensure_original_image_loaded(): return
        self.edited_image = self.original_image.copy().convert("RGB")
        self._update_edited_display()

    def apply_operation(self):
        if not self._ensure_image_loaded(): return
        operation = self.operation_var.get()
        try:
            value = float(self.value_entry.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid value for operation. Please enter a number.")
            return

        img_array = self._convert_to_array(self.edited_image)
        if img_array is None: return

        img_array_float = img_array.astype(np.float32) 
        if operation == "add": img_array_float += value
        elif operation == "subtract": img_array_float -= value
        elif operation == "multiply": img_array_float *= value
        elif operation == "divide":
            if abs(value) < 1e-9: 
                messagebox.showerror("Error", "Division by zero or a very small number.")
                return
            img_array_float /= value
        self.edited_image = self._convert_to_pil(img_array_float) 
        if self.edited_image: self._update_edited_display()

    def combine_images(self):
        if not self._ensure_image_loaded(): return
        if self.second_image is None:
            messagebox.showwarning("Warning", "Load second image first.")
            return
        
        if self.edited_image.mode != 'RGB':
            self.edited_image = self.edited_image.convert('RGB')
        
        resized_second = self.second_image.resize(self.edited_image.size, Image.Resampling.LANCZOS).convert('RGB')
        
        self.edited_image = ImageChops.add(self.edited_image, resized_second) 
        self._update_edited_display()

    def complement_image(self):
        if not self._ensure_image_loaded(): return
        if self.edited_image.mode == 'RGBA':
             self.edited_image = self.edited_image.convert('RGB')
        self.edited_image = ImageChops.invert(self.edited_image)
        self._update_edited_display()

    def remove_selected_channel(self):
        if not self._ensure_image_loaded(): return
        channel_map = {"Red": 0, "Green": 1, "Blue": 2}
        selected_channel_index = channel_map.get(self.remove_channel_var.get())
        
        if self.edited_image.mode != 'RGB':
            try:
                self.edited_image = self.edited_image.convert("RGB")
            except ValueError:
                 messagebox.showerror("Error", "Image must be convertible to RGB to remove a channel.")
                 return

        img_array = self._convert_to_array(self.edited_image)
        if img_array is None: return 
        if img_array.ndim == 3 and img_array.shape[2] == 3 : 
            img_array_copy = img_array.copy() 
            img_array_copy[:, :, selected_channel_index] = 0
            self.edited_image = self._convert_to_pil(img_array_copy)
            if self.edited_image: self._update_edited_display()
        else: 
            messagebox.showerror("Error", "Cannot remove channel. Image is not in a 3-channel RGB format after conversion.")


    def swap_selected_channels(self):
        if not self._ensure_image_loaded(): return
        channel_map = {"Red": 0, "Green": 1, "Blue": 2}
        from_ch_name = self.from_channel_var.get()
        to_ch_name = self.to_channel_var.get()

        if from_ch_name == to_ch_name:
            messagebox.showwarning("Warning", "Channels to swap must be different.")
            return

        from_ch, to_ch = channel_map[from_ch_name], channel_map[to_ch_name]
        
        if self.edited_image.mode != 'RGB':
            try:
                self.edited_image = self.edited_image.convert("RGB")
            except ValueError:
                messagebox.showerror("Error", "Image must be convertible to RGB to swap channels.")
                return
            
        img_array = self._convert_to_array(self.edited_image)
        if img_array is None: return

        if img_array.ndim == 3 and img_array.shape[2] == 3: 
            img_array_copy = img_array.copy() 
            temp_channel_data = img_array_copy[..., from_ch].copy()
            img_array_copy[..., from_ch] = img_array_copy[..., to_ch]
            img_array_copy[..., to_ch] = temp_channel_data
            
            self.edited_image = self._convert_to_pil(img_array_copy)
            if self.edited_image: self._update_edited_display()
        else:
            messagebox.showerror("Error", "Cannot swap channels. Image is not in a 3-channel RGB format after conversion.")

    def histogram_equalization(img_array):
        def equalize_channel(channel):
            flat = channel.flatten()
            hist = np.zeros(256, dtype=np.int32)
            for value in flat:
                hist[value] += 1
            cdf = np.cumsum(hist)
            cdf_min = cdf[np.nonzero(cdf)[0][0]]
            cdf_normalized = ((cdf - cdf_min) / (flat.size - cdf_min)) * 255
            cdf_normalized = np.clip(cdf_normalized, 0, 255).astype(np.uint8)
            equalized = cdf_normalized[flat].reshape(channel.shape)
            return equalized

        if img_array.ndim == 2:
            return equalize_channel(img_array)
        elif img_array.ndim == 3 and img_array.shape[2] == 3:
            return np.stack([equalize_channel(img_array[:, :, i]) for i in range(3)], axis=2)
        else:
            raise ValueError("Unsupported image format for histogram equalization.")
    def histogram_stretching(self):
        if not self._ensure_image_loaded(): return
        img_array = self._convert_to_array(self.edited_image)
        if img_array is None: return
        stretched_array_float = img_array.astype(np.float32)

        if img_array.ndim == 3 and img_array.shape[2] == 3: 
            for i in range(3): 
                channel = stretched_array_float[..., i]
                c_min, c_max = np.min(channel), np.max(channel)
                denominator = c_max - c_min
                if denominator > 1e-9: 
                    stretched_array_float[..., i] = ((channel - c_min) / denominator) * 255.0
               
        elif img_array.ndim == 2 or (img_array.ndim == 3 and img_array.shape[2] == 1): 
            if img_array.ndim == 3: 
                channel = stretched_array_float[...,0]
            else:
                channel = stretched_array_float
            
            c_min, c_max = np.min(channel), np.max(channel)
            denominator = c_max - c_min
            if denominator > 1e-9:
                result_channel = ((channel - c_min) / denominator) * 255.0
            else:
                result_channel = channel 

            if img_array.ndim == 3: 
                stretched_array_float[...,0] = result_channel
            else: 
                stretched_array_float = result_channel
        else:
            messagebox.showerror("Error", "Unsupported image format for histogram stretching. Must be RGB or Grayscale.")
            return

        self.edited_image = self._convert_to_pil(stretched_array_float) 
        if self.edited_image: self._update_edited_display()

    def apply_average_filter(self):
        if not self._ensure_image_loaded(): return
        img_array = self._convert_to_array(self.edited_image)
        if img_array is None: return
        kernel = np.ones((3, 3), dtype=np.float64) / 9.0
        result_array = _convolve_manual(img_array, kernel)
        self.edited_image = self._convert_to_pil(result_array)
        if self.edited_image: self._update_edited_display()

    def apply_laplacian_filter(self):
        if not self._ensure_image_loaded(): return
        img_array = self._convert_to_array(self.edited_image)
        if img_array is None: return

        kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=np.float64)
        gray_array = None
        if img_array.ndim == 3 and img_array.shape[2] == 3: 
            gray_array = np.mean(img_array, axis=2) 
        elif img_array.ndim == 2: 
            gray_array = img_array.astype(np.float64) 
        elif img_array.ndim == 3 and img_array.shape[2] == 1:
            gray_array = img_array[:,:,0].astype(np.float64) 
        else:
            messagebox.showerror("Error", "Laplacian filter supports grayscale or RGB images.")
            return
            
        result_array = _convolve_manual(gray_array, kernel) 
        self.edited_image = self._convert_to_pil(result_array) 
        if self.edited_image: self._update_edited_display()

    def apply_minimum_filter(self):
        if not self._ensure_image_loaded(): return
        img_array = self._convert_to_array(self.edited_image)
        if img_array is None: return
        result_array = _filter_neighborhood_manual(img_array, size=3, operation='min')
        self.edited_image = self._convert_to_pil(result_array)
        if self.edited_image: self._update_edited_display()

    def apply_maximum_filter(self):
        if not self._ensure_image_loaded(): return
        img_array = self._convert_to_array(self.edited_image)
        if img_array is None: return
        result_array = _filter_neighborhood_manual(img_array, size=3, operation='max')
        self.edited_image = self._convert_to_pil(result_array)
        if self.edited_image: self._update_edited_display()

    def apply_median_filter(self):
        if not self._ensure_image_loaded(): return
        img_array = self._convert_to_array(self.edited_image)
        if img_array is None: return 
        result_array = _filter_neighborhood_manual(img_array, size=3, operation='median')
        self.edited_image = self._convert_to_pil(result_array)
        if self.edited_image: self._update_edited_display()

    def segment_image(self): 
        if not self._ensure_image_loaded(): return
        img_array = self._convert_to_array(self.edited_image) 
        if img_array is None: return

        img_cv_gray = None
        if img_array.ndim == 3 and img_array.shape[2] == 3:
            img_cv_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        elif img_array.ndim == 2: 
            img_cv_gray = img_array
        elif img_array.ndim == 3 and img_array.shape[2] == 1: 
            img_cv_gray = img_array[:,:,0]
        else:
            messagebox.showerror("Error", "Segmentation (Otsu) requires a grayscale or RGB image.")
            return
            
        _, segmented_cv = cv2.threshold(img_cv_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        self.edited_image = self._convert_to_pil(segmented_cv)
        if self.edited_image: self._update_edited_display()
        
    def global_thresholding(self):
        if not self._ensure_image_loaded(): return 
        img_array = self._convert_to_array(self.edited_image)
        if img_array is None: return

        img_cv_gray = None
        if img_array.ndim == 3 and img_array.shape[2] == 3:
            img_cv_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        elif img_array.ndim == 2:
            img_cv_gray = img_array
        elif img_array.ndim == 3 and img_array.shape[2] == 1:
            img_cv_gray = img_array[:,:,0]
        else:
            messagebox.showerror("Error", "Global thresholding requires a grayscale or RGB image.")
            return

        T = 127.0 
        epsilon = 1.0 
        max_iterations = 100
        prev_T = 0.0
        iteration = 0

        for _ in range(max_iterations): 
            if abs(T - prev_T) < epsilon:
                break
            
            G1 = img_cv_gray[img_cv_gray > T]
            G2 = img_cv_gray[img_cv_gray <= T]

            if len(G1) == 0 or len(G2) == 0: 
                T_otsu, _ = cv2.threshold(img_cv_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                T = float(T_otsu) 
                break

            m1 = np.mean(G1) if len(G1) > 0 else T 
            m2 = np.mean(G2) if len(G2) > 0 else T 
            
            prev_T = T
            T = (m1 + m2) / 2.0 
            iteration += 1 
        
        _, segmented = cv2.threshold(img_cv_gray, int(round(T)), 255, cv2.THRESH_BINARY) 
        self.edited_image = self._convert_to_pil(segmented) 
        if self.edited_image: self._update_edited_display() 
    
    def automatic_thresholding(self): 
        if not self._ensure_image_loaded(): return
        img_array = self._convert_to_array(self.edited_image)
        if img_array is None: return

        gray_img = None
        if img_array.ndim == 3 and img_array.shape[2] == 3:
            gray_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        elif img_array.ndim == 2:
            gray_img = img_array
        elif img_array.ndim == 3 and img_array.shape[2] == 1:
             gray_img = img_array[:,:,0]
        else:
            messagebox.showerror("Error", "Automatic thresholding (Otsu) requires a grayscale or RGB image.")
            return

        hist = np.histogram(gray_img, bins=256, range=(0,256))[0] 
        total_pixels = gray_img.size
        
        current_max_variance = 0
        threshold = 0
        
        sum_total = np.dot(np.arange(256), hist)

        weight_background = 0.0
        sum_background = 0.0

        for t in range(256):
            weight_background += hist[t]
            if weight_background == 0:
                continue

            weight_foreground = total_pixels - weight_background
            if weight_foreground == 0:
                break 

            sum_background += t * hist[t]
            
            mean_background = sum_background / weight_background
            mean_foreground = (sum_total - sum_background) / weight_foreground
            
            variance_between_classes = weight_background * weight_foreground * (mean_background - mean_foreground) ** 2
            
            if variance_between_classes > current_max_variance:
                current_max_variance = variance_between_classes
                threshold = t

        binary = (gray_img > threshold).astype(np.uint8) * 255
        self.edited_image = self._convert_to_pil(binary) 
        if self.edited_image: self._update_edited_display() 

    def adaptive_thresholding(self):
        if not self._ensure_original_image_loaded(): return 
        
        img_array = self._convert_to_array(self.original_image.copy()) 
        if img_array is None: return

        img_cv_gray = None
        if img_array.ndim == 3 and img_array.shape[2] == 3:
            img_cv_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        elif img_array.ndim == 2:
            img_cv_gray = img_array
        elif img_array.ndim == 3 and img_array.shape[2] == 1:
            img_cv_gray = img_array[:,:,0]
        else:
            messagebox.showerror("Error", "Adaptive thresholding requires a grayscale or RGB image.")
            return
        
        thresholded_cv = cv2.adaptiveThreshold(
            img_cv_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2 
        )
        
        self.edited_image = self._convert_to_pil(thresholded_cv)
        if self.edited_image:
            self._update_edited_display()
        else:
            messagebox.showerror("Error", "Failed to convert adaptive threshold result.")

    def image_dilation(self):
        if not self._ensure_image_loaded(): return
        img_array = self._convert_to_array(self.edited_image)
        if img_array is None: return
        result_img = _morphology_manual(img_array, kernel_size=(5,5), operation='dilate')
        self.edited_image = result_img 
        if self.edited_image: self._update_edited_display()

    def image_erosion(self):
        if not self._ensure_image_loaded(): return
        img_array = self._convert_to_array(self.edited_image)
        if img_array is None: return
        result_img = _morphology_manual(img_array, kernel_size=(5,5), operation='erode')
        self.edited_image = result_img 
        if self.edited_image: self._update_edited_display()

    def save_image(self):
        if self.edited_image is None:
            messagebox.showwarning("Warning", "No edited image to save.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg;*.jpeg"),
                        ("BMP files", "*.bmp"), ("TIFF files", "*.tif;*.tiff"),
                        ("All files", "*.*")]
        )
        if path:
            try:
                save_img = self.edited_image
                if save_img.mode == 'RGBA' and (path.lower().endswith(('.jpg', '.jpeg'))):
                    save_img = save_img.convert('RGB')
                save_img.save(path)
                messagebox.showinfo("Success", f"Image saved successfully to\n{path}")
            except Exception as e:
                messagebox.showerror("Error Saving Image", f"Could not save image: {e}")
            
    def mode_manual(self): 
        if not self._ensure_image_loaded(): return
        img_array = self._convert_to_array(self.edited_image)
        if img_array is None: return

        gray_img = None
        if img_array.ndim == 3 and img_array.shape[2] == 3:
            gray_img = np.mean(img_array, axis=2).astype(np.uint8)
        elif img_array.ndim == 2:
            gray_img = img_array.astype(np.uint8)
        elif img_array.ndim == 3 and img_array.shape[2] == 1: 
            gray_img = img_array[:,:,0].astype(np.uint8) 
        else:
            messagebox.showerror("Error", "Mode filter requires a grayscale or RGB image.")
            return
        h_gray, w_gray = gray_img.shape 
        result = np.zeros_like(gray_img)
        padded = _pad_image_manual(gray_img, pad_width=1, mode='edge') 

        for i in range(h_gray):      
            for j in range(w_gray):  
                window = padded[i:i+3, j:j+3].flatten() 
                
                if window.size == 0: continue
                counts = np.bincount(window) 
                mode_val = np.argmax(counts) 
                result[i, j] = mode_val

        self.edited_image = self._convert_to_pil(result)
        if self.edited_image: self._update_edited_display()

    def detect_outliers_manual(self): 
        if not self._ensure_image_loaded(): return
        img_array = self._convert_to_array(self.edited_image)
        if img_array is None: return

        processed_array = img_array.copy() 

        if img_array.ndim == 3 and img_array.shape[2] == 3: 
            for c in range(3): 
                channel = img_array[..., c]
                q1 = np.percentile(channel, 25)
                q3 = np.percentile(channel, 75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                
                channel_median = np.median(channel)
                outliers_low = channel < lower_bound
                outliers_high = channel > upper_bound
                processed_array[outliers_low, c] = channel_median
                processed_array[outliers_high, c] = channel_median

        elif img_array.ndim == 2 or (img_array.ndim == 3 and img_array.shape[2] == 1): 
            current_channel_data = None
            if img_array.ndim == 3: 
                current_channel_data = img_array[...,0]
            else: 
                current_channel_data = img_array

            q1 = np.percentile(current_channel_data, 25)
            q3 = np.percentile(current_channel_data, 75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            channel_median = np.median(current_channel_data)
            outliers_low = current_channel_data < lower_bound
            outliers_high = current_channel_data > upper_bound
            if img_array.ndim == 3: 
                processed_array[outliers_low, 0] = channel_median
                processed_array[outliers_high, 0] = channel_median
            else: 
                processed_array[outliers_low] = channel_median
                processed_array[outliers_high] = channel_median
        else:
            messagebox.showerror("Error", "Outlier detection supports grayscale or RGB images.")
            return
            
        self.edited_image = self._convert_to_pil(processed_array.astype(np.uint8))
        if self.edited_image: self._update_edited_display()

    def image_averaging_manual(self):
        if not self._ensure_image_loaded(): return
        if self.second_image is None:
            messagebox.showwarning("Warning", "Load a second image to perform averaging.")
            return
        img1_array = self._convert_to_array(self.edited_image)
        try:
            img2_pil = self.second_image.resize(self.edited_image.size).convert(self.edited_image.mode)
        except Exception as e:
            messagebox.showerror("Error", f"Could not make second image compatible: {e}")
            return
        img2_array = self._convert_to_array(img2_pil)
        if img1_array is None or img2_array is None:
            messagebox.showerror("Error", "Could not convert images to arrays for averaging.")
            return
        
        if img1_array.shape != img2_array.shape:
            messagebox.showerror("Error", "Images have incompatible shapes for averaging after conversion.")
            return
        averaged_array = ((img1_array.astype(np.float32) + img2_array.astype(np.float32)) / 2.0)
        self.edited_image = self._convert_to_pil(averaged_array) 
        if self.edited_image:
            self._update_edited_display()
        else:
            messagebox.showerror("Error", "Failed to average images or convert result.")
if __name__ == "__main__":
    root = tk.Tk()
    app = ImageEditorApp(root)
    root.mainloop()