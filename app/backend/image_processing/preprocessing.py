"""
Module: preprocessing.py
Chương 2: Xử lý ảnh - Tiền xử lý ảnh

Mô tả: Module chứa các thuật toán tiền xử lý ảnh cơ bản được học trong Chương 2.
Các kỹ thuật bao gồm:
- Resize: Thay đổi kích thước ảnh
- Gaussian Blur: Làm mờ Gaussian để giảm nhiễu
- Median Blur: Làm mờ trung vị
- Chuyển đổi RGB sang HSV
- Chuyển đổi sang ảnh xám (Grayscale)
- Histogram Equalization: Cân bằng histogram để cải thiện độ tương phản
- Noise Removal: Khử nhiễu
- Morphological Operations: Opening và Closing

Author: Tomato Quality System
"""

import cv2
import numpy as np
from typing import Tuple, Optional
import logging

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """
    Lớp xử lý tiền ảnh - Thực hiện các phép biến đổi ảnh cơ bản.
    
    Các phương thức tuân theo nguyên tắc:
    - Mỗi phương thức xử lý một khía cạnh của ảnh
    - Trả về ảnh đã xử lý cùng với metadata
    - Có logging để theo dõi quá trình xử lý
    """
    
    def __init__(self, target_size: Tuple[int, int] = (512, 512)):
        """
        Khởi tạo preprocessor với kích thước mặc định.
        
        Args:
            target_size: Kích thước ảnh đích (width, height)
        """
        self.target_size = target_size
        logger.info(f"ImagePreprocessor initialized with target size: {target_size}")
    
    def resize(self, image: np.ndarray, width: Optional[int] = None, 
               height: Optional[int] = None, 
               interpolation: int = cv2.INTER_LINEAR) -> Tuple[np.ndarray, dict]:
        """
        Thay đổi kích thước ảnh.
        
        Thuật toán: Sử dụng các phương pháp nội suy (interpolation) để
        tính giá trị pixel mới từ các pixel lân cận.
        
        - INTER_LINEAR: Bilinear interpolation (mặc định, cân bằng tốc độ và chất lượng)
        - INTER_CUBIC: Bicubic interpolation (chất lượng cao hơn, chậm hơn)
        - INTER_AREA: Áp dụng khi thu nhỏ ảnh
        
        Args:
            image: Ảnh đầu vào (BGR format từ OpenCV)
            width: Chiều rộng mới (None = giữ nguyên tỷ lệ)
            height: Chiều cao mới (None = giữ nguyên tỷ lệ)
            interpolation: Phương pháp nội suy
            
        Returns:
            Tuple chứa (ảnh đã resize, metadata)
        """
        original_height, original_width = image.shape[:2]
        
        # Nếu chỉ cung cấp một kích thước, tính kích thước còn lại theo tỷ lệ
        if width is None and height is None:
            width, height = self.target_size
        elif width is None:
            ratio = height / original_height
            width = int(original_width * ratio)
        elif height is None:
            ratio = width / original_width
            height = int(original_height * ratio)
        
        # Thay đổi kích thước
        resized = cv2.resize(image, (width, height), interpolation=interpolation)
        
        metadata = {
            "original_size": (original_width, original_height),
            "new_size": (width, height),
            "scale_factor": width / original_width,
            "interpolation": interpolation
        }
        
        logger.info(f"Resized image from {metadata['original_size']} to {metadata['new_size']}")
        return resized, metadata
    
    def gaussian_blur(self, image: np.ndarray, kernel_size: Tuple[int, int] = (5, 5),
                      sigma: float = 0) -> Tuple[np.ndarray, dict]:
        """
        Làm mờ ảnh bằng bộ lọc Gaussian.
        
        Thuật toán: Gaussian Blur sử dụng hàm Gaussian để tính trọng số cho
        các pixel lân cận. Pixel ở giữa có trọng số cao nhất, giảm dần
        ra xa theo phân bố Gaussian.
        
        Công thức: G(x,y) = (1/2πσ²) * e^(-(x²+y²)/2σ²)
        
        Kernel_size phải là số lẻ (ví dụ: 3, 5, 7)
        
        Args:
            image: Ảnh đầu vào
            kernel_size: Kích thước kernel (tuple lẻ)
            sigma: Độ lệch chuẩn của Gaussian (0 = tự động tính)
            
        Returns:
            Tuple chứa (ảnh đã làm mờ, metadata)
        """
        # Đảm bảo kernel_size là số lẻ
        if kernel_size[0] % 2 == 0:
            kernel_size = (kernel_size[0] + 1, kernel_size[1])
        if kernel_size[1] % 2 == 0:
            kernel_size = (kernel_size[0], kernel_size[1] + 1)
        
        blurred = cv2.GaussianBlur(image, kernel_size, sigma)
        
        metadata = {
            "kernel_size": kernel_size,
            "sigma": sigma,
            "description": "Gaussian Blur - Làm mờ theo phân bố Gaussian"
        }
        
        logger.info(f"Applied Gaussian blur with kernel {kernel_size}")
        return blurred, metadata
    
    def median_blur(self, image: np.ndarray, kernel_size: int = 5) -> Tuple[np.ndarray, dict]:
        """
        Làm mờ ảnh bằng bộ lọc trung vị (Median Filter).
        
        Thuật toán: Với mỗi pixel, thay thế giá trị bằng trung vị của
        các pixel trong cửa sổ kernel. Rất hiệu quả trong việc loại bỏ
        nhiễu salt-and-pepper (nhiễu đốm).
        
        Ưu điểm: Bảo toàn đường nét tốt hơn Gaussian blur
        
        Args:
            image: Ảnh đầu vào
            kernel_size: Kích thước kernel (phải là số lẻ)
            
        Returns:
            Tuple chứa (ảnh đã làm mờ, metadata)
        """
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        blurred = cv2.medianBlur(image, kernel_size)
        
        metadata = {
            "kernel_size": kernel_size,
            "description": "Median Blur - Lọc trung vị, hiệu quả với nhiễu đốm"
        }
        
        logger.info(f"Applied median blur with kernel size {kernel_size}")
        return blurred, metadata
    
    def bilateral_filter(self, image: np.ndarray, d: int = 9,
                         sigma_color: float = 75, 
                         sigma_space: float = 75) -> Tuple[np.ndarray, dict]:
        """
        Lọc bilateral - làm mờ nhưng giữ biên cạnh sắc nét.
        
        Thuật toán: Kết hợp cả hai yếu tố:
        1. Similarity weight (theo giá trị pixel)
        2. Spatial weight (theo khoảng cách)
        
        Hiệu quả trong việc giảm nhiễu trong khi vẫn bảo toàn edges.
        
        Args:
            image: Ảnh đầu vào
            d: Đường kính pixel lân cận
            sigma_color: Độ lệch chuẩn trong không gian màu
            sigma_space: Độ lệch chuẩn trong không gian tọa độ
            
        Returns:
            Tuple chứa (ảnh đã lọc, metadata)
        """
        filtered = cv2.bilateralFilter(image, d, sigma_color, sigma_space)
        
        metadata = {
            "d": d,
            "sigma_color": sigma_color,
            "sigma_space": sigma_space,
            "description": "Bilateral Filter - Giữ edges, giảm nhiễu"
        }
        
        logger.info(f"Applied bilateral filter with d={d}")
        return filtered, metadata
    
    def convert_to_hsv(self, image: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Chuyển đổi ảnh từ RGB sang HSV.
        
        Thuật toán: HSV (Hue, Saturation, Value) là không gian màu mô tả:
        - Hue: Sắc thái màu (0-180 trong OpenCV)
        - Saturation: Độ bão hòa màu
        - Value: Độ sáng/Giá trị
        
        Ứng dụng trong phân tách màu sắc:
        - Màu đỏ của cà chua chín: H thường trong khoảng 0-10 hoặc 170-180
        - Màu xanh của cà chua chưa chín: H trong khoảng 30-90
        
        Args:
            image: Ảnh đầu vào (BGR từ OpenCV, chuyển sang RGB trước)
            
        Returns:
            Tuple chứa (ảnh HSV, metadata)
        """
        # OpenCV dùng BGR, chuyển sang RGB trước khi chuyển sang HSV
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
        
        metadata = {
            "original_color_space": "RGB",
            "new_color_space": "HSV",
            "h_range": "0-180",
            "s_range": "0-255",
            "v_range": "0-255",
            "description": "Chuyển đổi RGB sang HSV cho phân tách màu sắc"
        }
        
        logger.info("Converted RGB to HSV")
        return hsv_image, metadata
    
    def convert_to_grayscale(self, image: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Chuyển đổi ảnh màu sang ảnh xám (Grayscale).
        
        Thuật toán: Sử dụng công thức trọng số luminance:
        Y = 0.299*R + 0.587*G + 0.114*B
        
        Các trọng số này phản ánh độ nhạy của mắt người với các màu khác nhau.
        
        Args:
            image: Ảnh đầu vào (BGR)
            
        Returns:
            Tuple chứa (ảnh xám, metadata)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        metadata = {
            "original_channels": 3 if len(image.shape) == 3 else 1,
            "new_channels": 1,
            "formula": "Y = 0.299*R + 0.587*G + 0.114*B",
            "description": "Chuyển đổi sang ảnh xám theo công thức luminance"
        }
        
        logger.info("Converted image to grayscale")
        return gray, metadata
    
    def histogram_equalization(self, image: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Cân bằng histogram để cải thiện độ tương phản.
        
        Thuật toán: CLAHE (Contrast Limited Adaptive Histogram Equalization)
        
        1. Chia ảnh thành các vùng nhỏ (tiles)
        2. Cân bằng histogram cho từng vùng
        3. Áp dụng bilinear interpolation để trộn các kết quả
        
        Ưu điểm: Tránh hiện tượng "over-enhancement" do giới hạn độ tương phản
        
        Args:
            image: Ảnh đầu vào (grayscale)
            
        Returns:
            Tuple chứa (ảnh đã cân bằng, metadata)
        """
        # Nếu là ảnh màu, chuyển sang xám trước
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Tạo CLAHE object với clipLimit và tileGridSize
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        equalized = clahe.apply(image)
        
        metadata = {
            "clipLimit": 2.0,
            "tileGridSize": (8, 8),
            "algorithm": "CLAHE",
            "description": "Contrast Limited Adaptive Histogram Equalization"
        }
        
        logger.info("Applied CLAHE histogram equalization")
        return equalized, metadata
    
    def morphological_opening(self, image: np.ndarray, 
                             kernel_size: Tuple[int, int] = (5, 5)) -> Tuple[np.ndarray, dict]:
        """
        Morphological Opening: Erosion sau đó Dilation.
        
        Thuật toán:
        1. Erosion: Co lại các vùng sáng, loại bỏ noise nhỏ
        2. Dilation: Giãn nở trở lại để phục hồi kích thước đối tượng
        
        Ứng dụng: Loại bỏ nhiễu nhỏ (salt noise) mà không làm thay đổi
        đáng kể kích thước đối tượng chính.
        
        Args:
            image: Ảnh đầu vào (binary hoặc grayscale)
            kernel_size: Kích thước kernel
            
        Returns:
            Tuple chứa (ảnh đã xử lý, metadata)
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
        
        # Opening = Erosion -> Dilation
        opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        
        metadata = {
            "operation": "Opening",
            "kernel_type": "ELLIPSE",
            "kernel_size": kernel_size,
            "steps": ["Erosion", "Dilation"],
            "description": "Loại bỏ noise nhỏ, giữ nguyên kích thước đối tượng"
        }
        
        logger.info(f"Applied morphological opening with kernel {kernel_size}")
        return opened, metadata
    
    def morphological_closing(self, image: np.ndarray,
                             kernel_size: Tuple[int, int] = (5, 5)) -> Tuple[np.ndarray, dict]:
        """
        Morphological Closing: Dilation sau đó Erosion.
        
        Thuật toán:
        1. Dilation: Giãn nở các vùng sáng, lấp đầy holes nhỏ
        2. Erosion: Co lại để phục hồi kích thước
        
        Ứng dụng: Lấp đầy các lỗ hổng nhỏ trong đối tượng, loại bỏ
        noise nhỏ (pepper noise).
        
        Args:
            image: Ảnh đầu vào (binary hoặc grayscale)
            kernel_size: Kích thước kernel
            
        Returns:
            Tuple chứa (ảnh đã xử lý, metadata)
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
        
        # Closing = Dilation -> Erosion
        closed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        
        metadata = {
            "operation": "Closing",
            "kernel_type": "ELLIPSE",
            "kernel_size": kernel_size,
            "steps": ["Dilation", "Erosion"],
            "description": "Lấp đầy holes nhỏ trong đối tượng"
        }
        
        logger.info(f"Applied morphological closing with kernel {kernel_size}")
        return closed, metadata
    
    def remove_noise(self, image: np.ndarray, method: str = "gaussian") -> Tuple[np.ndarray, dict]:
        """
        Khử nhiễu cho ảnh.
        
        Các phương pháp:
        1. "gaussian": Gaussian Blur - Làm mờ theo phân bố Gaussian
        2. "median": Median Filter - Lọc trung vị
        3. "bilateral": Bilateral Filter - Giữ edges
        4. "nlm": Non-Local Means - Khử nhiễu thích nghi
        
        Args:
            image: Ảnh đầu vào
            method: Phương pháp khử nhiễu
            
        Returns:
            Tuple chứa (ảnh đã khử nhiễu, metadata)
        """
        if method == "gaussian":
            denoised, _ = self.gaussian_blur(image, (5, 5))
        elif method == "median":
            denoised, _ = self.median_blur(image, 5)
        elif method == "bilateral":
            denoised, _ = self.bilateral_filter(image)
        elif method == "nlm":
            # Non-Local Means Denoising
            if len(image.shape) == 3:
                denoised = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
            else:
                denoised = cv2.fastNlMeansDenoising(image, None, 10, 7, 21)
        else:
            raise ValueError(f"Unknown noise removal method: {method}")
        
        metadata = {
            "method": method,
            "description": f"Khử nhiễu bằng phương pháp {method}"
        }
        
        logger.info(f"Removed noise using {method} method")
        return denoised, metadata
    
    def normalize_image(self, image: np.ndarray, 
                       target_range: Tuple[float, float] = (0, 255)) -> Tuple[np.ndarray, dict]:
        """
        Chuẩn hóa giá trị pixel về một khoảng xác định.
        
        Args:
            image: Ảnh đầu vào
            target_range: Khoảng giá trị mục tiêu (min, max)
            
        Returns:
            Tuple chứa (ảnh đã chuẩn hóa, metadata)
        """
        normalized = cv2.normalize(image, None, target_range[0], target_range[1], 
                                   cv2.NORM_MINMAX)
        
        metadata = {
            "target_range": target_range,
            "description": "Chuẩn hóa giá trị pixel về khoảng [0, 255]"
        }
        
        logger.info(f"Normalized image to range {target_range}")
        return normalized, metadata
    
    def preprocess_pipeline(self, image: np.ndarray) -> Tuple[np.ndarray, list]:
        """
        Pipeline tiền xử lý hoàn chỉnh cho phân tích cà chua.
        
        Các bước:
        1. Resize về kích thước chuẩn
        2. Khử nhiễu bằng bilateral filter
        3. Chuyển sang HSV cho phân tích màu sắc
        
        Args:
            image: Ảnh đầu vào
            
        Returns:
            Tuple chứa (ảnh đã xử lý, danh sách các bước đã thực hiện)
        """
        steps = []
        
        # Bước 1: Resize
        processed, meta = self.resize(image)
        steps.append({"step": "resize", "result": processed.copy(), "metadata": meta})
        
        # Bước 2: Khử nhiễu (Bilateral giữ edges)
        processed, meta = self.bilateral_filter(processed)
        steps.append({"step": "denoise", "result": processed.copy(), "metadata": meta})
        
        # Bước 3: Chuyển sang HSV
        hsv_image, meta = self.convert_to_hsv(processed)
        steps.append({"step": "hsv", "result": hsv_image.copy(), "metadata": meta})
        
        logger.info("Completed full preprocessing pipeline")
        return hsv_image, steps


def apply_custom_kernel(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Áp dụng kernel tự chọn cho ảnh (Convolution).
    
    Args:
        image: Ảnh đầu vào
        kernel: Ma trận kernel
        
    Returns:
        Ảnh đã convolve
    """
    return cv2.filter2D(image, -1, kernel)


def create_sharpen_kernel() -> np.ndarray:
    """
    Tạo kernel làm sắc nét (Sharpening).
    
    Returns:
        Kernel sharpening 3x3
    """
    return np.array([
        [-1, -1, -1],
        [-1,  9, -1],
        [-1, -1, -1]
    ], dtype=np.float32)


def create_edge_detection_kernel() -> np.ndarray:
    """
    Tạo kernel phát hiện cạnh (Laplacian approximation).
    
    Returns:
        Kernel Laplacian
    """
    return np.array([
        [0,  1, 0],
        [1, -4, 1],
        [0,  1, 0]
    ], dtype=np.float32)


if __name__ == "__main__":
    # Test module
    import sys
    
    # Tạo ảnh test ngẫu nhiên
    test_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    
    preprocessor = ImagePreprocessor(target_size=(512, 512))
    
    # Test các phương thức
    print("Testing preprocessing module...")
    
    resized, _ = preprocessor.resize(test_image)
    print(f"Resize: {resized.shape}")
    
    blurred, _ = preprocessor.gaussian_blur(test_image)
    print(f"Gaussian Blur: {blurred.shape}")
    
    gray, _ = preprocessor.convert_to_grayscale(test_image)
    print(f"Grayscale: {gray.shape}")
    
    hsv, _ = preprocessor.convert_to_hsv(test_image)
    print(f"HSV: {hsv.shape}")
    
    print("All tests passed!")
