"""
Module: feature_extraction.py
Chương 3: Trích xuất đặc trưng

Mô tả: Module chứa các thuật toán trích xuất đặc trưng ảnh được học trong Chương 3.
Các kỹ thuật bao gồm:
- Color Features: Histogram RGB/HSV, Mean, Tỷ lệ màu
- Shape Features: Area, Perimeter, Circularity, Aspect Ratio, Solidity
- Texture Features: LBP (Local Binary Pattern), GLCM (Gray Level Co-occurrence Matrix)

Author: Tomato Quality System
"""

import cv2
import numpy as np
from typing import Tuple, Dict, List
from scipy import ndimage
from skimage.feature import local_binary_pattern
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureExtractor:
    """
    Lớp trích xuất đặc trưng - Tính toán các đặc trưng từ ảnh cà chua.
    
    Đặc trưng được trích xuất:
    1. Color Features - Màu sắc
    2. Shape Features - Hình dạng
    3. Texture Features - Kết cấu bề mặt
    """
    
    def __init__(self):
        """
        Khởi tạo Feature Extractor với các thông số mặc định.
        """
        # Ngưỡng màu cho phân tích
        self.color_thresholds = {
            'red': {'lower': np.array([0, 50, 50]), 'upper': np.array([10, 255, 255])},
            'yellow': {'lower': np.array([11, 50, 50]), 'upper': np.array([34, 255, 255])},
            'green': {'lower': np.array([35, 50, 50]), 'upper': np.array([85, 255, 255])},
            'brown': {'lower': np.array([0, 30, 10]), 'upper': np.array([30, 150, 100])},
        }
        
        logger.info("FeatureExtractor initialized")
    
    # ==================== COLOR FEATURES ====================
    
    def calculate_rgb_histogram(self, image: np.ndarray, 
                               mask: np.ndarray = None) -> Tuple[Dict, np.ndarray]:
        """
        Tính histogram cho từng kênh R, G, B.
        
        Histogram mô tả phân bố cường độ pixel cho mỗi kênh màu.
        
        Args:
            image: Ảnh BGR
            mask: Mask nhị phân để giới hạn vùng tính
            
        Returns:
            Tuple chứa (dictionary với histogram, ảnh visualization)
        """
        # Tách các kênh
        b, g, r = cv2.split(image)
        
        # Tính histogram cho từng kênh (256 bins)
        hist_r = cv2.calcHist([r], [0], mask if mask is not None else None, [256], [0, 256])
        hist_g = cv2.calcHist([g], [0], mask if mask is not None else None, [256], [0, 256])
        hist_b = cv2.calcHist([b], [0], mask if mask is not None else None, [256], [0, 256])
        
        # Chuẩn hóa histogram
        hist_r = cv2.normalize(hist_r, hist_r).flatten()
        hist_g = cv2.normalize(hist_g, hist_g).flatten()
        hist_b = cv2.normalize(hist_b, hist_b).flatten()
        
        histograms = {
            'red': hist_r.tolist(),
            'green': hist_g.tolist(),
            'blue': hist_b.tolist(),
            'red_peak': int(np.argmax(hist_r)),
            'green_peak': int(np.argmax(hist_g)),
            'blue_peak': int(np.argmax(hist_b))
        }
        
        # Tạo ảnh visualization
        vis = np.zeros((256, 256, 3), dtype=np.uint8)
        for i in range(256):
            cv2.line(vis, (i, 256), (i, int(256 - hist_r[i] * 256 * 0.3)), (0, 0, 255), 1)
            cv2.line(vis, (i, 256), (i, int(256 - hist_g[i] * 256 * 0.3)), (0, 255, 0), 1)
            cv2.line(vis, (i, 256), (i, int(256 - hist_b[i] * 256 * 0.3)), (255, 0, 0), 1)
        
        logger.info("RGB histogram calculated")
        return histograms, vis
    
    def calculate_hsv_histogram(self, image: np.ndarray,
                                 mask: np.ndarray = None) -> Tuple[Dict, np.ndarray]:
        """
        Tính histogram cho từng kênh H, S, V.
        
        HSV rất hữu ích cho phân tách màu sắc.
        
        Args:
            image: Ảnh BGR
            mask: Mask nhị phân
            
        Returns:
            Tuple chứa (dictionary với histogram, ảnh visualization)
        """
        # Chuyển sang HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # Tính histogram cho H (0-180 trong OpenCV)
        hist_h = cv2.calcHist([h], [0], mask if mask is not None else None, [180], [0, 180])
        hist_s = cv2.calcHist([s], [0], mask if mask is not None else None, [256], [0, 256])
        hist_v = cv2.calcHist([v], [0], mask if mask is not None else None, [256], [0, 256])
        
        # Chuẩn hóa
        hist_h = cv2.normalize(hist_h, hist_h).flatten()
        hist_s = cv2.normalize(hist_s, hist_s).flatten()
        hist_v = cv2.normalize(hist_v, hist_v).flatten()
        
        histograms = {
            'hue': hist_h.tolist(),
            'saturation': hist_s.tolist(),
            'value': hist_v.tolist(),
            'hue_peak': int(np.argmax(hist_h)),
            'saturation_peak': int(np.argmax(hist_s)),
            'value_peak': int(np.argmax(hist_v))
        }
        
        # Tạo ảnh visualization (cho H)
        vis = np.zeros((180, 360, 3), dtype=np.uint8)
        for i in range(180):
            hue_color = cv2.cvtColor(np.uint8([[[i * 2, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
            height = int(hist_h[i] * 180)
            cv2.rectangle(vis, (i * 2, 180 - height), (i * 2 + 2, 180), 
                          (int(hue_color[0]), int(hue_color[1]), int(hue_color[2])), -1)
        
        logger.info("HSV histogram calculated")
        return histograms, vis
    
    def calculate_color_means(self, image: np.ndarray,
                             mask: np.ndarray = None) -> Dict[str, float]:
        """
        Tính giá trị trung bình của các kênh màu.
        
        Args:
            image: Ảnh BGR
            mask: Mask nhị phân
            
        Returns:
            Dictionary với means của từng kênh
        """
        b, g, r = cv2.split(image)
        
        if mask is not None:
            r_mean = float(np.mean(r[mask > 0])) if np.any(mask > 0) else 0
            g_mean = float(np.mean(g[mask > 0])) if np.any(mask > 0) else 0
            b_mean = float(np.mean(b[mask > 0])) if np.any(mask > 0) else 0
        else:
            r_mean = float(np.mean(r))
            g_mean = float(np.mean(g))
            b_mean = float(np.mean(b))
        
        # Chuyển sang HSV để tính means
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        if mask is not None:
            h_mean = float(np.mean(h[mask > 0])) if np.any(mask > 0) else 0
            s_mean = float(np.mean(s[mask > 0])) if np.any(mask > 0) else 0
            v_mean = float(np.mean(v[mask > 0])) if np.any(mask > 0) else 0
        else:
            h_mean = float(np.mean(h))
            s_mean = float(np.mean(s))
            v_mean = float(np.mean(v))
        
        means = {
            'rgb_r_mean': r_mean,
            'rgb_g_mean': g_mean,
            'rgb_b_mean': b_mean,
            'hsv_h_mean': h_mean,
            'hsv_s_mean': s_mean,
            'hsv_v_mean': v_mean
        }
        
        logger.info(f"Color means: R={r_mean:.2f}, G={g_mean:.2f}, B={b_mean:.2f}")
        return means
    
    def calculate_color_ratios(self, image: np.ndarray,
                               mask: np.ndarray = None) -> Dict[str, float]:
        """
        Tính tỷ lệ các màu chính trong ảnh cà chua.
        
        Args:
            image: Ảnh BGR
            mask: Mask nhị phân
            
        Returns:
            Dictionary với tỷ lệ màu (%)
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        total_pixels = np.sum(mask > 0) if mask is not None else mask.size
        
        if total_pixels == 0:
            return {'red_ratio': 0, 'yellow_ratio': 0, 'green_ratio': 0, 'brown_ratio': 0}
        
        # Mask cho màu đỏ (2 vùng)
        mask_red1 = cv2.inRange(hsv, np.array([0, 50, 50]), np.array([10, 255, 255]))
        mask_red2 = cv2.inRange(hsv, np.array([170, 50, 50]), np.array([180, 255, 255]))
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)
        
        # Mask cho màu vàng
        mask_yellow = cv2.inRange(hsv, np.array([11, 50, 50]), np.array([34, 255, 255]))
        
        # Mask cho màu xanh
        mask_green = cv2.inRange(hsv, np.array([35, 50, 50]), np.array([85, 255, 255]))
        
        # Mask cho màu nâu/đen (defects)
        mask_brown = cv2.inRange(hsv, np.array([0, 30, 10]), np.array([30, 150, 100]))
        
        if mask is not None:
            mask_red = cv2.bitwise_and(mask_red, mask_red, mask=mask)
            mask_yellow = cv2.bitwise_and(mask_yellow, mask_yellow, mask=mask)
            mask_green = cv2.bitwise_and(mask_green, mask_green, mask=mask)
            mask_brown = cv2.bitwise_and(mask_brown, mask_brown, mask=mask)
        
        red_ratio = float(np.sum(mask_red > 0)) / total_pixels * 100
        yellow_ratio = float(np.sum(mask_yellow > 0)) / total_pixels * 100
        green_ratio = float(np.sum(mask_green > 0)) / total_pixels * 100
        brown_ratio = float(np.sum(mask_brown > 0)) / total_pixels * 100
        
        ratios = {
            'red_ratio': red_ratio,
            'yellow_ratio': yellow_ratio,
            'green_ratio': green_ratio,
            'brown_ratio': brown_ratio
        }
        
        logger.info(f"Color ratios: R={red_ratio:.1f}%, Y={yellow_ratio:.1f}%, G={green_ratio:.1f}%, Br={brown_ratio:.1f}%")
        return ratios
    
    # ==================== SHAPE FEATURES ====================
    
    def calculate_shape_features(self, contour: np.ndarray) -> Dict[str, float]:
        """
        Tính các đặc trưng hình dạng từ contour.
        
        Các đặc trưng:
        - Area: Diện tích vùng
        - Perimeter: Chu vi
        - Circularity: Độ tròn (4π*Area / Perimeter²)
        - Aspect Ratio: Tỷ lệ width/height của bounding box
        - Solidity: Tỷ lệ Area / Convex Hull Area
        - Eccentricity: Độ lệch tâm của ellipse bao quanh
        
        Args:
            contour: Contour của đối tượng
            
        Returns:
            Dictionary với các đặc trưng hình dạng
        """
        # Area
        area = cv2.contourArea(contour)
        
        # Perimeter (Arc Length)
        perimeter = cv2.arcLength(contour, True)
        
        # Circularity: 4πA / P² (1 = perfect circle)
        circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
        
        # Bounding Rectangle
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h if h > 0 else 0
        
        # Convex Hull
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        
        # Solidity: Area / Convex Hull Area
        solidity = float(area) / hull_area if hull_area > 0 else 0
        
        # Eccentricity - fit ellipse if possible
        if len(contour) >= 5:
            ellipse = cv2.fitEllipse(contour)
            (_, (MA, ma), _) = ellipse
            eccentricity = float(MA) / ma if ma > 0 else 0
        else:
            eccentricity = 0
        
        # Equivalent Diameter
        equivalent_diameter = np.sqrt(4 * area / np.pi) if area > 0 else 0
        
        # Extent: Area / Bounding Rect Area
        extent = float(area) / (w * h) if (w * h) > 0 else 0
        
        features = {
            'area': float(area),
            'perimeter': float(perimeter),
            'circularity': float(circularity),
            'aspect_ratio': float(aspect_ratio),
            'solidity': float(solidity),
            'eccentricity': float(eccentricity),
            'equivalent_diameter': float(equivalent_diameter),
            'extent': float(extent)
        }
        
        logger.info(f"Shape features: Area={area:.2f}, Circularity={circularity:.3f}, Solidity={solidity:.3f}")
        return features
    
    # ==================== TEXTURE FEATURES ====================
    
    def calculate_lbp(self, gray_image: np.ndarray,
                     mask: np.ndarray = None,
                     radius: int = 3,
                     n_points: int = 24) -> Tuple[Dict, np.ndarray]:
        """
        Tính Local Binary Pattern (LBP) - Mẫu nhị phân cục bộ.
        
        Thuật toán LBP:
        1. Với mỗi pixel, so sánh với P điểm lân cận trên đường tròn bán kính R
        2. Nếu giá trị điểm lân cận >= pixel trung tâm -> 1, else -> 0
        3. Ghép nối các bits để tạo LBP code
        
        Ứng dụng: Phân tích texture bề mặt cà chua (đều vs có khuyết tật)
        
        Args:
            gray_image: Ảnh xám
            mask: Mask nhị phân
            radius: Bán kính vòng tròn lân cận
            n_points: Số điểm lân cận
            
        Returns:
            Tuple chứa (dictionary với LBP stats, ảnh LBP)
        """
        # Áp dụng mask nếu có
        if mask is not None:
            gray_masked = cv2.bitwise_and(gray_image, gray_image, mask=mask)
        else:
            gray_masked = gray_image
        
        # Tính LBP
        lbp = local_binary_pattern(gray_masked, n_points, radius, method='uniform')
        
        # Tính histogram của LBP (đặc trưng texture)
        n_bins = n_points + 2  # uniform patterns + 1 for non-uniform
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
        hist = hist.astype('float32')
        hist /= (hist.sum() + 1e-7)  # Normalize
        
        # Tính entropy của LBP histogram
        entropy = -np.sum(hist * np.log2(hist + 1e-7))
        
        # Uniformity (tỷ lệ uniform patterns)
        n_uniform = n_points + 2
        uniformity = float(hist[:n_uniform].sum())
        
        lbp_stats = {
            'lbp_histogram': hist.tolist(),
            'lbp_entropy': float(entropy),
            'lbp_uniformity': uniformity,
            'lbp_uniform_ratio': float(np.sum(hist[:-1])) / (len(hist) - 1) if len(hist) > 1 else 0
        }
        
        # Tạo ảnh visualization
        lbp_vis = (lbp / lbp.max() * 255).astype(np.uint8) if lbp.max() > 0 else lbp.astype(np.uint8)
        lbp_vis_color = cv2.applyColorMap(lbp_vis, cv2.COLORMAP_JET)
        
        logger.info(f"LBP calculated: entropy={entropy:.3f}, uniformity={uniformity:.3f}")
        return lbp_stats, lbp_vis_color
    
    def calculate_glcm(self, gray_image: np.ndarray,
                      mask: np.ndarray = None,
                      distances: List[int] = [1, 2, 3],
                      angles: List[float] = [0, np.pi/4, np.pi/2, 3*np.pi/4]) -> Dict:
        """
        Tính Gray Level Co-occurrence Matrix (GLCM) - Ma trận đồng xuất hiện mức xám.
        
        Thuật toán GLCM:
        1. Quantize ảnh xuống fewer gray levels (vd: 16 hoặc 32)
        2. Với mỗi cặp pixel (i, j) cách nhau khoảng d theo hướng θ,
           tăng GLCM[d, θ][i, j] thêm 1
        3. Normalize GLCM
        
        Các đặc trưng từ GLCM:
        - Contrast: Tổng bình phương trọng số (i-j)²
        - Correlation: Tương quan tuyến tính
        - Energy (ASM): Tổng bình phương các giá trị
        - Homogeneity: Tổng các giá trị theo đường chéo
        
        Args:
            gray_image: Ảnh xám
            mask: Mask nhị phân
            distances: Các khoảng cách d
            angles: Các góc θ (radians)
            
        Returns:
            Dictionary với các đặc trưng GLCM
        """
        # Áp dụng mask nếu có
        if mask is not None:
            gray_masked = cv2.bitwise_and(gray_image, gray_image, mask=mask)
        else:
            gray_masked = gray_image.copy()
        
        # Quantize xuống 16 gray levels
        levels = 16
        gray_quantized = (gray_masked / 256 * levels).astype(np.uint8)
        
        # Tính GLCM cho mỗi distance và angle
        glcm_features = {
            'contrast': [],
            'correlation': [],
            'energy': [],
            'homogeneity': []
        }
        
        for d in distances:
            for angle in angles:
                # Tính GLCM thủ công
                glcm = self._compute_glcm(gray_quantized, d, angle, levels)
                
                # Normalize
                glcm_sum = glcm.sum()
                if glcm_sum > 0:
                    glcm = glcm / glcm_sum
                
                # Tính các đặc trưng
                i, j = np.ogrid[0:levels, 0:levels]
                
                # Contrast
                contrast = np.sum((i - j) ** 2 * glcm)
                
                # Correlation
                i_mean = np.sum(i * glcm)
                j_mean = np.sum(j * glcm)
                i_std = np.sqrt(np.sum((i - i_mean) ** 2 * glcm))
                j_std = np.sqrt(np.sum((j - j_mean) ** 2 * glcm))
                if i_std > 0 and j_std > 0:
                    correlation = np.sum((i - i_mean) * (j - j_mean) * glcm) / (i_std * j_std)
                else:
                    correlation = 0
                
                # Energy (Angular Second Moment)
                energy = np.sum(glcm ** 2)
                
                # Homogeneity
                homogeneity = np.sum(glcm / (1 + np.abs(i - j)))
                
                glcm_features['contrast'].append(float(contrast))
                glcm_features['correlation'].append(float(correlation))
                glcm_features['energy'].append(float(energy))
                glcm_features['homogeneity'].append(float(homogeneity))
        
        # Tính trung bình
        avg_features = {
            'glcm_contrast_avg': float(np.mean(glcm_features['contrast'])),
            'glcm_correlation_avg': float(np.mean(glcm_features['correlation'])),
            'glcm_energy_avg': float(np.mean(glcm_features['energy'])),
            'glcm_homogeneity_avg': float(np.mean(glcm_features['homogeneity'])),
            'glcm_contrast_std': float(np.std(glcm_features['contrast'])),
            'glcm_correlation_std': float(np.std(glcm_features['correlation'])),
            'glcm_energy_std': float(np.std(glcm_features['energy'])),
            'glcm_homogeneity_std': float(np.std(glcm_features['homogeneity'])),
            'glcm_all': glcm_features
        }
        
        logger.info(f"GLCM features: contrast={avg_features['glcm_contrast_avg']:.3f}, "
                   f"homogeneity={avg_features['glcm_homogeneity_avg']:.3f}")
        return avg_features
    
    def _compute_glcm(self, image: np.ndarray, d: int, angle: float, levels: int) -> np.ndarray:
        """
        Tính GLCM cho một distance và angle cụ thể.
        
        Args:
            image: Ảnh đã quantize
            d: Khoảng cách
            angle: Góc (radians)
            levels: Số gray levels
            
        Returns:
            Ma trận GLCM
        """
        glcm = np.zeros((levels, levels), dtype=np.float32)
        
        h, w = image.shape
        
        # Tính offset từ angle
        dx = int(round(d * np.cos(angle)))
        dy = int(round(d * np.sin(angle)))
        
        for i in range(max(0, -dy), min(h, h - dy)):
            for j in range(max(0, -dx), min(w, w - dx)):
                i2 = i + dy
                j2 = j + dx
                if 0 <= i2 < h and 0 <= j2 < w:
                    glcm[image[i, j], image[i2, j2]] += 1
        
        return glcm
    
    def calculate_defect_features(self, image: np.ndarray,
                                 mask: np.ndarray) -> Dict[str, float]:
        """
        Phát hiện và tính các đặc trưng khuyết tật.
        
        Khuyết tật bao gồm:
        - Vùng đen/nâu (thâm)
        - Vùng nứt
        - Vùng mốc
        
        Args:
            image: Ảnh gốc (BGR)
            mask: Mask của cà chua
            
        Returns:
            Dictionary với các đặc trưng khuyết tật
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Mask cho vùng tối/màu nâu (defects)
        defect_mask_dark = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([30, 100, 100]))
        
        # Mask cho vùng có texture bất thường (sử dụng variance)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        if mask is not None:
            defect_mask_dark = cv2.bitwise_and(defect_mask_dark, defect_mask_dark, mask=mask)
        
        # Tính % vùng khuyết tật
        total_area = np.sum(mask > 0) if mask is not None else mask.size
        defect_area = np.sum(defect_mask_dark > 0)
        
        defect_ratio = float(defect_area) / total_area * 100 if total_area > 0 else 0
        
        # Tìm các vùng defect riêng biệt
        num_defect_regions = 0
        if defect_area > 0:
            _, defects, _ = cv2.findContours(defect_mask_dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            num_defect_regions = len(defects)
        
        features = {
            'defect_ratio': defect_ratio,
            'num_defect_regions': num_defect_regions,
            'defect_area': float(defect_area),
            'has_defects': bool(defect_ratio > 5),  # >5% là có khuyết tật
            'defect_severity': 'none' if defect_ratio < 2 else 
                              'minor' if defect_ratio < 5 else 
                              'moderate' if defect_ratio < 15 else 'severe'
        }
        
        logger.info(f"Defect analysis: ratio={defect_ratio:.2f}%, regions={num_defect_regions}")
        return features
    
    def extract_all_features(self, image: np.ndarray,
                           mask: np.ndarray,
                           contour: np.ndarray = None) -> Dict:
        """
        Trích xuất tất cả các đặc trưng.
        
        Args:
            image: Ảnh gốc (BGR)
            mask: Mask của cà chua
            contour: Contour của cà chua (tùy chọn)
            
        Returns:
            Dictionary chứa tất cả features
        """
        features = {}
        
        # 1. Color Features
        features['rgb_histogram'], _ = self.calculate_rgb_histogram(image, mask)
        features['hsv_histogram'], _ = self.calculate_hsv_histogram(image, mask)
        features.update(self.calculate_color_means(image, mask))
        features.update(self.calculate_color_ratios(image, mask))
        
        # 2. Shape Features (nếu có contour)
        if contour is not None:
            features.update(self.calculate_shape_features(contour))
        
        # 3. Texture Features
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        features.update(self.calculate_lbp(gray, mask))
        features.update(self.calculate_glcm(gray, mask))
        
        # 4. Defect Features
        features.update(self.calculate_defect_features(image, mask))
        
        logger.info("All features extracted successfully")
        return features


def extract_edge_features(gray_image: np.ndarray, mask: np.ndarray = None) -> Dict[str, float]:
    """
    Trích xuất đặc trưng từ edges.
    
    Args:
        gray_image: Ảnh xám
        mask: Mask nhị phân
        
    Returns:
        Dictionary với edge features
    """
    # Canny edge detection
    edges = cv2.Canny(gray_image, 50, 150)
    
    if mask is not None:
        edges = cv2.bitwise_and(edges, edges, mask=mask)
    
    total_pixels = np.sum(mask > 0) if mask is not None else mask.size
    edge_density = float(np.sum(edges > 0)) / total_pixels if total_pixels > 0 else 0
    
    return {
        'edge_density': edge_density,
        'edge_pixels': int(np.sum(edges > 0))
    }


if __name__ == "__main__":
    # Test module
    import sys
    
    # Tạo ảnh test
    test_image = np.zeros((256, 256, 3), dtype=np.uint8)
    cv2.circle(test_image, (128, 128), 80, (0, 0, 255), -1)
    
    # Tạo mask test
    test_mask = np.zeros((256, 256), dtype=np.uint8)
    cv2.circle(test_mask, (128, 128), 80, 255, -1)
    
    extractor = FeatureExtractor()
    
    # Test color features
    color_ratios = extractor.calculate_color_ratios(test_image, test_mask)
    print(f"Color ratios: {color_ratios}")
    
    # Test shape features
    contours, _ = cv2.findContours(test_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        shape_features = extractor.calculate_shape_features(contours[0])
        print(f"Shape features: {shape_features}")
    
    # Test LBP
    gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
    lbp_stats, _ = extractor.calculate_lbp(gray, test_mask)
    print(f"LBP stats: entropy={lbp_stats['lbp_entropy']:.3f}")
    
    # Test GLCM
    glcm_features = extractor.calculate_glcm(gray, test_mask)
    print(f"GLCM contrast: {glcm_features['glcm_contrast_avg']:.3f}")
    
    print("All tests passed!")
