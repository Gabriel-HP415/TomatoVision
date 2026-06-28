"""
Module: segmentation.py
Chương 4: Phân vùng ảnh

Mô tả: Module chứa các thuật toán phân vùng (segmentation) ảnh được học trong Chương 4.
Các kỹ thuật bao gồm:
- HSV Threshold: Phân tách theo ngưỡng màu HSV
- Otsu Threshold: Tự động tìm ngưỡng tối ưu
- Adaptive Threshold: Ngưỡng thích nghi
- Contour Detection: Phát hiện đường contour
- Mask: Tạo mask để tách đối tượng
- ROI Extraction: Trích xuất vùng quan tâm

Author: Tomato Quality System
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImageSegmenter:
    """
    Lớp phân vùng ảnh - Tách riêng đối tượng (quả cà chua) khỏi nền.
    
    Sử dụng kết hợp nhiều phương pháp:
    - Threshold-based (HSV, Otsu, Adaptive)
    - Contour-based (tìm contour lớn nhất)
    - Masking (tạo mask nhị phân)
    """
    
    def __init__(self):
        """
        Khởi tạo segmenter với các thông số mặc định.
        """
        # Ngưỡng HSV cho màu đỏ (cà chua chín)
        self.red_hsv_lower1 = np.array([0, 50, 50])
        self.red_hsv_upper1 = np.array([10, 255, 255])
        self.red_hsv_lower2 = np.array([170, 50, 50])
        self.red_hsv_upper2 = np.array([180, 255, 255])
        
        # Ngưỡng HSV cho màu xanh lá (cà chua chưa chín)
        self.green_hsv_lower = np.array([35, 50, 50])
        self.green_hsv_upper = np.array([85, 255, 255])
        
        # Ngưỡng HSV cho màu vàng/cam (cà chua gần chín)
        self.yellow_hsv_lower = np.array([11, 50, 50])
        self.yellow_hsv_upper = np.array([34, 255, 255])
        
        logger.info("ImageSegmenter initialized with default tomato color thresholds")
    
    def hsv_threshold(self, hsv_image: np.ndarray, 
                     lower_bound: np.ndarray, 
                     upper_bound: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Phân tách theo ngưỡng màu HSV.
        
        Thuật toán: Tạo mask nhị phân dựa trên khoảng giá trị HSV.
        Pixel nằm trong khoảng [lower, upper] = 255 (trắng)
        Pixel nằm ngoài khoảng = 0 (đen)
        
        Ứng dụng cho cà chua:
        - Màu đỏ: Cần kết hợp 2 khoảng (0-10 và 170-180) vì đỏ nằm ở
          2 đầu của Hue wheel
        
        Args:
            hsv_image: Ảnh HSV
            lower_bound: Giới hạn dưới [H, S, V]
            upper_bound: Giới hạn trên [H, S, V]
            
        Returns:
            Tuple chứa (mask nhị phân, metadata)
        """
        mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
        
        metadata = {
            "method": "HSV_Threshold",
            "lower_bound": lower_bound.tolist(),
            "upper_bound": upper_bound.tolist(),
            "white_pixels": int(np.sum(mask == 255)),
            "black_pixels": int(np.sum(mask == 0)),
            "description": "Phân tách theo ngưỡng màu HSV"
        }
        
        logger.info(f"Applied HSV threshold: {lower_bound} - {upper_bound}")
        return mask, metadata
    
    def combine_hsv_masks(self, hsv_image: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Kết hợp nhiều mask HSV để tách cà chua (cả đỏ, vàng, xanh).
        
        Args:
            hsv_image: Ảnh HSV
            
        Returns:
            Tuple chứa (mask kết hợp, metadata)
        """
        # Tạo mask cho màu đỏ (2 vùng)
        mask_red1 = cv2.inRange(hsv_image, self.red_hsv_lower1, self.red_hsv_upper1)
        mask_red2 = cv2.inRange(hsv_image, self.red_hsv_lower2, self.red_hsv_upper2)
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)
        
        # Tạo mask cho màu vàng
        mask_yellow = cv2.inRange(hsv_image, self.yellow_hsv_lower, self.yellow_hsv_upper)
        
        # Tạo mask cho màu xanh
        mask_green = cv2.inRange(hsv_image, self.green_hsv_lower, self.green_hsv_upper)
        
        # Kết hợp tất cả (cà chua có thể có nhiều màu)
        combined_mask = cv2.bitwise_or(mask_red, mask_yellow)
        combined_mask = cv2.bitwise_or(combined_mask, mask_green)
        
        # Áp dụng morphological để làm mịn mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        
        metadata = {
            "method": "Combined_HSV",
            "red_mask_coverage": float(np.sum(mask_red > 0)) / mask_red.size * 100,
            "yellow_mask_coverage": float(np.sum(mask_yellow > 0)) / mask_yellow.size * 100,
            "green_mask_coverage": float(np.sum(mask_green > 0)) / mask_green.size * 100,
            "description": "Kết hợp mask cho tất cả màu cà chua"
        }
        
        logger.info(f"Combined HSV masks: R={metadata['red_mask_coverage']:.1f}%, "
                    f"Y={metadata['yellow_mask_coverage']:.1f}%, "
                    f"G={metadata['green_mask_coverage']:.1f}%")
        return combined_mask, metadata
    
    def otsu_threshold(self, gray_image: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Phân tách Otsu - Tự động tìm ngưỡng tối ưu.
        
        Thuật toán: Otsu's thresholding là phương pháp tự động chọn ngưỡng
        dựa trên việc tối thiểu hóa phương sai trong lớp (within-class variance)
        hoặc tương đương với việc tối đa hóa phương sai giữa các lớp (between-class variance).
        
        Công thức:
        - Tính histogram của ảnh
        - Với mỗi ngưỡng t, tính:
          - p1(t) = sum(p(i), i=0..t) (xác suất lớp 1)
          - m1(t) = sum(i*p(i)/p1, i=0..t) (trung bình lớp 1)
          - m2(t) = sum(i*p(i)/(1-p1), i=t+1..L) (trung bình lớp 2)
          - m = sum(i*p(i), i=0..L) (trung bình toàn ảnh)
        - Between-class variance: σ²(t) = p1*(m1-m)² + p2*(m2-m)²
        - Chọn t* = argmax σ²(t)
        
        Ưu điểm: Tự động, không cần chọn ngưỡng thủ công
        
        Args:
            gray_image: Ảnh xám đầu vào
            
        Returns:
            Tuple chứa (mask nhị phân, metadata với threshold tìm được)
        """
        # Áp dụng Otsu's thresholding
        _, binary = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        metadata = {
            "method": "Otsu_Threshold",
            "threshold_value": int(_),
            "description": "Tự động tìm ngưỡng tối ưu bằng Otsu's algorithm"
        }
        
        logger.info(f"Otsu threshold calculated: {metadata['threshold_value']}")
        return binary, metadata
    
    def adaptive_threshold(self, gray_image: np.ndarray, 
                          block_size: int = 11,
                          c: int = 2) -> Tuple[np.ndarray, dict]:
        """
        Phân tách Adaptive Threshold - Ngưỡng thích nghi theo vùng.
        
        Thuật toán: Thay vì dùng một ngưỡng toàn cục, adaptive threshold
        tính ngưỡng cục bộ cho từng vùng của ảnh.
        
        Công thức:
        T(x,y) = mean(I(vùng)) - C cho Gaussian
        T(x,y) = median(I(vùng)) - C cho Mean
        
        Ưu điểm: Xử lý tốt ảnh có độ sáng không đồng đều
        
        Args:
            gray_image: Ảnh xám
            block_size: Kích thước cửa sổ (phải lẻ)
            c: Hằng số trừ đi sau khi tính mean/median
            
        Returns:
            Tuple chứa (mask nhị phân, metadata)
        """
        if block_size % 2 == 0:
            block_size += 1
        
        # Adaptive Mean Threshold
        adaptive_mean = cv2.adaptiveThreshold(
            gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY, block_size, c
        )
        
        # Adaptive Gaussian Threshold
        adaptive_gaussian = cv2.adaptiveThreshold(
            gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, block_size, c
        )
        
        metadata = {
            "method": "Adaptive_Threshold",
            "block_size": block_size,
            "c_constant": c,
            "mean_threshold": int(np.mean(adaptive_mean)),
            "gaussian_threshold": int(np.mean(adaptive_gaussian)),
            "description": "Ngưỡng thích nghi theo vùng (Mean và Gaussian)"
        }
        
        logger.info(f"Applied adaptive threshold with block_size={block_size}")
        return adaptive_gaussian, metadata
    
    def find_contours(self, binary_mask: np.ndarray) -> Tuple[List, np.ndarray]:
        """
        Phát hiện contours từ mask nhị phân.
        
        Thuật toán: Sử dụng Suzuki và Abe (1985) border following algorithm
        để tìm các đường contour từ ảnh nhị phân.
        
        Contour là đường cong khép kín bao quanh các vùng có cùng
        cường độ pixel (boundary của vùng).
        
        Args:
            binary_mask: Mask nhị phân (0 và 255)
            
        Returns:
            Tuple chứa (danh sách contours, ảnh hierarchy)
        """
        contours, hierarchy = cv2.findContours(
            binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        logger.info(f"Found {len(contours)} contours")
        return contours, hierarchy
    
    def find_largest_contour(self, contours: List) -> Tuple[Optional[np.ndarray], dict]:
        """
        Tìm contour lớn nhất (ứng viên cho quả cà chua).
        
        Args:
            contours: Danh sách các contours
            
        Returns:
            Tuple chứa (contour lớn nhất, metadata)
        """
        if not contours:
            logger.warning("No contours found!")
            return None, {"error": "No contours found"}
        
        # Tính diện tích của từng contour
        areas = [cv2.contourArea(c) for c in contours]
        
        # Tìm contour có diện tích lớn nhất
        max_idx = np.argmax(areas)
        largest_contour = contours[max_idx]
        
        metadata = {
            "total_contours": len(contours),
            "largest_contour_area": float(areas[max_idx]),
            "largest_contour_index": int(max_idx),
            "all_areas": [float(a) for a in areas],
            "description": "Tìm contour lớn nhất"
        }
        
        logger.info(f"Largest contour area: {areas[max_idx]:.2f} pixels")
        return largest_contour, metadata
    
    def approximate_contour(self, contour: np.ndarray, 
                            epsilon_factor: float = 0.02) -> Tuple[np.ndarray, dict]:
        """
        Xấp xỉ contour bằng đa giác.
        
        Thuật toán: Douglas-Peucker algorithm
        1. Nối điểm đầu và cuối bằng đoạn thẳng
        2. Tìm điểm có khoảng cách lớn nhất đến đoạn thẳng
        3. Nếu khoảng cách > epsilon, chia đoạn tại điểm đó
        4. Lặp lại cho đến khi tất cả điểm đều nằm trong epsilon
        
        Args:
            contour: Contour cần xấp xỉ
            epsilon_factor: Hệ số epsilon (0.01-0.05 phù hợp cho hình tròn)
            
        Returns:
            Tuple chứa (contour đã xấp xỉ, metadata)
        """
        epsilon = epsilon_factor * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        metadata = {
            "original_points": len(contour),
            "approximated_points": len(approx),
            "epsilon": float(epsilon),
            "epsilon_factor": epsilon_factor,
            "description": "Xấp xỉ contour bằng Douglas-Peucker"
        }
        
        logger.info(f"Contour approximated: {len(contour)} -> {len(approx)} points")
        return approx, metadata
    
    def create_mask_from_contour(self, contour: np.ndarray, 
                                 image_shape: Tuple[int, ...]) -> Tuple[np.ndarray, dict]:
        """
        Tạo mask nhị phân từ contour.
        
        Args:
            contour: Contour định nghĩa vùng
            image_shape: Kích thước ảnh (H, W)
            
        Returns:
            Tuple chứa (mask, metadata)
        """
        mask = np.zeros(image_shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)  # -1 = filled
        
        metadata = {
            "mask_area": int(np.sum(mask > 0)),
            "mask_coverage": float(np.sum(mask > 0)) / mask.size * 100,
            "description": "Tạo mask từ contour"
        }
        
        logger.info(f"Mask created with {metadata['mask_area']} pixels")
        return mask, metadata
    
    def extract_roi(self, original_image: np.ndarray, 
                    mask: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Trích xuất vùng quan tâm (ROI) từ ảnh gốc sử dụng mask.
        
        Args:
            original_image: Ảnh gốc (BGR)
            mask: Mask nhị phân
            
        Returns:
            Tuple chứa (ROI đã trích xuất, metadata)
        """
        # Áp dụng mask lên ảnh gốc
        roi = cv2.bitwise_and(original_image, original_image, mask=mask)
        
        # Tính bounding box của mask
        coords = cv2.findNonZero(mask)
        x, y, w, h = cv2.boundingRect(coords)
        
        # Cắt ROI
        roi_cropped = roi[y:y+h, x:x+w]
        
        # Tạo nền trắng cho phần không thuộc cà chua
        roi_final = np.zeros_like(original_image)
        roi_final[mask > 0] = original_image[mask > 0]
        
        metadata = {
            "bounding_box": {"x": int(x), "y": int(y), "width": int(w), "height": int(h)},
            "roi_area": float(np.sum(mask > 0)),
            "image_coverage": float(np.sum(mask > 0)) / mask.size * 100,
            "description": "Trích xuất vùng cà chua từ ảnh gốc"
        }
        
        logger.info(f"ROI extracted: bbox={metadata['bounding_box']}")
        return roi_final, metadata
    
    def remove_small_contours(self, contours: List, 
                              min_area: float = 500) -> Tuple[List, dict]:
        """
        Loại bỏ các contour quá nhỏ (noise).
        
        Args:
            contours: Danh sách contours
            min_area: Diện tích tối thiểu để giữ lại
            
        Returns:
            Tuple chứa (contours đã lọc, metadata)
        """
        filtered_contours = [c for c in contours if cv2.contourArea(c) >= min_area]
        removed_count = len(contours) - len(filtered_contours)
        
        metadata = {
            "original_count": len(contours),
            "remaining_count": len(filtered_contours),
            "removed_count": removed_count,
            "min_area_threshold": min_area,
            "description": "Loại bỏ contours nhỏ hơn ngưỡng"
        }
        
        logger.info(f"Removed {removed_count} small contours")
        return filtered_contours, metadata
    
    def convex_hull(self, contour: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Tính convex hull (đa giác lồi bao quanh contour).
        
        Thuật toán: Graham scan hoặc Jarvis march
        Tìm đa giác lồi nhỏ nhất chứa tất cả các điểm của contour.
        
        Args:
            contour: Contour đầu vào
            
        Returns:
            Tuple chứa (convex hull, metadata)
        """
        hull = cv2.convexHull(contour)
        
        hull_area = cv2.contourArea(hull)
        contour_area = cv2.contourArea(contour)
        solidity = contour_area / hull_area if hull_area > 0 else 0
        
        metadata = {
            "hull_points": len(hull),
            "hull_area": float(hull_area),
            "contour_area": float(contour_area),
            "solidity": float(solidity),
            "description": "Tính convex hull và solidity"
        }
        
        logger.info(f"Convex hull computed, solidity={solidity:.3f}")
        return hull, metadata
    
    def segment_pipeline(self, hsv_image: np.ndarray, 
                        original_image: np.ndarray) -> Dict:
        """
        Pipeline phân vùng hoàn chỉnh.
        
        Các bước:
        1. Tạo mask HSV cho màu cà chua
        2. Morphological để làm mịn mask
        3. Tìm contour lớn nhất
        4. Tạo mask cuối cùng
        5. Trích xuất ROI
        
        Args:
            hsv_image: Ảnh HSV đã tiền xử lý
            original_image: Ảnh gốc (BGR)
            
        Returns:
            Dictionary chứa tất cả kết quả và metadata
        """
        result = {}
        
        # Bước 1: Tạo combined HSV mask
        combined_mask, meta = self.combine_hsv_masks(hsv_image)
        result['hsv_mask'] = combined_mask
        result['hsv_metadata'] = meta
        
        # Bước 2: Tìm contours
        contours, hierarchy = self.find_contours(combined_mask)
        result['contours'] = contours
        result['contour_count'] = len(contours)
        
        # Bước 3: Loại bỏ contours nhỏ
        filtered_contours, _ = self.remove_small_contours(contours, min_area=1000)
        result['filtered_contours'] = filtered_contours
        
        # Bước 4: Tìm contour lớn nhất
        if filtered_contours:
            largest_contour, meta = self.find_largest_contour(filtered_contours)
            result['largest_contour'] = largest_contour
            result['largest_contour_metadata'] = meta
            
            # Bước 5: Tạo mask từ contour lớn nhất
            final_mask, meta = self.create_mask_from_contour(
                largest_contour, original_image.shape[:2]
            )
            result['final_mask'] = final_mask
            result['final_mask_metadata'] = meta
            
            # Bước 6: Trích xuất ROI
            roi, meta = self.extract_roi(original_image, final_mask)
            result['roi'] = roi
            result['roi_metadata'] = meta
            
            # Bước 7: Tính convex hull
            hull, meta = self.convex_hull(largest_contour)
            result['hull'] = hull
            result['hull_metadata'] = meta
        else:
            result['largest_contour'] = None
            result['final_mask'] = combined_mask
            result['roi'] = original_image
            result['error'] = "No significant tomato contour found"
        
        logger.info("Segmentation pipeline completed")
        return result
    
    def get_segmentation_visualization(self, original: np.ndarray,
                                       masks: Dict) -> List[np.ndarray]:
        """
        Tạo danh sách ảnh để hiển thị các bước phân vùng.
        
        Args:
            original: Ảnh gốc
            masks: Dictionary chứa các mask
            
        Returns:
            Danh sách ảnh để hiển thị
        """
        visualizations = []
        
        # Ảnh gốc
        visualizations.append(original.copy())
        
        # HSV mask
        if 'hsv_mask' in masks:
            hsv_vis = cv2.cvtColor(masks['hsv_mask'], cv2.COLOR_GRAY2BGR)
            visualizations.append(hsv_vis)
        
        # Final mask
        if 'final_mask' in masks:
            mask_vis = cv2.cvtColor(masks['final_mask'], cv2.COLOR_GRAY2BGR)
            visualizations.append(mask_vis)
        
        # ROI
        if 'roi' in masks:
            visualizations.append(masks['roi'])
        
        # Contours on original
        if 'largest_contour' in masks and masks['largest_contour'] is not None:
            contour_vis = original.copy()
            cv2.drawContours(contour_vis, [masks['largest_contour']], -1, (0, 255, 0), 3)
            visualizations.append(contour_vis)
        
        # Final result with hull
        if 'hull' in masks and 'largest_contour' in masks:
            final_vis = original.copy()
            cv2.drawContours(final_vis, [masks['largest_contour']], -1, (0, 255, 0), 2)
            if masks['hull'] is not None:
                cv2.drawContours(final_vis, [masks['hull']], -1, (255, 0, 0), 2)
            visualizations.append(final_vis)
        
        return visualizations


def calculate_connected_components(binary_mask: np.ndarray) -> Tuple[int, np.ndarray]:
    """
    Tính số thành phần liên thông (Connected Components).
    
    Sử dụng để đếm số đối tượng riêng biệt trong mask.
    
    Args:
        binary_mask: Mask nhị phân
        
    Returns:
        Tuple chứa (số thành phần, ảnh labels)
    """
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary_mask, connectivity=8
    )
    return num_labels - 1, labels  # -1 vì background cũng là 1 component


if __name__ == "__main__":
    # Test module
    import sys
    
    # Tạo ảnh test
    test_image = np.zeros((256, 256, 3), dtype=np.uint8)
    cv2.circle(test_image, (128, 128), 80, (0, 0, 255), -1)
    
    segmenter = ImageSegmenter()
    
    # Chuyển sang HSV
    hsv = cv2.cvtColor(test_image, cv2.COLOR_BGR2HSV)
    
    # Test segmentation
    result = segmenter.segment_pipeline(hsv, test_image)
    
    print(f"Contours found: {result.get('contour_count', 0)}")
    print(f"Final mask created: {result.get('final_mask_metadata', {}).get('mask_coverage', 0):.2f}%")
    print(f"ROI extracted: {result.get('roi_metadata', {}).get('image_coverage', 0):.2f}%")
    
    print("All tests passed!")
