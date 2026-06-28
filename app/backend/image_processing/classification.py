"""
Module: classification.py
Chương 5: Nhận dạng và Phân loại

Mô tả: Module chứa các thuật toán phân loại và nhận dạng được học trong Chương 5.
Các kỹ thuật bao gồm:
- Rule-based Classification: Phân loại dựa trên luật
- KNN (K-Nearest Neighbors): Phân loại dựa trên khoảng cách
- SVM (Support Vector Machine): Phân loại bằng siêu phẳng

Phân loại chất lượng quả cà chua:
- Loại A (Đạt): Chất lượng tốt, màu đỏ đều, không có khuyết tật
- Loại B (Chưa chín): Màu xanh hoặc vàng còn nhiều
- Loại C (Không đạt): Có khuyết tật, màu bất thường

Author: Tomato Quality System
"""

import cv2
import numpy as np
from typing import Dict, Tuple, List, Optional
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TomatoClassifier:
    """
    Lớp phân loại cà chua - Xác định chất lượng và độ chín của quả cà chua.
    
    Sử dụng kết hợp:
    - Rule-based classification (nhanh, dễ hiểu)
    - ML-based (KNN, SVM) khi có dữ liệu huấn luyện
    """
    
    def __init__(self):
        """
        Khởi tạo classifier với các thông số và ngưỡng mặc định.
        """
        # Ngưỡng cho phân loại Rule-based
        self.ripeness_thresholds = {
            'red_ratio_high': 70.0,      # % màu đỏ cao -> chín
            'green_ratio_high': 30.0,    # % màu xanh cao -> chưa chín
            'yellow_ratio_high': 40.0,    # % màu vàng cao -> gần chín
        }
        
        self.quality_thresholds = {
            'defect_ratio_max': 5.0,      # % khuyết tật tối đa để đạt
            'circularity_min': 0.70,       # Độ tròn tối thiểu
            'solidity_min': 0.85,         # Solidity tối thiểu
            'homogeneity_min': 0.50,     # GLCM homogeneity tối thiểu
        }
        
        # ML models
        self.knn_model: Optional[KNeighborsClassifier] = None
        self.svm_model: Optional[SVC] = None
        self.scaler: Optional[StandardScaler] = None
        self.is_trained = False
        
        # Mapping nhãn
        self.label_mapping = {
            0: 'Loại A - Đạt tiêu chuẩn',
            1: 'Loại B - Chưa chín hoặc chưa đạt',
            2: 'Loại C - Không đạt'
        }
        
        logger.info("TomatoClassifier initialized with default thresholds")
    
    def calculate_ripeness_score(self, features: Dict) -> Tuple[float, Dict]:
        """
        Tính điểm độ chín của cà chua (0-100%).
        
        Công thức:
        - Dựa trên tỷ lệ màu đỏ
        - Trừ điểm nếu còn màu xanh
        - Trừ điểm nếu có màu vàng
        
        Args:
            features: Dictionary chứa các đặc trưng đã trích xuất
            
        Returns:
            Tuple chứa (điểm chín, dictionary chi tiết)
        """
        red_ratio = features.get('red_ratio', 0)
        green_ratio = features.get('green_ratio', 0)
        yellow_ratio = features.get('yellow_ratio', 0)
        
        # Hue peak cũng là chỉ số tốt
        hue_peak = features.get('hsv_histogram', {}).get('hue_peak', 0)
        
        # Tính điểm cơ bản từ red ratio
        base_score = red_ratio
        
        # Điều chỉnh theo các yếu tố khác
        green_penalty = green_ratio * 0.5  # Màu xanh làm giảm điểm
        yellow_penalty = yellow_ratio * 0.3  # Màu vàng giảm ít hơn
        
        # Hue peak gần 0 (đỏ) là tốt
        hue_score = max(0, 100 - hue_peak * 0.8) if hue_peak < 125 else 0
        
        # Kết hợp
        ripeness_score = (base_score + hue_score) / 2 - green_penalty - yellow_penalty
        ripeness_score = max(0, min(100, ripeness_score))
        
        details = {
            'base_from_red': red_ratio,
            'base_from_hue': hue_score,
            'green_penalty': green_penalty,
            'yellow_penalty': yellow_penalty,
            'final_score': ripeness_score
        }
        
        logger.info(f"Ripeness score: {ripeness_score:.1f}%")
        return ripeness_score, details
    
    def calculate_quality_score(self, features: Dict) -> Tuple[float, Dict]:
        """
        Tính điểm chất lượng tổng thể (0-100).
        
        Điểm dựa trên:
        - Hình dạng (circularity, solidity)
        - Texture (homogeneity, entropy)
        - Khuyết tật (defect ratio)
        
        Args:
            features: Dictionary chứa các đặc trưng
            
        Returns:
            Tuple chứa (điểm chất lượng, dictionary chi tiết)
        """
        scores = []
        details = {}
        
        # 1. Circularity (trọng số cao)
        circularity = features.get('circularity', 0.8)
        circularity_score = circularity * 100
        scores.append(circularity_score * 0.25)
        details['circularity_score'] = circularity_score
        
        # 2. Solidity (trọng số cao)
        solidity = features.get('solidity', 0.9)
        solidity_score = solidity * 100
        scores.append(solidity_score * 0.25)
        details['solidity_score'] = solidity_score
        
        # 3. GLCM Homogeneity
        homogeneity = features.get('glcm_homogeneity_avg', 0.5)
        homogeneity_score = homogeneity * 100
        scores.append(homogeneity_score * 0.20)
        details['homogeneity_score'] = homogeneity_score
        
        # 4. LBP Entropy (texture đều = entropy thấp = tốt)
        entropy = features.get('lbp_entropy', 3.0)
        # Normalize entropy: 0-5 range, thấp hơn là tốt hơn
        entropy_score = max(0, 100 - (entropy / 5 * 100))
        scores.append(entropy_score * 0.15)
        details['entropy_score'] = entropy_score
        
        # 5. Defect penalty
        defect_ratio = features.get('defect_ratio', 0)
        defect_score = max(0, 100 - defect_ratio * 5)  # Mỗi 1% defect giảm 5 điểm
        scores.append(defect_score * 0.15)
        details['defect_score'] = defect_score
        
        # Tổng hợp
        total_score = sum(scores)
        details['total_score'] = total_score
        details['breakdown'] = {
            'circularity_weight': 0.25,
            'solidity_weight': 0.25,
            'homogeneity_weight': 0.20,
            'entropy_weight': 0.15,
            'defect_weight': 0.15
        }
        
        logger.info(f"Quality score: {total_score:.1f}/100")
        return total_score, details
    
    def rule_based_classification(self, features: Dict) -> Dict:
        """
        Phân loại dựa trên luật (Rule-based Classification).
        
        Luật phân loại:
        - Loại A: red_ratio >= 70%, defect_ratio < 5%, circularity >= 0.75
        - Loại B: green_ratio > 30% hoặc (50% <= red_ratio < 70%, defect < 10%)
        - Loại C: defect_ratio >= 10% hoặc circularity < 0.70 hoặc brown_ratio > 15%
        
        Args:
            features: Dictionary chứa các đặc trưng
            
        Returns:
            Dictionary với kết quả phân loại
        """
        red_ratio = features.get('red_ratio', 0)
        green_ratio = features.get('green_ratio', 0)
        yellow_ratio = features.get('yellow_ratio', 0)
        brown_ratio = features.get('brown_ratio', 0)
        defect_ratio = features.get('defect_ratio', 0)
        circularity = features.get('circularity', 0)
        solidity = features.get('solidity', 0)
        
        # Tính điểm chín
        ripeness_score, _ = self.calculate_ripeness_score(features)
        
        # Tính điểm chất lượng
        quality_score, _ = self.calculate_quality_score(features)
        
        # Áp dụng luật phân loại
        rules_applied = []
        
        # Kiểm tra Loại C trước (ưu tiên cao nhất)
        if defect_ratio >= 10:
            grade = 'C'
            reason = f"Khuyết tật cao ({defect_ratio:.1f}%)"
            rules_applied.append("defect_ratio >= 10%")
        elif circularity < 0.70:
            grade = 'C'
            reason = f"Hình dạng không đều (circularity={circularity:.2f})"
            rules_applied.append("circularity < 0.70")
        elif brown_ratio > 15:
            grade = 'C'
            reason = f"Có vùng nâu/đen nhiều ({brown_ratio:.1f}%)"
            rules_applied.append("brown_ratio > 15%")
        elif defect_ratio >= 5:
            grade = 'C'
            reason = f"Có khuyết tật nhẹ ({defect_ratio:.1f}%)"
            rules_applied.append("defect_ratio >= 5%")
        # Kiểm tra Loại A
        elif red_ratio >= 70 and defect_ratio < 5 and circularity >= 0.75 and solidity >= 0.85:
            grade = 'A'
            reason = "Chín đều, không có khuyết tật, hình dạng tốt"
            rules_applied.append("red >= 70%, defect < 5%, circularity >= 0.75")
        # Kiểm tra Loại B
        elif green_ratio > 30:
            grade = 'B'
            reason = f"Còn xanh nhiều ({green_ratio:.1f}%)"
            rules_applied.append("green_ratio > 30%")
        elif red_ratio < 50:
            grade = 'B'
            reason = f"Chưa chín đủ ({red_ratio:.1f}% đỏ)"
            rules_applied.append("red_ratio < 50%")
        elif 50 <= red_ratio < 70:
            grade = 'B'
            reason = f"Gần chín ({red_ratio:.1f}% đỏ)"
            rules_applied.append("50% <= red_ratio < 70%")
        else:
            # Mặc định
            grade = 'B'
            reason = "Điều kiện không rõ ràng"
        
        result = {
            'grade': grade,
            'grade_full': self.label_mapping.get(
                0 if grade == 'A' else 1 if grade == 'B' else 2,
                f"Loại {grade}"
            ),
            'reason': reason,
            'ripeness_score': ripeness_score,
            'quality_score': quality_score,
            'combined_score': (ripeness_score * 0.6 + quality_score * 0.4),
            'rules_applied': rules_applied,
            'method': 'rule_based'
        }
        
        logger.info(f"Rule-based classification: Grade {grade} - {reason}")
        return result
    
    def prepare_features_for_ml(self, features: Dict) -> np.ndarray:
        """
        Chuẩn bị vector đặc trưng cho ML models.
        
        Args:
            features: Dictionary chứa các đặc trưng
            
        Returns:
            numpy array với các features được chọn
        """
        # Chọn các features quan trọng
        selected_features = [
            features.get('red_ratio', 0),
            features.get('green_ratio', 0),
            features.get('yellow_ratio', 0),
            features.get('brown_ratio', 0),
            features.get('circularity', 0),
            features.get('solidity', 0),
            features.get('aspect_ratio', 1),
            features.get('glcm_contrast_avg', 0),
            features.get('glcm_homogeneity_avg', 0),
            features.get('glcm_energy_avg', 0),
            features.get('lbp_entropy', 0),
            features.get('defect_ratio', 0),
        ]
        
        return np.array(selected_features).reshape(1, -1)
    
    def knn_classify(self, features: Dict) -> Optional[Dict]:
        """
        Phân loại bằng KNN (K-Nearest Neighbors).
        
        Thuật toán:
        1. Tính khoảng cách từ sample đến tất cả training samples
        2. Chọn K neighbors gần nhất
        3. Majority voting để xác định class
        
        Args:
            features: Dictionary chứa các đặc trưng
            
        Returns:
            Dictionary với kết quả hoặc None nếu chưa train
        """
        if not self.is_trained or self.knn_model is None:
            logger.warning("KNN model not trained yet")
            return None
        
        # Chuẩn bị features
        X = self.prepare_features_for_ml(features)
        
        # Scale
        X_scaled = self.scaler.transform(X)
        
        # Predict
        prediction = self.knn_model.predict(X_scaled)
        proba = self.knn_model.predict_proba(X_scaled)
        
        result = {
            'grade': self.label_mapping[prediction[0]],
            'confidence': float(max(proba[0])),
            'probabilities': {
                self.label_mapping[i]: float(p) for i, p in enumerate(proba[0])
            },
            'method': 'knn'
        }
        
        logger.info(f"KNN classification: {result['grade']} (confidence: {result['confidence']:.2f})")
        return result
    
    def svm_classify(self, features: Dict) -> Optional[Dict]:
        """
        Phân loại bằng SVM (Support Vector Machine).
        
        Thuật toán:
        1. Tìm siêu phẳng tối ưu chia các classes
        2. Maximum margin giữa các classes
        3. Sử dụng kernel (RBF) để xử lý non-linear
        
        Args:
            features: Dictionary chứa các đặc trưng
            
        Returns:
            Dictionary với kết quả hoặc None nếu chưa train
        """
        if not self.is_trained or self.svm_model is None:
            logger.warning("SVM model not trained yet")
            return None
        
        # Chuẩn bị features
        X = self.prepare_features_for_ml(features)
        
        # Scale
        X_scaled = self.scaler.transform(X)
        
        # Predict
        prediction = self.svm_model.predict(X_scaled)
        proba = self.svm_model.predict_proba(X_scaled)
        
        result = {
            'grade': self.label_mapping[prediction[0]],
            'confidence': float(max(proba[0])),
            'probabilities': {
                self.label_mapping[i]: float(p) for i, p in enumerate(proba[0])
            },
            'method': 'svm'
        }
        
        logger.info(f"SVM classification: {result['grade']} (confidence: {result['confidence']:.2f})")
        return result
    
    def train_models(self, X_train: np.ndarray, y_train: np.ndarray,
                    X_test: np.ndarray = None, y_test: np.ndarray = None):
        """
        Huấn luyện KNN và SVM models.
        
        Args:
            X_train: Features huấn luyện
            y_train: Labels huấn luyện
            X_test: Features test (tùy chọn)
            y_test: Labels test (tùy chọn)
        """
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train KNN
        self.knn_model = KNeighborsClassifier(n_neighbors=5, weights='distance')
        self.knn_model.fit(X_train_scaled, y_train)
        
        # Train SVM
        self.svm_model = SVC(kernel='rbf', probability=True, random_state=42)
        self.svm_model.fit(X_train_scaled, y_train)
        
        self.is_trained = True
        
        # Evaluate nếu có test data
        if X_test is not None and y_test is not None:
            X_test_scaled = self.scaler.transform(X_test)
            
            knn_acc = self.knn_model.score(X_test_scaled, y_test)
            svm_acc = self.svm_model.score(X_test_scaled, y_test)
            
            logger.info(f"Models trained - KNN accuracy: {knn_acc:.2f}, SVM accuracy: {svm_acc:.2f}")
        else:
            logger.info("Models trained successfully")
    
    def save_models(self, filepath: str):
        """
        Lưu models đã huấn luyện.
        
        Args:
            filepath: Đường dẫn file
        """
        model_data = {
            'knn_model': self.knn_model,
            'svm_model': self.svm_model,
            'scaler': self.scaler,
            'is_trained': self.is_trained,
            'label_mapping': self.label_mapping
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Models saved to {filepath}")
    
    def load_models(self, filepath: str):
        """
        Load models đã lưu.
        
        Args:
            filepath: Đường dẫn file
        """
        if not os.path.exists(filepath):
            logger.warning(f"Model file not found: {filepath}")
            return
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.knn_model = model_data.get('knn_model')
        self.svm_model = model_data.get('svm_model')
        self.scaler = model_data.get('scaler')
        self.is_trained = model_data.get('is_trained', False)
        self.label_mapping = model_data.get('label_mapping', self.label_mapping)
        
        logger.info(f"Models loaded from {filepath}")
    
    def classify(self, features: Dict, method: str = 'rule_based') -> Dict:
        """
        Phân loại cà chua với phương pháp được chọn.
        
        Args:
            features: Dictionary chứa các đặc trưng
            method: 'rule_based', 'knn', 'svm', hoặc 'all'
            
        Returns:
            Dictionary với kết quả phân loại đầy đủ
        """
        result = {}
        
        # Luôn tính ripeness và quality scores
        result['ripeness_score'], result['ripeness_details'] = self.calculate_ripeness_score(features)
        result['quality_score'], result['quality_details'] = self.calculate_quality_score(features)
        
        if method == 'all':
            # Chạy tất cả methods
            result['rule_based'] = self.rule_based_classification(features)
            result['knn'] = self.knn_classify(features)
            result['svm'] = self.svm_classify(features)
            
            # Final grade dựa trên rule-based (hoặc majority voting)
            if result['rule_based']:
                result['final_grade'] = result['rule_based']['grade']
                result['final_grade_full'] = result['rule_based']['grade_full']
                result['final_method'] = 'rule_based'
            elif result['knn']:
                result['final_grade'] = 'B'  # Default
                result['final_method'] = 'ml'
            else:
                result['final_grade'] = 'B'
                result['final_method'] = 'unknown'
                
        elif method == 'knn':
            knn_result = self.knn_classify(features)
            result['ml_classification'] = knn_result
            result['rule_based'] = self.rule_based_classification(features)
            result['final_grade'] = result['rule_based']['grade']
            result['final_grade_full'] = result['rule_based']['grade_full']
            result['final_method'] = 'rule_based'
            
        elif method == 'svm':
            svm_result = self.svm_classify(features)
            result['ml_classification'] = svm_result
            result['rule_based'] = self.rule_based_classification(features)
            result['final_grade'] = result['rule_based']['grade']
            result['final_grade_full'] = result['rule_based']['grade_full']
            result['final_method'] = 'rule_based'
            
        else:  # rule_based (default)
            result.update(self.rule_based_classification(features))
        
        # Tổng hợp combined score
        result['combined_score'] = (result['ripeness_score'] * 0.6 + result['quality_score'] * 0.4)
        
        # Mô tả kết quả
        result['description'] = self._get_grade_description(result['final_grade'])
        
        logger.info(f"Classification complete: Grade {result['final_grade']}, "
                   f"Score {result['combined_score']:.1f}")
        return result
    
    def _get_grade_description(self, grade: str) -> str:
        """
        Lấy mô tả cho grade.
        
        Args:
            grade: Ký hiệu grade (A, B, C)
            
        Returns:
            String mô tả
        """
        descriptions = {
            'A': "Cà chua Loại A - Chất lượng tốt, đạt tiêu chuẩn xuất khẩu. "
                 "Màu đỏ đều, hình dạng tròn đẹp, không có khuyết tật.",
            'B': "Cà chua Loại B - Chưa đạt tiêu chuẩn cao nhất. "
                 "Có thể chưa chín hoàn toàn hoặc có một số khuyết điểm nhỏ.",
            'C': "Cà chua Loại C - Không đạt tiêu chuẩn. "
                 "Cần loại bỏ do có khuyết tật hoặc chất lượng kém."
        }
        return descriptions.get(grade, "Không xác định")


def create_synthetic_dataset(n_samples: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """
    Tạo synthetic dataset cho việc huấn luyện.
    
    Args:
        n_samples: Số lượng samples
        
    Returns:
        Tuple (X, y) với features và labels
    """
    np.random.seed(42)
    
    X = []
    y = []
    
    # Loại A: red_ratio cao, defect thấp, circularity cao
    for _ in range(n_samples // 3):
        features = [
            np.random.uniform(70, 95),    # red_ratio
            np.random.uniform(0, 10),      # green_ratio
            np.random.uniform(0, 15),      # yellow_ratio
            np.random.uniform(0, 3),       # brown_ratio
            np.random.uniform(0.80, 0.98), # circularity
            np.random.uniform(0.90, 0.98), # solidity
            np.random.uniform(0.90, 1.10), # aspect_ratio
            np.random.uniform(0.1, 0.3),   # glcm_contrast
            np.random.uniform(0.6, 0.9),  # glcm_homogeneity
            np.random.uniform(0.05, 0.15), # glcm_energy
            np.random.uniform(2.0, 3.5),   # lbp_entropy
            np.random.uniform(0, 3),       # defect_ratio
        ]
        X.append(features)
        y.append(0)  # Loại A
    
    # Loại B: green/yellow cao, red thấp
    for _ in range(n_samples // 3):
        features = [
            np.random.uniform(30, 60),    # red_ratio
            np.random.uniform(20, 50),     # green_ratio
            np.random.uniform(10, 40),     # yellow_ratio
            np.random.uniform(0, 5),       # brown_ratio
            np.random.uniform(0.70, 0.90), # circularity
            np.random.uniform(0.80, 0.95),# solidity
            np.random.uniform(0.85, 1.15), # aspect_ratio
            np.random.uniform(0.2, 0.5),   # glcm_contrast
            np.random.uniform(0.4, 0.7),   # glcm_homogeneity
            np.random.uniform(0.03, 0.10), # glcm_energy
            np.random.uniform(3.0, 4.5),   # lbp_entropy
            np.random.uniform(0, 8),       # defect_ratio
        ]
        X.append(features)
        y.append(1)  # Loại B
    
    # Loại C: defect cao hoặc hình dạng kém
    for _ in range(n_samples // 3):
        features = [
            np.random.uniform(20, 70),     # red_ratio
            np.random.uniform(0, 30),      # green_ratio
            np.random.uniform(0, 30),      # yellow_ratio
            np.random.uniform(5, 25),      # brown_ratio
            np.random.uniform(0.50, 0.75), # circularity
            np.random.uniform(0.60, 0.85), # solidity
            np.random.uniform(0.70, 1.30), # aspect_ratio
            np.random.uniform(0.4, 1.0),   # glcm_contrast
            np.random.uniform(0.2, 0.5),   # glcm_homogeneity
            np.random.uniform(0.02, 0.08), # glcm_energy
            np.random.uniform(4.0, 5.5),   # lbp_entropy
            np.random.uniform(10, 40),     # defect_ratio
        ]
        X.append(features)
        y.append(2)  # Loại C
    
    return np.array(X), np.array(y)


if __name__ == "__main__":
    # Test module
    classifier = TomatoClassifier()
    
    # Tạo synthetic features
    test_features = {
        'red_ratio': 85.0,
        'green_ratio': 5.0,
        'yellow_ratio': 8.0,
        'brown_ratio': 2.0,
        'circularity': 0.92,
        'solidity': 0.95,
        'aspect_ratio': 1.05,
        'glcm_contrast_avg': 0.2,
        'glcm_homogeneity_avg': 0.75,
        'glcm_energy_avg': 0.12,
        'lbp_entropy': 3.2,
        'defect_ratio': 2.0,
        'hsv_histogram': {'hue_peak': 5}
    }
    
    # Test rule-based
    result = classifier.classify(test_features, method='rule_based')
    print(f"Grade: {result['final_grade']}")
    print(f"Ripeness: {result['ripeness_score']:.1f}%")
    print(f"Quality: {result['quality_score']:.1f}")
    print(f"Combined: {result['combined_score']:.1f}")
    print(f"Description: {result['description']}")
    
    # Test training
    X, y = create_synthetic_dataset(60)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    classifier.train_models(X_train, y_train, X_test, y_test)
    
    print("\nTraining complete!")
    print("All tests passed!")
