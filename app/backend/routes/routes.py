"""
Module: routes.py
FastAPI Routes cho API endpoints

Mô tả: Định nghĩa các API endpoints cho hệ thống.

Author: Tomato Quality System
"""

import os
import uuid
import base64
from io import BytesIO
from datetime import datetime
from typing import Optional

import cv2
import numpy as np
from fastapi import APIRouter, UploadFile, File, HTTPException, Form, Query
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ..database.database import get_database
from ..models.models import (
    AnalysisResponse, HistoryResponse, StatisticsResponse,
    SuccessResponse, ErrorResponse
)
from ..image_processing.preprocessing import ImagePreprocessor
from ..image_processing.segmentation import ImageSegmenter
from ..image_processing.feature_extraction import FeatureExtractor
from ..image_processing.classification import TomatoClassifier

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Routers
api_router = APIRouter(prefix="/api", tags=["api"])
upload_router = APIRouter(prefix="/upload", tags=["upload"])

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
STATIC_DIR = os.path.join(BASE_DIR, "static")

# Database path - cho phép override qua environment variable (Docker)
DB_PATH = os.environ.get(
    "DATABASE_PATH",
    os.path.join(BASE_DIR, "tomato_quality.db")
)

# Tạo thư mục nếu chưa có
os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)

# Services
preprocessor = ImagePreprocessor(target_size=(512, 512))
segmenter = ImageSegmenter()
feature_extractor = FeatureExtractor()
classifier = TomatoClassifier()
db = get_database(DB_PATH)


def save_uploaded_file(upload_file: UploadFile) -> str:
    """Lưu file upload và trả về đường dẫn."""
    # Tạo filename unique
    ext = os.path.splitext(upload_file.filename)[1] or ".jpg"
    filename = f"{uuid.uuid4().hex}{ext}"
    filepath = os.path.join(UPLOADS_DIR, filename)
    
    # Đọc và lưu file
    content = upload_file.file.read()
    with open(filepath, "wb") as f:
        f.write(content)
    
    return filepath


def save_processed_image(image: np.ndarray, prefix: str = "step") -> str:
    """Lưu ảnh đã xử lý và trả về đường dẫn."""
    filename = f"{prefix}_{uuid.uuid4().hex}.jpg"
    filepath = os.path.join(RESULTS_DIR, filename)
    cv2.imwrite(filepath, image)
    return filepath


def image_to_base64(image: np.ndarray) -> str:
    """Chuyển ảnh sang base64 string."""
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode('utf-8')


def analyze_image(image_path: str) -> dict:
    """
    Phân tích toàn diện một ảnh cà chua.
    
    Pipeline:
    1. Đọc ảnh
    2. Tiền xử lý
    3. Phân vùng
    4. Trích xuất đặc trưng
    5. Phân loại
    6. Lưu kết quả vào database
    
    Args:
        image_path: Đường dẫn ảnh
        
    Returns:
        Dictionary với kết quả phân tích
    """
    # Đọc ảnh
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Không thể đọc ảnh: {image_path}")
    
    result = {
        'image_path': image_path,
        'image_name': os.path.basename(image_path),
        'processing_steps': [],
        'images': {}
    }
    
    # === Bước 1: Tiền xử lý (Chapter 2) ===
    original = image.copy()
    result['processing_steps'].append({
        'step': 'original',
        'name': 'Ảnh gốc',
        'description': 'Ảnh đầu vào ban đầu'
    })
    
    # Resize
    resized, _ = preprocessor.resize(image)
    result['processing_steps'].append({
        'step': 'resize',
        'name': 'Resize',
        'description': f'Resize về {resized.shape[1]}x{resized.shape[0]}'
    })
    
    # Gaussian Blur
    blurred, _ = preprocessor.gaussian_blur(resized, kernel_size=(5, 5))
    result['processing_steps'].append({
        'step': 'gaussian_blur',
        'name': 'Gaussian Blur',
        'description': 'Làm mờ để giảm nhiễu'
    })
    
    # Convert to HSV
    hsv, _ = preprocessor.convert_to_hsv(blurred)
    result['processing_steps'].append({
        'step': 'hsv',
        'name': 'HSV Conversion',
        'description': 'Chuyển đổi sang không gian màu HSV'
    })
    
    # === Bước 2: Phân vùng (Chapter 4) ===
    segmentation_result = segmenter.segment_pipeline(hsv, blurred)
    
    result['processing_steps'].append({
        'step': 'segmentation',
        'name': 'Segmentation',
        'description': f'Phát hiện cà chua với {segmentation_result.get("contour_count", 0)} contours'
    })
    
    # Morphological Opening
    if 'final_mask' in segmentation_result:
        morph_opened, _ = preprocessor.morphological_opening(
            segmentation_result['final_mask'], kernel_size=(5, 5)
        )
        result['processing_steps'].append({
            'step': 'morphology',
            'name': 'Morphology',
            'description': 'Morphological opening để làm mịn mask'
        })
        mask = morph_opened
    else:
        mask = segmentation_result.get('hsv_mask', np.zeros(image.shape[:2], dtype=np.uint8))
    
    # Get ROI
    roi = segmentation_result.get('roi', image)
    contour = segmentation_result.get('largest_contour')
    
    result['processing_steps'].append({
        'step': 'roi',
        'name': 'ROI Extraction',
        'description': 'Trích xuất vùng cà chua'
    })
    
    # === Bước 3: Trích xuất đặc trưng (Chapter 3) ===
    features = feature_extractor.extract_all_features(roi, mask, contour)
    
    result['processing_steps'].append({
        'step': 'feature_extraction',
        'name': 'Feature Extraction',
        'description': 'Trích xuất đặc trưng màu sắc, hình dạng, texture'
    })
    
    # === Bước 4: Phân loại (Chapter 5) ===
    classification_result = classifier.classify(features, method='rule_based')
    
    result['processing_steps'].append({
        'step': 'classification',
        'name': 'Classification',
        'description': f'Phân loại: {classification_result.get("final_grade", "B")}'
    })
    
    # === Tổng hợp kết quả ===
    result.update({
        'features': features,
        'classification': classification_result,
        'ripeness_score': classification_result.get('ripeness_score', 0),
        'quality_score': classification_result.get('quality_score', 0),
        'combined_score': classification_result.get('combined_score', 0),
        'grade': classification_result.get('final_grade', 'B'),
        'grade_full': classification_result.get('final_grade_full', ''),
        'description': classification_result.get('description', ''),
        'red_ratio': features.get('red_ratio', 0),
        'green_ratio': features.get('green_ratio', 0),
        'yellow_ratio': features.get('yellow_ratio', 0),
        'brown_ratio': features.get('brown_ratio', 0),
        'circularity': features.get('circularity', 0),
        'solidity': features.get('solidity', 0),
        'area': features.get('area', 0),
        'perimeter': features.get('perimeter', 0),
        'lbp_entropy': features.get('lbp_entropy', 0),
        'glcm_homogeneity': features.get('glcm_homogeneity_avg', 0),
        'defect_ratio': features.get('defect_ratio', 0),
        'has_defects': features.get('has_defects', False),
        'defect_severity': features.get('defect_severity', 'none')
    })
    
    # === Tạo ảnh visualization ===
    # Original
    result['images']['original'] = image_to_base64(original)
    
    # Blurred
    result['images']['blurred'] = image_to_base64(blurred)
    
    # HSV mask
    if 'hsv_mask' in segmentation_result:
        hsv_vis = cv2.cvtColor(segmentation_result['hsv_mask'], cv2.COLOR_GRAY2BGR)
        result['images']['hsv_mask'] = image_to_base64(hsv_vis)
    
    # Final mask
    if 'final_mask' in segmentation_result:
        mask_vis = cv2.cvtColor(segmentation_result['final_mask'], cv2.COLOR_GRAY2BGR)
        result['images']['final_mask'] = image_to_base64(mask_vis)
    
    # ROI
    result['images']['roi'] = image_to_base64(roi)
    
    # Contour visualization
    contour_vis = original.copy()
    if contour is not None:
        cv2.drawContours(contour_vis, [contour], -1, (0, 255, 0), 3)
    result['images']['contour'] = image_to_base64(contour_vis)
    
    # RGB Histogram
    rgb_hist, _ = feature_extractor.calculate_rgb_histogram(roi, mask)
    
    # Save all images
    try:
        cv2.imwrite(os.path.join(RESULTS_DIR, f"original_{uuid.uuid4().hex}.jpg"), original)
        cv2.imwrite(os.path.join(RESULTS_DIR, f"roi_{uuid.uuid4().hex}.jpg"), roi)
        if contour is not None:
            cv2.imwrite(os.path.join(RESULTS_DIR, f"contour_{uuid.uuid4().hex}.jpg"), contour_vis)
    except Exception as e:
        logger.warning(f"Lỗi khi lưu ảnh: {e}")
    
    return result


# ==================== API ENDPOINTS ====================

@api_router.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Tomato Quality System API", "version": "1.0.0"}


@api_router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@api_router.post("/analyze", response_model=AnalysisResponse)
async def analyze(file: UploadFile = File(...)):
    """
    Phân tích ảnh cà chua.
    
    Upload ảnh và nhận kết quả phân tích đầy đủ.
    """
    try:
        # Lưu file upload
        filepath = save_uploaded_file(file)
        
        # Phân tích
        result = analyze_image(filepath)
        
        # Lưu vào database
        analysis_id = db.save_analysis({
            'image_name': result['image_name'],
            'image_path': filepath,
            'ripeness_score': result['ripeness_score'],
            'quality_score': result['quality_score'],
            'combined_score': result['combined_score'],
            'grade': result['grade'],
            'grade_full': result['grade_full'],
            'red_ratio': result['red_ratio'],
            'green_ratio': result['green_ratio'],
            'yellow_ratio': result['yellow_ratio'],
            'brown_ratio': result['brown_ratio'],
            'defect_ratio': result['defect_ratio'],
            'circularity': result['circularity'],
            'solidity': result['solidity'],
            'lbp_entropy': result['lbp_entropy'],
            'glcm_homogeneity_avg': result['glcm_homogeneity'],
            'has_defects': result['has_defects'],
            'defect_severity': result['defect_severity'],
            'features': result['features'],
            'result': result['classification']
        })
        
        return AnalysisResponse(
            id=analysis_id,
            image_name=result['image_name'],
            analysis_date=datetime.now(),
            ripeness_score=result['ripeness_score'],
            quality_score=result['quality_score'],
            combined_score=result['combined_score'],
            grade=result['grade'],
            grade_full=result['grade_full'],
            description=result['description'],
            red_ratio=result['red_ratio'],
            green_ratio=result['green_ratio'],
            yellow_ratio=result['yellow_ratio'],
            brown_ratio=result['brown_ratio'],
            circularity=result['circularity'],
            solidity=result['solidity'],
            area=result['area'],
            perimeter=result['perimeter'],
            lbp_entropy=result['lbp_entropy'],
            glcm_homogeneity=result['glcm_homogeneity'],
            defect_ratio=result['defect_ratio'],
            has_defects=result['has_defects'],
            defect_severity=result['defect_severity'],
            processing_steps=[s['name'] for s in result['processing_steps']],
            images=result['images']
        )
        
    except Exception as e:
        logger.error(f"Lỗi khi phân tích: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.post("/analyze/base64")
async def analyze_base64(image_data: dict):
    """
    Phân tích ảnh từ base64 string.
    
    Nhận ảnh dưới dạng base64 và trả về kết quả phân tích.
    """
    try:
        # Decode base64
        image_bytes = base64.b64decode(image_data.get('image', ''))
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("Không thể giải mã ảnh")
        
        # Lưu tạm file
        filename = f"base64_{uuid.uuid4().hex}.jpg"
        filepath = os.path.join(UPLOADS_DIR, filename)
        cv2.imwrite(filepath, image)
        
        # Phân tích
        result = analyze_image(filepath)
        
        # Lưu vào database
        analysis_id = db.save_analysis({
            'image_name': result['image_name'],
            'image_path': filepath,
            'ripeness_score': result['ripeness_score'],
            'quality_score': result['quality_score'],
            'combined_score': result['combined_score'],
            'grade': result['grade'],
            'grade_full': result['grade_full'],
            'red_ratio': result['red_ratio'],
            'green_ratio': result['green_ratio'],
            'yellow_ratio': result['yellow_ratio'],
            'brown_ratio': result['brown_ratio'],
            'defect_ratio': result['defect_ratio'],
            'circularity': result['circularity'],
            'solidity': result['solidity'],
            'lbp_entropy': result['lbp_entropy'],
            'glcm_homogeneity_avg': result['glcm_homogeneity'],
            'has_defects': result['has_defects'],
            'defect_severity': result['defect_severity'],
            'features': result['features'],
            'result': result['classification']
        })
        
        return {
            'success': True,
            'id': analysis_id,
            'image_name': result['image_name'],
            'ripeness_score': result['ripeness_score'],
            'quality_score': result['quality_score'],
            'combined_score': result['combined_score'],
            'grade': result['grade'],
            'grade_full': result['grade_full'],
            'description': result['description'],
            'features': {
                'red_ratio': result['red_ratio'],
                'green_ratio': result['green_ratio'],
                'yellow_ratio': result['yellow_ratio'],
                'brown_ratio': result['brown_ratio'],
                'circularity': result['circularity'],
                'solidity': result['solidity'],
                'defect_ratio': result['defect_ratio'],
                'has_defects': result['has_defects'],
                'defect_severity': result['defect_severity']
            },
            'processing_steps': result['processing_steps'],
            'images': result['images']
        }
        
    except Exception as e:
        logger.error(f"Lỗi khi phân tích base64: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.get("/history", response_model=HistoryResponse)
async def get_history(
    page: int = Query(1, ge=1),
    per_page: int = Query(50, ge=1, le=100),
    grade: Optional[str] = Query(None, description="Lọc theo grade (A, B, C)")
):
    """
    Lấy lịch sử phân tích.
    """
    try:
        offset = (page - 1) * per_page
        items = db.get_analysis_history(limit=per_page, offset=offset, grade_filter=grade)
        
        # Chuyển đổi thành AnalysisResponse
        history_items = []
        for item in items:
            history_items.append(AnalysisResponse(
                id=item['id'],
                image_name=item['image_name'],
                analysis_date=datetime.fromisoformat(item['analysis_date']) if item['analysis_date'] else None,
                ripeness_score=item['ripeness_score'],
                quality_score=item['quality_score'],
                combined_score=item['combined_score'],
                grade=item['grade'],
                grade_full=item['grade_full'],
                description="",
                red_ratio=item['red_ratio'],
                green_ratio=item['green_ratio'],
                yellow_ratio=item['yellow_ratio'],
                brown_ratio=item['brown_ratio'],
                circularity=item['circularity'],
                solidity=item['solidity'],
                area=None,
                perimeter=None,
                lbp_entropy=None,
                glcm_homogeneity=None,
                defect_ratio=item['defect_ratio'],
                has_defects=item['has_defects'],
                defect_severity=item['defect_severity']
            ))
        
        stats = db.get_statistics()
        
        return HistoryResponse(
            items=history_items,
            total=stats['total_analyses'],
            page=page,
            per_page=per_page
        )
        
    except Exception as e:
        logger.error(f"Lỗi khi lấy lịch sử: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.get("/history/{analysis_id}", response_model=AnalysisResponse)
async def get_history_detail(analysis_id: int):
    """
    Lấy chi tiết một bản ghi phân tích.
    """
    try:
        item = db.get_analysis_by_id(analysis_id)
        
        if not item:
            raise HTTPException(status_code=404, detail="Không tìm thấy bản ghi")
        
        return AnalysisResponse(
            id=item['id'],
            image_name=item['image_name'],
            analysis_date=datetime.fromisoformat(item['analysis_date']) if item['analysis_date'] else None,
            ripeness_score=item['ripeness_score'],
            quality_score=item['quality_score'],
            combined_score=item['combined_score'],
            grade=item['grade'],
            grade_full=item['grade_full'],
            description="",
            red_ratio=item['red_ratio'],
            green_ratio=item['green_ratio'],
            yellow_ratio=item['yellow_ratio'],
            brown_ratio=item['brown_ratio'],
            circularity=item['circularity'],
            solidity=item['solidity'],
            area=None,
            perimeter=None,
            lbp_entropy=item['lbp_entropy'],
            glcm_homogeneity=item['glcm_homogeneity'],
            defect_ratio=item['defect_ratio'],
            has_defects=bool(item['has_defects']),
            defect_severity=item['defect_severity']
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Lỗi khi lấy chi tiết: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.delete("/history/{analysis_id}", response_model=SuccessResponse)
async def delete_history(analysis_id: int):
    """
    Xóa một bản ghi phân tích.
    """
    try:
        deleted = db.delete_analysis(analysis_id)
        
        if not deleted:
            raise HTTPException(status_code=404, detail="Không tìm thấy bản ghi")
        
        return SuccessResponse(message=f"Đã xóa bản ghi #{analysis_id}")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Lỗi khi xóa: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.delete("/history", response_model=SuccessResponse)
async def clear_history():
    """
    Xóa toàn bộ lịch sử phân tích.
    """
    try:
        count = db.clear_history()
        return SuccessResponse(message=f"Đã xóa {count} bản ghi")
    except Exception as e:
        logger.error(f"Lỗi khi xóa lịch sử: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.get("/statistics", response_model=StatisticsResponse)
async def get_statistics():
    """
    Lấy thống kê tổng quan.
    """
    try:
        stats = db.get_statistics()
        return StatisticsResponse(**stats)
    except Exception as e:
        logger.error(f"Lỗi khi lấy thống kê: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== STATIC FILE SERVING ====================

def mount_static_files(app):
    """Mount static files cho frontend."""
    if os.path.exists(STATIC_DIR):
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    
    if os.path.exists(UPLOADS_DIR):
        app.mount("/uploads", StaticFiles(directory=UPLOADS_DIR), name="uploads")
    
    if os.path.exists(RESULTS_DIR):
        app.mount("/results", StaticFiles(directory=RESULTS_DIR), name="results")


if __name__ == "__main__":
    # Test
    import uvicorn
    uvicorn.run("routes:api", host="0.0.0.0", port=8000, reload=True)
