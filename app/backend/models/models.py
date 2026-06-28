"""
Module: models.py
Pydantic Models cho API

Mô tả: Định nghĩa các Pydantic models để validate và serialize data.

Author: Tomato Quality System
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class GradeEnum(str, Enum):
    """Enum cho các loại cà chua."""
    A = "A"
    B = "B"
    C = "C"


class AnalysisRequest(BaseModel):
    """Request model cho việc phân tích ảnh."""
    filename: str = Field(..., description="Tên file ảnh")
    save_to_history: bool = Field(default=True, description="Lưu vào lịch sử")


class AnalysisResponse(BaseModel):
    """Response model cho kết quả phân tích."""
    id: Optional[int] = Field(None, description="ID của bản ghi")
    image_name: str = Field(..., description="Tên ảnh")
    analysis_date: Optional[datetime] = Field(None, description="Thời gian phân tích")
    
    # Scores
    ripeness_score: float = Field(..., description="Điểm độ chín (%)")
    quality_score: float = Field(..., description="Điểm chất lượng (0-100)")
    combined_score: float = Field(..., description="Điểm tổng hợp")
    
    # Grade
    grade: str = Field(..., description="Loại cà chua (A/B/C)")
    grade_full: str = Field(..., description="Mô tả loại")
    description: str = Field(..., description="Mô tả chi tiết")
    
    # Color ratios
    red_ratio: float = Field(..., description="Tỷ lệ màu đỏ (%)")
    green_ratio: float = Field(..., description="Tỷ lệ màu xanh (%)")
    yellow_ratio: float = Field(..., description="Tỷ lệ màu vàng (%)")
    brown_ratio: float = Field(..., description="Tỷ lệ màu nâu (%)")
    
    # Shape features
    circularity: Optional[float] = Field(None, description="Độ tròn")
    solidity: Optional[float] = Field(None, description="Solidity")
    area: Optional[float] = Field(None, description="Diện tích (pixels)")
    perimeter: Optional[float] = Field(None, description="Chu vi (pixels)")
    
    # Texture features
    lbp_entropy: Optional[float] = Field(None, description="LBP Entropy")
    glcm_contrast: Optional[float] = Field(None, description="GLCM Contrast")
    glcm_homogeneity: Optional[float] = Field(None, description="GLCM Homogeneity")
    
    # Defect
    defect_ratio: float = Field(..., description="Tỷ lệ khuyết tật (%)")
    has_defects: bool = Field(..., description="Có khuyết tật không")
    defect_severity: str = Field(..., description="Mức độ khuyết tật")
    
    # Process visualization
    processing_steps: Optional[List[str]] = Field(default_factory=list, description="Các bước xử lý")
    images: Optional[Dict[str, str]] = Field(default_factory=dict, description="Đường dẫn ảnh đã xử lý")


class HistoryResponse(BaseModel):
    """Response model cho lịch sử phân tích."""
    items: List[AnalysisResponse] = Field(default_factory=list)
    total: int = Field(0, description="Tổng số bản ghi")
    page: int = Field(1, description="Trang hiện tại")
    per_page: int = Field(50, description="Số bản ghi mỗi trang")


class StatisticsResponse(BaseModel):
    """Response model cho thống kê."""
    total_analyses: int = Field(0, description="Tổng số phân tích")
    grade_distribution: Dict[str, int] = Field(default_factory=dict)
    avg_ripeness: float = Field(0, description="Độ chín trung bình")
    avg_quality: float = Field(0, description="Chất lượng trung bình")
    avg_combined: float = Field(0, description="Điểm tổng hợp trung bình")
    defective_count: int = Field(0, description="Số mẫu có khuyết tật")
    good_count: int = Field(0, description="Số mẫu tốt")


class ErrorResponse(BaseModel):
    """Response model cho lỗi."""
    error: str = Field(..., description="Mô tả lỗi")
    detail: Optional[str] = Field(None, description="Chi tiết lỗi")


class SuccessResponse(BaseModel):
    """Response model cho thành công."""
    message: str = Field(..., description="Thông báo")
    success: bool = Field(True)


class ProcessingStep(BaseModel):
    """Model cho một bước xử lý."""
    step: str = Field(..., description="Tên bước")
    description: str = Field(..., description="Mô tả bước")
    image_path: Optional[str] = Field(None, description="Đường dẫn ảnh minh họa")


class DatasetSample(BaseModel):
    """Model cho sample trong dataset."""
    id: Optional[int] = Field(None)
    image_name: str
    image_path: str
    category: str  # ripe, unripe, defective
    grade: Optional[str] = None
    added_date: Optional[datetime] = None
