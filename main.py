"""
Main application file - Tomato Quality System
FastAPI Application Entry Point

Hệ thống phân loại độ chín và đánh giá chất lượng quả cà chua
bằng xử lý ảnh và thị giác máy tính

Author: Tomato Quality System
"""

import os
import sys

# Thêm đường dẫn để import modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi import Request
from fastapi.templating import Jinja2Templates
import uvicorn

# Import routes
from app.backend.routes.routes import api_router, mount_static_files

# Khởi tạo FastAPI app
app = FastAPI(
    title="Tomato Quality System",
    description="""
    Hệ thống phân loại độ chín và đánh giá chất lượng quả cà chua 
    bằng xử lý ảnh và thị giác máy tính
    
    ## Các tính năng:
    - Upload ảnh hoặc chụp từ webcam
    - Phân tích độ chín
    - Phát hiện khuyết tật
    - Phân loại chất lượng (Loại A/B/C)
    - Lưu lịch sử phân tích
    
    ## Các chương học áp dụng:
    - Chương 2: Tiền xử lý ảnh
    - Chương 3: Trích xuất đặc trưng
    - Chương 4: Phân vùng ảnh
    - Chương 5: Nhận dạng và phân loại
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
STATIC_DIR = os.path.join(BASE_DIR, "static")
FRONTEND_DIR = os.path.join(BASE_DIR, "app", "frontend")

# Tạo thư mục nếu chưa có
os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

# Mount static directories
mount_static_files(app)

# Include API routes
app.include_router(api_router)


@app.get("/", response_class=HTMLResponse)
async def root():
    """Trang chủ - Serve frontend."""
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    
    if os.path.exists(index_path):
        with open(index_path, 'r', encoding='utf-8') as f:
            return f.read()
    else:
        return """
        <!DOCTYPE html>
        <html lang="vi">
        <head>
            <meta charset="UTF-8">
            <title>Tomato Quality System</title>
            <style>
                body { 
                    font-family: Arial, sans-serif; 
                    display: flex; 
                    justify-content: center; 
                    align-items: center; 
                    min-height: 100vh; 
                    margin: 0;
                    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
                }
                .container {
                    text-align: center;
                    padding: 40px;
                    background: white;
                    border-radius: 20px;
                    box-shadow: 0 10px 40px rgba(0,0,0,0.1);
                }
                h1 { color: #b7131a; }
                .api-link {
                    display: inline-block;
                    margin-top: 20px;
                    padding: 12px 30px;
                    background: #b7131a;
                    color: white;
                    text-decoration: none;
                    border-radius: 8px;
                    font-weight: bold;
                }
                .api-link:hover { background: #93000d; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🍅 Tomato Quality System</h1>
                <p>Hệ thống phân loại độ chín và đánh giá chất lượng quả cà chua</p>
                <a href="/docs" class="api-link">Truy cập API Documentation</a>
            </div>
        </body>
        </html>
        """


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    """Trang dashboard chính."""
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


@app.get("/favicon.ico")
async def favicon():
    """Favicon."""
    return FileResponse(os.path.join(STATIC_DIR, "favicon.ico"))


def print_startup_message():
    """In thông tin khởi động."""
    print("\n" + "=" * 60)
    print("🍅 TOMATO QUALITY SYSTEM - KHỞI ĐỘNG")
    print("=" * 60)
    print("\n📋 Các module đã tải:")
    print("   ✅ Chương 2: Tiền xử lý ảnh (preprocessing.py)")
    print("   ✅ Chương 3: Trích xuất đặc trưng (feature_extraction.py)")
    print("   ✅ Chương 4: Phân vùng ảnh (segmentation.py)")
    print("   ✅ Chương 5: Nhận dạng và Phân loại (classification.py)")
    print("   ✅ Database: SQLite (database.py)")
    print("\n🌐 Các endpoints:")
    print("   • Trang chủ:     http://localhost:8000/")
    print("   • Dashboard:     http://localhost:8000/dashboard")
    print("   • API Docs:      http://localhost:8000/docs")
    print("   • ReDoc:         http://localhost:8000/redoc")
    print("\n📁 Thư mục:")
    print(f"   • Uploads:  {UPLOADS_DIR}")
    print(f"   • Results:  {RESULTS_DIR}")
    print(f"   • Static:   {STATIC_DIR}")
    print("\n" + "=" * 60)
    print("🎉 Hệ thống sẵn sàng!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    # Khởi động server
    print_startup_message()
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
