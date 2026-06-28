# Tomato Quality System

Hệ thống phân loại độ chín và đánh giá chất lượng quả cà chua bằng xử lý ảnh và thị giác máy tính.

## Công nghệ sử dụng

- **Backend**: FastAPI (Python 3.10)
- **Frontend**: HTML5, Bootstrap 5, JavaScript
- **Xử lý ảnh**: OpenCV, scikit-image, scikit-learn
- **Database**: SQLite

## Cấu trúc dự án

```
TomatoVision/
├── app/
│   ├── backend/
│   │   ├── database/         # SQLite database
│   │   ├── image_processing/ # Tiền xử lý, phân vùng, trích xuất đặc trưng
│   │   ├── models/           # Pydantic models
│   │   ├── routes/           # API endpoints
│   │   └── services/
│   └── frontend/
│       └── index.html        # Giao diện web
├── dataset/
│   ├── ripe/                 # 97 ảnh cà chua chín
│   └── unripe/               # 80 ảnh cà chua xanh
├── main.py                   # Entry point FastAPI
├── requirements.txt          # Python dependencies
├── Dockerfile                # Docker image config
├── docker-compose.yml        # Docker Compose config
├── docker-entrypoint.py      # Docker entry point
└── .dockerignore
```

## Chạy với Docker (Khuyến nghị)

### Yêu cầu
- Docker Desktop 20.10+
- Docker Compose v2+

### Cách chạy

```bash
# Build và chạy container
docker-compose up --build -d

# Xem logs
docker-compose logs -f

# Dừng container
docker-compose down

# Dừng và xóa volumes
docker-compose down -v
```

Sau khi container chạy, truy cập:

- **Giao diện web**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Health check**: http://localhost:8000/api/health

### Chạy với Docker thuần (không dùng compose)

```bash
# Build image
docker build -t tomato-vision .

# Chạy container
docker run -d \
  --name tomato-vision-app \
  -p 8000:8000 \
  -v tomato_uploads:/app/uploads \
  -v tomato_results:/app/results \
  -v tomato_db:/app/data \
  -e DATABASE_PATH=/app/data/tomato_quality.db \
  tomato-vision

# Xem logs
docker logs -f tomato-vision-app

# Dừng và xóa
docker stop tomato-vision-app
docker rm tomato-vision-app
```

## Chạy trực tiếp (không dùng Docker)

### Cài đặt

```bash
# Tạo virtual environment
python -m venv venv

# Kích hoạt (Windows)
venv\Scripts\activate

# Kích hoạt (Linux/Mac)
source venv/bin/activate

# Cài đặt dependencies
pip install -r requirements.txt
```

### Chạy server

```bash
python main.py
```

Truy cập http://localhost:8000

## Tính năng

1. **Upload ảnh hoặc chụp từ webcam** để phân tích
2. **Phân vùng ảnh** tự động phát hiện vùng cà chua
3. **Trích xuất đặc trưng** màu sắc (HSV), hình dạng (Hu Moments), texture (LBP, GLCM)
4. **Phân loại chất lượng** thành 3 loại:
   - **Loại A**: Chất lượng cao, đạt tiêu chuẩn xuất khẩu
   - **Loại B**: Chất lượng trung bình
   - **Loại C**: Chất lượng thấp hoặc có khuyết tật
5. **Phát hiện khuyết tật**: đốm đen, vết nứt, dị dạng
6. **Lưu lịch sử phân tích** với SQLite database
7. **Thống kê tổng quan**: phân bố grade, điểm trung bình

## API Endpoints

| Method | Endpoint | Mô tả |
|--------|----------|-------|
| GET | `/` | Trang chủ (Frontend) |
| GET | `/dashboard` | Dashboard |
| GET | `/api/health` | Health check |
| POST | `/api/analyze` | Phân tích ảnh upload |
| POST | `/api/analyze/base64` | Phân tích ảnh base64 |
| GET | `/api/history` | Lấy lịch sử |
| GET | `/api/history/{id}` | Chi tiết 1 bản ghi |
| DELETE | `/api/history/{id}` | Xóa 1 bản ghi |
| DELETE | `/api/history` | Xóa toàn bộ lịch sử |
| GET | `/api/statistics` | Thống kê |

## Cấu trúc Chapter (Môn học áp dụng)

- **Chương 2**: Tiền xử lý ảnh (preprocessing.py)
- **Chương 3**: Trích xuất đặc trưng (feature_extraction.py)
- **Chương 4**: Phân vùng ảnh (segmentation.py)
- **Chương 5**: Nhận dạng và phân loại (classification.py)
