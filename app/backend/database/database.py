"""
Module: database.py
Quản lý Database SQLite cho lưu trữ lịch sử phân tích

Mô tả: Module chứa các class và hàm để quản lý database SQLite.
Lưu trữ:
- Lịch sử kiểm tra
- Đường dẫn ảnh
- Kết quả phân loại
- Thời gian
- Các thông số phân tích

Author: Tomato Quality System
"""

import sqlite3
import json
from datetime import datetime
from typing import List, Optional, Dict, Any
from pathlib import Path
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Database:
    """
    Lớp quản lý Database SQLite.
    
    Quản lý kết nối, tạo bảng, CRUD operations cho lịch sử phân tích.
    """
    
    def __init__(self, db_path: str = "tomato_quality.db"):
        """
        Khởi tạo Database.
        
        Args:
            db_path: Đường dẫn file database
        """
        self.db_path = db_path
        self._init_database()
        logger.info(f"Database initialized at {db_path}")
    
    def _get_connection(self) -> sqlite3.Connection:
        """
        Lấy kết nối database.
        
        Returns:
            sqlite3 Connection
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def _init_database(self):
        """
        Khởi tạo database và tạo các bảng.
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Tạo bảng analysis_history
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS analysis_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_name TEXT NOT NULL,
                image_path TEXT NOT NULL,
                analysis_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                ripeness_score REAL,
                quality_score REAL,
                combined_score REAL,
                grade TEXT,
                grade_full TEXT,
                red_ratio REAL,
                green_ratio REAL,
                yellow_ratio REAL,
                brown_ratio REAL,
                defect_ratio REAL,
                circularity REAL,
                solidity REAL,
                lbp_entropy REAL,
                glcm_homogeneity REAL,
                has_defects INTEGER,
                defect_severity TEXT,
                features_json TEXT,
                result_json TEXT,
                status TEXT DEFAULT 'completed'
            )
        ''')
        
        # Tạo bảng dataset_samples (để quản lý dataset)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS dataset_samples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_name TEXT NOT NULL,
                image_path TEXT NOT NULL,
                category TEXT NOT NULL,
                added_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                grade TEXT,
                features_json TEXT
            )
        ''')
        
        # Tạo bảng settings
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT,
                updated_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
        
        logger.info("Database tables created successfully")
    
    def save_analysis(self, analysis_data: Dict) -> int:
        """
        Lưu kết quả phân tích vào database.
        
        Args:
            analysis_data: Dictionary chứa thông tin phân tích
            
        Returns:
            ID của record vừa lưu
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO analysis_history (
                image_name, image_path, ripeness_score, quality_score,
                combined_score, grade, grade_full,
                red_ratio, green_ratio, yellow_ratio, brown_ratio,
                defect_ratio, circularity, solidity,
                lbp_entropy, glcm_homogeneity,
                has_defects, defect_severity,
                features_json, result_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            analysis_data.get('image_name', 'unknown.jpg'),
            analysis_data.get('image_path', ''),
            analysis_data.get('ripeness_score', 0),
            analysis_data.get('quality_score', 0),
            analysis_data.get('combined_score', 0),
            analysis_data.get('grade', 'B'),
            analysis_data.get('grade_full', ''),
            analysis_data.get('red_ratio', 0),
            analysis_data.get('green_ratio', 0),
            analysis_data.get('yellow_ratio', 0),
            analysis_data.get('brown_ratio', 0),
            analysis_data.get('defect_ratio', 0),
            analysis_data.get('circularity', 0),
            analysis_data.get('solidity', 0),
            analysis_data.get('lbp_entropy', 0),
            analysis_data.get('glcm_homogeneity_avg', 0),
            1 if analysis_data.get('has_defects', False) else 0,
            analysis_data.get('defect_severity', 'none'),
            json.dumps(analysis_data.get('features', {})),
            json.dumps(analysis_data.get('result', {}))
        ))
        
        analysis_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        logger.info(f"Analysis saved with ID: {analysis_id}")
        return analysis_id
    
    def get_analysis_history(self, limit: int = 50, 
                            offset: int = 0,
                            grade_filter: Optional[str] = None) -> List[Dict]:
        """
        Lấy lịch sử phân tích.
        
        Args:
            limit: Số lượng records tối đa
            offset: Bắt đầu từ record
            grade_filter: Lọc theo grade (A, B, C)
            
        Returns:
            List chứa các records
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        query = 'SELECT * FROM analysis_history'
        params = []
        
        if grade_filter:
            query += ' WHERE grade = ?'
            params.append(grade_filter)
        
        query += ' ORDER BY analysis_date DESC LIMIT ? OFFSET ?'
        params.extend([limit, offset])
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        results = []
        for row in rows:
            results.append({
                'id': row['id'],
                'image_name': row['image_name'],
                'image_path': row['image_path'],
                'analysis_date': row['analysis_date'],
                'ripeness_score': row['ripeness_score'],
                'quality_score': row['quality_score'],
                'combined_score': row['combined_score'],
                'grade': row['grade'],
                'grade_full': row['grade_full'],
                'red_ratio': row['red_ratio'],
                'green_ratio': row['green_ratio'],
                'yellow_ratio': row['yellow_ratio'],
                'brown_ratio': row['brown_ratio'],
                'defect_ratio': row['defect_ratio'],
                'circularity': row['circularity'],
                'solidity': row['solidity'],
                'has_defects': bool(row['has_defects']),
                'defect_severity': row['defect_severity']
            })
        
        conn.close()
        return results
    
    def get_analysis_by_id(self, analysis_id: int) -> Optional[Dict]:
        """
        Lấy một record phân tích theo ID.
        
        Args:
            analysis_id: ID của record
            
        Returns:
            Dictionary hoặc None nếu không tìm thấy
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM analysis_history WHERE id = ?', (analysis_id,))
        row = cursor.fetchone()
        
        conn.close()
        
        if row:
            return dict(row)
        return None
    
    def delete_analysis(self, analysis_id: int) -> bool:
        """
        Xóa một record phân tích.
        
        Args:
            analysis_id: ID của record
            
        Returns:
            True nếu xóa thành công
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('DELETE FROM analysis_history WHERE id = ?', (analysis_id,))
        deleted = cursor.rowcount > 0
        
        conn.commit()
        conn.close()
        
        if deleted:
            logger.info(f"Analysis {analysis_id} deleted")
        return deleted
    
    def get_statistics(self) -> Dict:
        """
        Lấy thống kê tổng quan.
        
        Returns:
            Dictionary với các thống kê
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Tổng số phân tích
        cursor.execute('SELECT COUNT(*) as total FROM analysis_history')
        total = cursor.fetchone()['total']
        
        # Phân bố theo grade
        cursor.execute('''
            SELECT grade, COUNT(*) as count 
            FROM analysis_history 
            GROUP BY grade
        ''')
        grade_distribution = {row['grade']: row['count'] for row in cursor.fetchall()}
        
        # Trung bình scores
        cursor.execute('''
            SELECT 
                AVG(ripeness_score) as avg_ripeness,
                AVG(quality_score) as avg_quality,
                AVG(combined_score) as avg_combined
            FROM analysis_history
        ''')
        avg_scores = dict(cursor.fetchone())
        
        # Đếm theo defect
        cursor.execute('''
            SELECT 
                SUM(CASE WHEN has_defects = 1 THEN 1 ELSE 0 END) as defective_count,
                SUM(CASE WHEN has_defects = 0 THEN 1 ELSE 0 END) as good_count
            FROM analysis_history
        ''')
        defect_stats = dict(cursor.fetchone())
        
        conn.close()
        
        return {
            'total_analyses': total,
            'grade_distribution': grade_distribution,
            'avg_ripeness': avg_scores.get('avg_ripeness', 0) or 0,
            'avg_quality': avg_scores.get('avg_quality', 0) or 0,
            'avg_combined': avg_scores.get('avg_combined', 0) or 0,
            'defective_count': defect_stats.get('defective_count', 0) or 0,
            'good_count': defect_stats.get('good_count', 0) or 0
        }
    
    def add_dataset_sample(self, sample_data: Dict) -> int:
        """
        Thêm sample vào dataset.
        
        Args:
            sample_data: Dictionary chứa thông tin sample
            
        Returns:
            ID của sample
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO dataset_samples (
                image_name, image_path, category, grade, features_json
            ) VALUES (?, ?, ?, ?, ?)
        ''', (
            sample_data.get('image_name', ''),
            sample_data.get('image_path', ''),
            sample_data.get('category', ''),
            sample_data.get('grade', ''),
            json.dumps(sample_data.get('features', {}))
        ))
        
        sample_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        logger.info(f"Dataset sample added: {sample_id}")
        return sample_id
    
    def get_dataset_samples(self, category: Optional[str] = None,
                          limit: int = 100) -> List[Dict]:
        """
        Lấy samples từ dataset.
        
        Args:
            category: Lọc theo category (ripe, unripe, defective)
            limit: Số lượng tối đa
            
        Returns:
            List chứa các samples
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        query = 'SELECT * FROM dataset_samples'
        params = []
        
        if category:
            query += ' WHERE category = ?'
            params.append(category)
        
        query += ' ORDER BY added_date DESC LIMIT ?'
        params.append(limit)
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        conn.close()
        
        return [dict(row) for row in rows]
    
    def save_setting(self, key: str, value: Any):
        """
        Lưu cài đặt.
        
        Args:
            key: Tên cài đặt
            value: Giá trị
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO settings (key, value, updated_date)
            VALUES (?, ?, CURRENT_TIMESTAMP)
        ''', (key, json.dumps(value)))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Setting saved: {key}")
    
    def get_setting(self, key: str, default: Any = None) -> Any:
        """
        Lấy giá trị cài đặt.
        
        Args:
            key: Tên cài đặt
            default: Giá trị mặc định
            
        Returns:
            Giá trị cài đặt hoặc default
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT value FROM settings WHERE key = ?', (key,))
        row = cursor.fetchone()
        
        conn.close()
        
        if row:
            return json.loads(row['value'])
        return default
    
    def clear_history(self) -> int:
        """
        Xóa toàn bộ lịch sử.
        
        Returns:
            Số records đã xóa
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT COUNT(*) FROM analysis_history')
        count = cursor.fetchone()[0]
        
        cursor.execute('DELETE FROM analysis_history')
        
        conn.commit()
        conn.close()
        
        logger.info(f"Cleared {count} history records")
        return count


# Singleton instance
_db_instance: Optional[Database] = None


def get_database(db_path: str = "tomato_quality.db") -> Database:
    """
    Lấy singleton instance của Database.
    
    Args:
        db_path: Đường dẫn database
        
    Returns:
        Database instance
    """
    global _db_instance
    
    if _db_instance is None:
        _db_instance = Database(db_path)
    
    return _db_instance


if __name__ == "__main__":
    # Test database
    db = Database("test.db")
    
    # Save test analysis
    test_data = {
        'image_name': 'test.jpg',
        'image_path': '/uploads/test.jpg',
        'ripeness_score': 85.5,
        'quality_score': 90.0,
        'combined_score': 87.3,
        'grade': 'A',
        'grade_full': 'Loại A - Đạt tiêu chuẩn',
        'red_ratio': 85.0,
        'green_ratio': 5.0,
        'yellow_ratio': 8.0,
        'brown_ratio': 2.0,
        'defect_ratio': 2.0,
        'circularity': 0.92,
        'solidity': 0.95,
        'lbp_entropy': 3.2,
        'glcm_homogeneity_avg': 0.75,
        'has_defects': False,
        'defect_severity': 'none',
        'features': {'test': 'data'},
        'result': {'method': 'rule_based'}
    }
    
    analysis_id = db.save_analysis(test_data)
    print(f"Saved analysis with ID: {analysis_id}")
    
    # Get history
    history = db.get_analysis_history()
    print(f"History records: {len(history)}")
    
    # Get statistics
    stats = db.get_statistics()
    print(f"Statistics: {stats}")
    
    print("Database tests passed!")
