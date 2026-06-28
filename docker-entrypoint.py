"""
TomatoVision Docker Entrypoint
Khởi động ứng dụng trong container Docker
"""
import os
import sys
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Khởi động Uvicorn server."""
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    workers = int(os.environ.get("WORKERS", "1"))
    reload = os.environ.get("RELOAD", "false").lower() == "true"

    logger.info(f"Starting TomatoVision on {host}:{port}")
    logger.info(f"Workers: {workers}, Reload: {reload}")

    import uvicorn
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        workers=workers,
        reload=reload,
        log_level="info"
    )


if __name__ == "__main__":
    main()
