import uvicorn
import os


def main() -> None:
    # Tăng timeout và worker để xử lý file lớn
    uvicorn.run(
        "src.app:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 6968)),
        timeout_keep_alive=3600,
        log_level="info"
    )
    return None


if __name__ == '__main__':
    main()
