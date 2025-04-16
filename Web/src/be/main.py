import uvicorn


def main() -> None:
    # Tăng timeout và worker để xử lý file lớn
    uvicorn.run(
        "src.app:app",
        host="0.0.0.0", 
        port=6967,
        timeout_keep_alive=300,
        log_level="info"
    )
    return None


if __name__ == '__main__':
    main()
