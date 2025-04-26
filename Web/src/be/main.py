import uvicorn
import sys

# Add Windows-specific event loop policy to fix connection errors
if sys.platform.startswith('win'):
    import asyncio
    # Use the following for Python 3.8+
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

def main() -> None:
    # Tăng timeout và worker để xử lý file lớn
    uvicorn.run(
        "src.app:app",
        host="0.0.0.0", 
        port=6967, 
        timeout_keep_alive=3600,
        log_level="info"
    )
    return None


if __name__ == '__main__':
    main()
