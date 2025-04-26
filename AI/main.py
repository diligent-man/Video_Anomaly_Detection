import os
import warnings
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import uvicorn
def main() -> None:
    if not os.path.exists(os.environ["CONFIG_PATH"]):
        warnings.warn(f"Config path is wrong so exit programme")
        exit()

    if not os.path.exists(os.environ["WEIGHT_PATH"]):
        warnings.warn(f"Weight path is wrong so exit programme")
        exit()

    uvicorn.run(
        "src.app:app",
        host="0.0.0.0",
        port=6968,
        timeout_keep_alive=300,
        log_level="info",
    )
    return None


if __name__ == '__main__':
    main()
