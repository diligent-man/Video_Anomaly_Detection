import requests as rq
from typing import Dict
from fastapi import File


def main() -> None:
    files: Dict[str, File] = {
        "file": open("./crawled_assault_000002.mp4", "rb")
    }

    response = rq.post(url="http://0.0.0.0:6968/infer", files=files)
    print(response.json())
    return None


if __name__ == '__main__':
    main()
