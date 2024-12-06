import uvicorn

from fastapi import FastAPI


def main() -> None:
    uvicorn.run("src:app", host="0.0.0.0", port=6969, reload=False)
    return None


if __name__ == '__main__':
    main()
