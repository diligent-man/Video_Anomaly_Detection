import uvicorn
from fastapi import FastAPI



def main() -> None:
    uvicorn.run("src:app", host="0.0.0.0", port=8000)
    return None


if __name__ == '__main__':
    main()
