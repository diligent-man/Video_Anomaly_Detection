import uvicorn


def main() -> None:
    uvicorn.run("src:app", host="0.0.0.0", port=6969, reload=True)
    return None


if __name__ == '__main__':
    main()
