import os
import requests


def main() -> None:
    usr = os.getenv("MLFLOW_ADMIN_USERNAME")
    pwd = os.getenv("MLFLOW_ADMIN_PASSWORD")

    if requests.get("http://0.0.0.0:5000/", auth=(usr, pwd)).status_code == 200:
        print("true")
    return None


if __name__ == '__main__':
    main()