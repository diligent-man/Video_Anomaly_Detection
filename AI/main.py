
import warnings
from dotenv import load_dotenv
import uvicorn


load_dotenv()
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def main() -> None:
    config_path = os.getenv("CONFIG_PATH")
    weight_path = os.getenv("WEIGHT_PATH")
    print("CONFIG_PATH:", config_path)
    print("WEIGHT_PATH:", weight_path)

    if not config_path or not os.path.exists(config_path):
        warnings.warn("Config path is wrong or not set, exiting program")
        exit()

    if not weight_path or not os.path.exists(weight_path):
        warnings.warn("Weight path is wrong or not set, exiting program")
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



# import os
# import warnings
# import sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# import uvicorn
# def main() -> None:
#     if not os.path.exists(os.environ["CONFIG_PATH"]):
#         warnings.warn(f"Config path is wrong so exit programme")
#         exit()

#     if not os.path.exists(os.environ["WEIGHT_PATH"]):
#         warnings.warn(f"Weight path is wrong so exit programme")
#         exit()

#     uvicorn.run(
#         "src.app:app",
#         host="0.0.0.0",
#         port=6968,
#         timeout_keep_alive=300,
#         log_level="info",
#     )
#     return None


# if __name__ == '__main__':
#     main()
