# from fastapi import FastAPI
# import gradio as gr
# 
# CUSTOM_PATH_1 = "/gradio_1"
# CUSTOM_PATH_2 = "/gradio_2"
# 
# app = FastAPI()
# 
# 
# @app.get("/")
# def read_main():
#     return {"message": "This is your main app"}
# 
# 
# io1 = gr.Interface(lambda x: "Hello, " + x + "!", "textbox", "textbox")
# io2 = gr.Interface(lambda x: "Hello new world, " + x + "!", "textbox", "textbox")
# 
# app = gr.mount_gradio_app(app, io1, path=CUSTOM_PATH_1)
# app = gr.mount_gradio_app(app, io2, path=CUSTOM_PATH_2)

import torch


def main() -> None:
    print(torch.cuda.is_available())
    return None


if __name__ == '__main__':
    main()