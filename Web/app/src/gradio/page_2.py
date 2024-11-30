import gradio as gr


__all__ = ["page_2"]


def greet(name: str):
    return "Hello " + name + "!"


with gr.Blocks() as page_2:
    name = gr.Textbox(label="Name")
    output = gr.Textbox(label="Output Box")
    greet_btn = gr.Button("Greet")
    greet_btn.click(fn=greet, inputs=name, outputs=output, api_name="Click button", show_api=True)

