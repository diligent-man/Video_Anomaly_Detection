import gradio as gr

__all__ = ["page_1"]


def greet(name, intensity):
    return "Hello, " + name + "!" * int(intensity)


page_1 = gr.Interface(
    fn=greet,
    inputs=["text", "slider"],
    outputs=["text"],
)
