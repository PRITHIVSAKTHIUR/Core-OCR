import os
import random
import uuid
import json
import time
import asyncio
from threading import Thread

import gradio as gr
import spaces
import torch
import numpy as np
from PIL import Image
import cv2

from transformers import (
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    AutoModelForImageTextToText,
    AutoProcessor,
    TextIteratorStreamer,
)
from transformers.image_utils import load_image

# Constants for text generation
MAX_MAX_NEW_TOKENS = 2048
DEFAULT_MAX_NEW_TOKENS = 1024
MAX_INPUT_TOKEN_LENGTH = int(os.getenv("MAX_INPUT_TOKEN_LENGTH", "4096"))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load docscopeOCR-7B-050425-exp
MODEL_ID_M = "prithivMLmods/docscopeOCR-7B-050425-exp"
processor_m = AutoProcessor.from_pretrained(MODEL_ID_M, trust_remote_code=True)
model_m = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID_M,
    trust_remote_code=True,
    torch_dtype=torch.float16
).to(device).eval()

# Load coreOCR-7B-050325-preview
MODEL_ID_X = "prithivMLmods/coreOCR-7B-050325-preview"
processor_x = AutoProcessor.from_pretrained(MODEL_ID_X, trust_remote_code=True)
model_x = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_ID_X,
    trust_remote_code=True,
    torch_dtype=torch.float16
).to(device).eval()

#--------------------------------------------------------------------------------------#
#Load MonkeyOCR
MODEL_ID_G = "echo840/MonkeyOCR"
SUBFOLDER = "Recognition"

processor_g = AutoProcessor.from_pretrained(
    MODEL_ID_G,
    trust_remote_code=True,
    subfolder=SUBFOLDER
)

model_g = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID_G,
    trust_remote_code=True,
    subfolder=SUBFOLDER,
    torch_dtype=torch.float16
).to(device).eval()
#--------------------------------------------------------------------------------------#

def downsample_video(video_path):
    """
    Downsamples the video to evenly spaced frames.
    Each frame is returned as a PIL image along with its timestamp.
    """
    vidcap = cv2.VideoCapture(video_path)
    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    frames = []
    frame_indices = np.linspace(0, total_frames - 1, 10, dtype=int)
    for i in frame_indices:
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, i)
        success, image = vidcap.read()
        if success:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image)
            timestamp = round(i / fps, 2)
            frames.append((pil_image, timestamp))
    vidcap.release()
    return frames

@spaces.GPU
def generate_image(model_name: str, text: str, image: Image.Image,
                   max_new_tokens: int = 1024,
                   temperature: float = 0.6,
                   top_p: float = 0.9,
                   top_k: int = 50,
                   repetition_penalty: float = 1.2):
    """
    Generates responses using the selected model for image input.
    """
    if model_name == "docscopeOCR-7B-050425-exp":
        processor = processor_m
        model = model_m
    elif model_name == "coreOCR-7B-050325-preview":
        processor = processor_x
        model = model_x
    elif model_name == "MonkeyOCR-Recognition":
        processor = processor_g
        model = model_g
    else:
        yield "Invalid model selected."
        return

    if image is None:
        yield "Please upload an image."
        return

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": text},
        ]
    }]
    prompt_full = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[prompt_full],
        images=[image],
        return_tensors="pt",
        padding=True,
        truncation=False,
        max_length=MAX_INPUT_TOKEN_LENGTH
    ).to(device)
    streamer = TextIteratorStreamer(processor, skip_prompt=True, skip_special_tokens=True)
    generation_kwargs = {**inputs, "streamer": streamer, "max_new_tokens": max_new_tokens}
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    buffer = ""
    for new_text in streamer:
        buffer += new_text
        buffer = buffer.replace("<|im_end|>", "")
        time.sleep(0.01)
        yield buffer

@spaces.GPU
def generate_video(model_name: str, text: str, video_path: str,
                   max_new_tokens: int = 1024,
                   temperature: float = 0.6,
                   top_p: float = 0.9,
                   top_k: int = 50,
                   repetition_penalty: float = 1.2):
    """
    Generates responses using the selected model for video input.
    """
    if model_name == "docscopeOCR-7B-050425-exp":
        processor = processor_m
        model = model_m
    elif model_name == "coreOCR-7B-050325-preview":
        processor = processor_x
        model = model_x
    elif model_name == "MonkeyOCR-Recognition":
        processor = processor_g
        model = model_g
    else:
        yield "Invalid model selected."
        return

    if video_path is None:
        yield "Please upload a video."
        return

    frames = downsample_video(video_path)
    messages = [
        {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
        {"role": "user", "content": [{"type": "text", "text": text}]}
    ]
    for frame in frames:
        image, timestamp = frame
        messages[1]["content"].append({"type": "text", "text": f"Frame {timestamp}:"})
        messages[1]["content"].append({"type": "image", "image": image})
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        truncation=False,
        max_length=MAX_INPUT_TOKEN_LENGTH
    ).to(device)
    streamer = TextIteratorStreamer(processor, skip_prompt=True, skip_special_tokens=True)
    generation_kwargs = {
        **inputs,
        "streamer": streamer,
        "max_new_tokens": max_new_tokens,
        "do_sample": True,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "repetition_penalty": repetition_penalty,
    }
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    buffer = ""
    for new_text in streamer:
        buffer += new_text
        buffer = buffer.replace("<|im_end|>", "")
        time.sleep(0.01)
        yield buffer

# Define examples for image and video inference
image_examples = [
    ["fill the correct numbers", "example/image3.png"],
    ["ocr the image", "example/image1.png"],
    ["explain the scene", "example/image2.jpg"],
]

video_examples = [
    ["Explain the ad in detail", "example/1.mp4"],
    ["Identify the main actions in the coca cola ad...", "example/2.mp4"]
]

css = """
.submit-btn {
    background-color: #2980b9 !important;
    color: white !important;
}
.submit-btn:hover {
    background-color: #3498db !important;
}
"""

# Create the Gradio Interface
with gr.Blocks(css=css, theme="bethecloud/storj_theme") as demo:
    gr.Markdown("# **[core OCR](https://huggingface.co/collections/prithivMLmods/core-and-docscope-ocr-models-6816d7f1bde3f911c6c852bc)**")
    with gr.Row():
        with gr.Column():
            with gr.Tabs():
                with gr.TabItem("Image Inference"):
                    image_query = gr.Textbox(label="Query Input", placeholder="Enter your query here...")
                    image_upload = gr.Image(type="pil", label="Image")
                    image_submit = gr.Button("Submit", elem_classes="submit-btn")
                    gr.Examples(
                        examples=image_examples,
                        inputs=[image_query, image_upload]
                    )
                with gr.TabItem("Video Inference"):
                    video_query = gr.Textbox(label="Query Input", placeholder="Enter your query here...")
                    video_upload = gr.Video(label="Video")
                    video_submit = gr.Button("Submit", elem_classes="submit-btn")
                    gr.Examples(
                        examples=video_examples,
                        inputs=[video_query, video_upload]
                    )
            with gr.Accordion("Advanced options", open=False):
                max_new_tokens = gr.Slider(label="Max new tokens", minimum=1, maximum=MAX_MAX_NEW_TOKENS, step=1, value=DEFAULT_MAX_NEW_TOKENS)
                temperature = gr.Slider(label="Temperature", minimum=0.1, maximum=4.0, step=0.1, value=0.6)
                top_p = gr.Slider(label="Top-p (nucleus sampling)", minimum=0.05, maximum=1.0, step=0.05, value=0.9)
                top_k = gr.Slider(label="Top-k", minimum=1, maximum=1000, step=1, value=50)
                repetition_penalty = gr.Slider(label="Repetition penalty", minimum=1.0, maximum=2.0, step=0.05, value=1.2)
        with gr.Column():
            output = gr.Textbox(label="Output", interactive=False, lines=3, scale=2)
            model_choice = gr.Radio(
                choices=["docscopeOCR-7B-050425-exp", "MonkeyOCR-Recognition", "coreOCR-7B-050325-preview"],
                label="Select Model",
                value="docscopeOCR-7B-050425-exp"
            )
            
            gr.Markdown("**Model Info**")
            gr.Markdown("> [docscopeOCR-7B-050425-exp](https://huggingface.co/prithivMLmods/docscopeOCR-7B-050425-exp): The docscopeOCR-7B-050425-exp model is a fine-tuned version of Qwen2.5-VL-7B-Instruct, optimized for Document-Level Optical Character Recognition (OCR), long-context vision-language understanding, and accurate image-to-text conversion with mathematical LaTeX formatting.")
            gr.Markdown("> [MonkeyOCR](https://huggingface.co/echo840/MonkeyOCR): MonkeyOCR adopts a Structure-Recognition-Relation (SRR) triplet paradigm, which simplifies the multi-tool pipeline of modular approaches while avoiding the inefficiency of using large multimodal models for full-page document processing.")
            gr.Markdown("> [coreOCR-7B-050325-preview](https://huggingface.co/prithivMLmods/coreOCR-7B-050325-preview): The coreOCR-7B-050325-preview model is a fine-tuned version of Qwen2-VL-7B, optimized for Document-Level Optical Character Recognition (OCR), long-context vision-language understanding, and accurate image-to-text conversion with mathematical LaTeX formatting.")

    image_submit.click(
        fn=generate_image,
        inputs=[model_choice, image_query, image_upload, max_new_tokens, temperature, top_p, top_k, repetition_penalty],
        outputs=output
    )
    video_submit.click(
        fn=generate_video,
        inputs=[model_choice, video_query, video_upload, max_new_tokens, temperature, top_p, top_k, repetition_penalty],
        outputs=output
    )

if __name__ == "__main__":
    demo.queue(max_size=30).launch(share=True, mcp_server=True, ssr_mode=False, show_error=True)