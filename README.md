# **Core-OCR**

Core-OCR is an advanced, experimental Optical Character Recognition (OCR) and document analysis suite designed for highly accurate text extraction, table reconstruction, and complex visual reasoning. Built on the robust Qwen2.5-VL and Qwen2-VL multimodal architectures, this application provides a modern, interactive web interface capable of processing a wide array of visual inputs, from standard documents and multilingual texts to structured tables and complex scene images. By integrating specialized state-of-the-art vision-language models—such as Camel-Doc-OCR, docscopeOCR, MonkeyOCR, and coreOCR—the tool empowers users to precisely reconstruct document formatting, perform fill-in-the-blank number extraction, and generate detailed scene explanations. The suite is fully GPU-accelerated and offers granular control over text generation parameters, creating an optimal environment for deploying and testing robust document intelligence workflows.

<img width="1920" height="1798" alt="Screenshot 2026-03-22 at 12-20-38 core OCR - a Hugging Face Space by prithivMLmods" src="https://github.com/user-attachments/assets/b93d88da-81c3-4354-9c30-263a43af1b3c" />

### **Key Features**

* **Multi-Model Architecture:** Seamlessly switch between specialized vision-language models directly from the interface. Supported models include `Camel-Doc-OCR-080125(v2)`, `docscopeOCR-7B-050425-exp`, `MonkeyOCR-Recognition`, and `coreOCR-7B-050325-preview`.
* **Custom User Interface:** Features a bespoke, responsive Gradio frontend built with custom HTML, CSS, and JavaScript. It includes a drag-and-drop media zone, real-time output streaming, and an integrated advanced settings panel.
* **Granular Inference Controls:** Fine-tune the AI's output by adjusting text generation parameters such as Maximum New Tokens, Temperature, Top-p, Top-k, and Repetition Penalty.
* **Output Management:** Built-in actions allow users to instantly copy the raw output text to their clipboard or save the generated response directly as a `.txt` file.
* **Flash Attention 2 Integration:** Utilizes `kernels-community/flash-attn2` for optimized, memory-efficient inference on compatible GPUs.

### **Repository Structure**

```text
├── images/
│   ├── 0.png
│   ├── doc.jpg
│   ├── image1.png
│   ├── image2.jpg
│   ├── image3.png
│   └── zh.png
├── app.py
├── LICENSE
├── pre-requirements.txt
├── README.md
└── requirements.txt
```

### **Installation and Requirements**

To run Core-OCR locally, you need to configure a Python environment with the following dependencies. Ensure you have a compatible CUDA-enabled GPU for optimal performance.

**1. Install Pre-requirements**
Run the following command to update pip to the required version:
```bash
pip install pip>=23.0.0
```

**2. Install Core Requirements**
Install the necessary machine learning and UI libraries. You can place these in a `requirements.txt` file and run `pip install -r requirements.txt`.

```text
git+https://github.com/huggingface/transformers.git@v4.57.6
git+https://github.com/huggingface/accelerate.git
git+https://github.com/huggingface/peft.git
transformers-stream-generator
huggingface_hub
qwen-vl-utils
sentencepiece
opencv-python
torch==2.8.0
torchvision
matplotlib
requests
kernels
hf_xet
spaces
pillow
gradio
av
```

### **Usage**

Once your environment is set up and the dependencies are installed, you can launch the application by running the main Python script:

```bash
python app.py
```

After the script initializes the interface, it will provide a local web address (usually `http://127.0.0.1:7860/`) which you can open in your browser to interact with the models. Note that the selected models will be downloaded and loaded into VRAM upon their first invocation.

### **License and Source**

* **License:** Apache License - Version 2.0
* **GitHub Repository:** [https://github.com/PRITHIVSAKTHIUR/Core-OCR](https://github.com/PRITHIVSAKTHIUR/Core-OCR)
