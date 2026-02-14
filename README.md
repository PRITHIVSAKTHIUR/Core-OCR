# **Core OCR**

> [!note]
HF Demo: https://huggingface.co/spaces/prithivMLmods/core-OCR


A specialized optical character recognition (OCR) application built on advanced vision-language models, designed for document-level OCR, long-context understanding, and mathematical LaTeX formatting. Supports both image and video processing with multiple state-of-the-art models.

> [!important] 
note: remove kernels and flash_attn3 implementation if you are using it on *non-hopper* architecture gpus.

 <img width="1757" height="1225" alt="Screenshot 2025-10-16 at 11-41-48 core OCR - a Hugging Face Space by prithivMLmods" src="https://github.com/user-attachments/assets/cfeadabe-c2ff-4d38-96e8-00da4ce61768" />

## Features

- **Advanced OCR Models**: Three specialized models optimized for different document processing tasks
- **Document-Level OCR**: Comprehensive text extraction with context understanding
- **Mathematical LaTeX Support**: Accurate conversion of mathematical expressions to LaTeX format
- **Image & Video Processing**: Handle both static images and video content
- **Interactive Web Interface**: User-friendly Gradio interface with real-time streaming
- **Long-Context Understanding**: Process complex documents with extended context
- **Structure-Recognition-Relation**: Advanced document understanding paradigm

## Supported Models

### docscopeOCR-7B-050425-exp
Fine-tuned version of Qwen2.5-VL-7B-Instruct, optimized for Document-Level Optical Character Recognition (OCR), long-context vision-language understanding, and accurate image-to-text conversion with mathematical LaTeX formatting.

### MonkeyOCR-Recognition
Adopts a Structure-Recognition-Relation (SRR) triplet paradigm, which simplifies the multi-tool pipeline of modular approaches while avoiding the inefficiency of using large multimodal models for full-page document processing.

### coreOCR-7B-050325-preview
Fine-tuned version of Qwen2-VL-7B, optimized for Document-Level Optical Character Recognition (OCR), long-context vision-language understanding, and accurate image-to-text conversion with mathematical LaTeX formatting.

---

## Image Inference

![ROAfZvsoUpDMB1YzezCnU](https://github.com/user-attachments/assets/4e1524dd-bde9-4cd1-9ac6-2ca2453490e2)

---

![hT8d2059Jkke4sAjD1fyj](https://github.com/user-attachments/assets/01a6105b-5468-404d-86bb-53bc673de639)

---

![HmCbs0HRM4uuqaUDaZz3M](https://github.com/user-attachments/assets/f3d0a6a7-aba8-4eda-b069-3bc9772e3d90)

---

## Video Inference

https://github.com/user-attachments/assets/291e9ef0-d7a5-49ca-a90a-190c90e48df2

---

## Installation

1. Clone the repository:
```bash
git clone https://github.com/PRITHIVSAKTHIUR/Core-OCR.git
cd Core-OCR
```

2. Install required dependencies:
```bash
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
gradio # - gradio@6.3.0
av
```

## Usage

### Running the Application

```bash
python app.py
```

The application will launch a Gradio interface accessible through your web browser at the displayed URL.

### Image OCR Processing

1. Navigate to the "Image Inference" tab
2. Enter your query in the text box:
   - "fill the correct numbers" - For mathematical problem solving
   - "ocr the image" - For general text extraction
   - "explain the scene" - For comprehensive image analysis
3. Upload an image file (PNG, JPG, JPEG supported)
4. Select your preferred model from the radio buttons
5. Click "Submit" to process

### Video OCR Processing

1. Navigate to the "Video Inference" tab
2. Enter your query:
   - "Explain the ad in detail" - For advertisement analysis
   - "Identify the main actions in the coca cola ad..." - For action recognition
3. Upload a video file (MP4, AVI, MOV supported)
4. Select your preferred model
5. Click "Submit" to process

The application automatically extracts 10 evenly distributed frames from videos for comprehensive analysis.

## Advanced Configuration

### Generation Parameters

- **Max New Tokens**: Control response length (1-2048 tokens)
- **Temperature**: Adjust creativity/randomness (0.1-4.0)
- **Top-p**: Configure nucleus sampling (0.05-1.0)
- **Top-k**: Set vocabulary consideration range (1-1000)
- **Repetition Penalty**: Prevent repetitive outputs (1.0-2.0)

### Model Selection Guidelines

- **docscopeOCR-7B-050425-exp**: Best for complex documents with mathematical content
- **MonkeyOCR-Recognition**: Optimal for structured document processing with relation understanding
- **coreOCR-7B-050325-preview**: Excellent for general OCR tasks with long-context requirements

## Technical Specifications

### Hardware Requirements

- **GPU**: CUDA-compatible GPU recommended (minimum 8GB VRAM)
- **RAM**: 16GB+ system memory recommended
- **Storage**: 30GB+ free space for model downloads
- **CPU**: Multi-core processor for video processing

### Model Architecture

- **Base Models**: Qwen2-VL and Qwen2.5-VL architectures
- **Precision**: Half-precision (float16) for memory efficiency
- **Context Length**: Up to 4096 tokens input context
- **Streaming**: Real-time text generation with TextIteratorStreamer

### Video Processing Details

- **Frame Extraction**: 10 evenly distributed frames per video
- **Timestamp Preservation**: Each frame tagged with temporal information
- **Format Support**: MP4, AVI, MOV, and other OpenCV-compatible formats
- **Resolution**: Maintains original aspect ratio with PIL processing

## Examples and Use Cases

### Document Processing
- Academic papers with mathematical equations
- Financial documents with tables and charts
- Legal documents with complex formatting
- Technical manuals with diagrams

### Mathematical Content
- LaTeX equation extraction
- Formula recognition and conversion
- Mathematical problem solving
- Scientific notation processing

### Video Analysis
- Advertisement content analysis
- Educational video transcription
- Presentation slide extraction
- Tutorial step identification

## API Reference

### Core Functions

```python
generate_image(model_name, text, image, max_new_tokens, temperature, top_p, top_k, repetition_penalty)
```
Processes single images with specified model and parameters.

```python
generate_video(model_name, text, video_path, max_new_tokens, temperature, top_p, top_k, repetition_penalty)
```
Processes video files with frame extraction and temporal analysis.

```python
downsample_video(video_path)
```
Extracts evenly distributed frames from video files with timestamps.

## Performance Optimization

### Memory Management
- Automatic GPU memory optimization
- Model loading with efficient tensor operations
- Batch processing for multiple requests

### Processing Speed
- Asynchronous text generation
- Threaded model inference
- Optimized frame extraction algorithms

## Troubleshooting

### Common Issues

1. **GPU Memory Error**: Reduce max_new_tokens or use CPU inference
2. **Model Loading Failed**: Ensure sufficient disk space and internet connection
3. **Video Processing Slow**: Consider reducing video resolution or length
4. **Out of Memory**: Lower batch size or use smaller models

### Performance Tips

- Use GPU acceleration when available
- Optimize generation parameters for your use case
- Consider model size vs. accuracy trade-offs
- Monitor system resources during processing

## Contributing

We welcome contributions to improve Core OCR:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

### Development Setup

```bash
git clone https://github.com/PRITHIVSAKTHIUR/Core-OCR.git
cd Core-OCR
pip install -e .
```

## Repository

GitHub: https://github.com/PRITHIVSAKTHIUR/Core-OCR.git

## License

This project is open source. Please refer to the LICENSE file for specific terms and conditions.

## Citation

If you use Core OCR in your research or projects, please cite:

```bibtex
@software{core_ocr_2024,
  title={Core OCR: Advanced Document-Level OCR with Vision-Language Models},
  author={PRITHIVSAKTHIUR},
  year={2024},
  url={https://github.com/PRITHIVSAKTHIUR/Core-OCR}
}
```

## Acknowledgments

- Hugging Face for the Transformers library and model hosting
- Qwen team for the base vision-language models
- Gradio for the web interface framework
- OpenCV community for video processing capabilities
- The broader OCR and computer vision research community

## Support

For questions, issues, or feature requests:
- Open an issue on GitHub
- Check existing documentation
- Review model-specific guides on Hugging Face


