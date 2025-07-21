# Physics Inference for Cue Sports

A research project focused on analyzing cue sports (like snooker) using computer vision and machine learning techniques to understand game physics including object detection, position tracking, speed analysis, and collision prediction.

## Project Structure

### 1. PureCV
This directory contains implementations using traditional computer vision techniques without deep learning dependencies. Focus areas include:
- Ball detection and tracking
- Table detection and perspective correction
- Cue detection and angle estimation
- Physics-based motion analysis
- Basic collision detection

### 2. CVModelInference
This directory focuses on deep learning-based approaches for understanding cue sports:
- Object detection models for balls, cues, and table
- Instance segmentation for precise object boundaries
- Optical flow for motion analysis
- Trajectory prediction models
- Physics-informed neural networks

### 3. VLMInference
This directory explores Vision-Language Models (VLMs) for advanced understanding:
- Scene description and analysis
- Shot prediction and recommendation
- Rule-based reasoning about game state
- Natural language querying of game state
- Action recognition and classification

## Getting Started

### Prerequisites
- Python 3.8+
- OpenCV
- PyTorch or TensorFlow (for CVModelInference and VLMInference)
- Additional dependencies listed in requirements.txt

### Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/physics_inference.git
cd physics_inference

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt
```

## Usage
Each directory contains its own README with specific usage instructions and examples.

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- [OpenCV](https://opencv.org/)
- [PyTorch](https://pytorch.org/)
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- Related research papers and open-source projects
