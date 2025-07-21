# Vision-Language Models for Cue Sports

This directory contains implementations and interfaces for using vision-language models to understand and analyze cue sports through both visual and textual understanding.

## Features

- **Scene Understanding**: Generate rich descriptions of game states
- **Natural Language Queries**: Ask questions about the game state in natural language
- **Rule-Based Reasoning**: Understand and apply game rules through language models
- **Shot Recommendation**: Get AI-powered shot suggestions
- **Game Commentary**: Generate natural language commentary on game events

## Structure

```
VLMInference/
├── models/          # Pre-trained VLM models
├── prompts/         # Prompt templates and few-shot examples
├── interfaces/      # APIs and interfaces for interaction
├── evaluation/      # Evaluation metrics and scripts
└── utils/           # Utility functions
```

## Usage

```python
from VLMInference.models import load_vlm
from VLMInference.interfaces import GameAnalyzer

# Load a pre-trained vision-language model
model = load_vlm('blip2')

# Initialize game analyzer
analyzer = GameAnalyzer(model)

# Analyze a game frame
frame = load_image('game_frame.jpg')
analysis = analyzer.analyze_frame(frame, query="Describe the current game state")
print(analysis)
```

## Dependencies
- Transformers (Hugging Face)
- PyTorch
- OpenCV
- PIL
