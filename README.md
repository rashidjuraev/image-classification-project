# Classification Models

A comprehensive repository of various image classification models implemented in PyTorch.

## Models Implemented

- Xception
- Inception (GoogleNet)
- ResNet
- MobileNetV2
- VGG
- SqueezeNet

## Project Structure

```
Classification-Models/
├── models/               # Model architectures
├── data/                 # Data processing utilities
├── utils/                # Helper utilities
├── train.py              # Training script
├── test.py               # Testing script
├── eval.py               # Evaluation script
├── config.py             # Configuration
├── requirements.txt      # Dependencies
└── README.md             # Documentation
```

## Installation

```bash
git clone https://github.com/yourusername/Classification-Models.git
cd Classification-Models
pip install -r requirements.txt
```

## Usage

### Training a Model

```bash
python train.py --model xception
```

### Resume Training from Checkpoint

```bash
python train.py --model xception --resume --checkpoint checkpoints/xception/xception_best.pth
```

### Testing a Model

```bash
python test.py --model xception
```

### Evaluating on a New Image

```bash
python eval.py --model xception --image path/to/image.jpg --class_names path/to/class_names.txt
```

## Model Configuration

Model configurations are defined in `config.py`. You can modify the configurations to fit your needs.

## Dataset

The repository is designed to work with the face expression recognition dataset by default, but it can be easily adapted to work with any image classification dataset by modifying the dataset classes.

## Requirements

See `requirements.txt` for dependencies.

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.