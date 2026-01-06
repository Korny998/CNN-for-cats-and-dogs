# CNN for Cats and Dogs

## Project Description
This project implements a Convolutional Neural Network (CNN) for classifying images of cats and dogs.  
The model uses `TensorFlow` and `Keras` and is trained on a subset of the [Cats and Dogs dataset](https://storage.yandexcloud.net/academy.ai/cat-and-dog.zip).

Key features:
- Image preprocessing with `ImageDataGenerator`
- Data augmentation for training
- CNN architecture with multiple convolutional and pooling layers
- Training with `binary_crossentropy` loss and `RMSprop` optimizer
- Accuracy and loss visualization
- Saved trained model for future inference

---

## Table of Contents
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Requirements](#requirements)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/Korny998/CNN-for-cats-and-dogs.git
cd CNN-for-cats-and-dogs
```

2. Create a virtual environment:

```bash
python -m venv venv
```

3. Activate the environment:

Windows:

```bash
venv\Scripts\activate
```

Linux / macOS:

```bash
source venv/bin/activate
```

4. Install dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Usage

1. **Download and preprocess the dataset:**

```bash
python dataset.py
```

2. **Train the CNN model:**

```bash
python train.py
```

3. **You can visualize a batch of confirmation images:**

```bash
python image_example.py
```

## Model Architecture

The CNN model consists of the following layers:

1. Input layer: 150x150x3
2. Convolution + ReLU + MaxPooling:
    - Conv2D: 32 filters
    - MaxPooling2D
    - Conv2D: 64 filters
    - MaxPooling2D
    - Conv2D: 128 filters
    - MaxPooling2D
    - Conv2D: 128 filters
    - MaxPooling2D
3. Flatten layer
4. Dropout layer: 0.5
5. Dense layer: 512 units, ReLU activation
6. Dense layer: 1 unit, Sigmoid activation (binary classification)

## Dataset

The dataset is downloaded automatically from:

```bash
https://storage.yandexcloud.net/academy.ai/cat-and-dog.zip
```

The dataset.py script splits the dataset into:
  - Train: 1000 images per class
  - Validation: 500 images per class
  - Test: 500 images per class