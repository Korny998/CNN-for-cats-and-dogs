import os

# Project directories
PROJECT_DIR: str = os.path.dirname(__file__)
BASE_DIR: str = os.path.join(PROJECT_DIR, 'dataset')

# Training parameters
BATCH_SIZE: int = 24
CLASS_LIST: list[str] = ['cats', 'dogs']
CLASS_COUNT: int = len(CLASS_LIST)

# Paths to the image dataset
IMAGE_PATH: str = os.path.join(
    PROJECT_DIR, 'data', 'training_set', 'training_set'
)
