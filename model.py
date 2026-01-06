from keras import layers, models


def build_model() -> models.Sequential:
    """
    Build and return a CNN model for binary classification of cats vs dogs.
    """
    return models.Sequential([
        layers.Input(shape=(150, 150, 3)),

        layers.Conv2D(
            32, (3, 3),
            activation='relu',
            name='Conv_1'
        ),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(
            64, (3, 3),
            activation='relu',
            name='Conv_2'
        ),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(
            128, (3, 3),
            activation='relu',
            name='Conv_3'
        ),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(
            128, (3, 3),
            activation='relu',
            name='Conv_4'
        ),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),

        layers.Dropout(
            0.5,
            name='Dropout'
        ),

        layers.Dense(
            512,
            activation='relu',
            name='Dense_1'
        ),

        layers.Dense(
            1,
            activation='sigmoid',
            name='Dense_2'
        )
    ])
