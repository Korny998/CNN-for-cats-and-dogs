import matplotlib.pyplot as plt

from keras.optimizers import RMSprop

from dataset import train_generator, validation_generator
from model import build_model


def train():
    """Compile and train the CNN model."""
    model = build_model()

    model.compile(
        optimizer=RMSprop(learning_rate=1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    history = model.fit(
        train_generator,
        epochs=50,
        validation_data=validation_generator
    )

    model.save('cats_and_dogs_small_model.h5')
    return history


def accuracy_and_loss(history) -> None:
    """Plot accuracy and loss graphs for training and validation sets."""
    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(accuracy) + 1)
    plt.plot(epochs, accuracy, 'r', label='Accuracy in the training sample')
    plt.plot(epochs, val_accuracy, 'bo', label='Accuracy in the test sample')
    plt.title('Accuracy graph for training and validation samples')
    plt.legend()
    plt.figure()
    plt.plot(epochs, loss, 'r', label='Losses in the training sample')
    plt.plot(epochs, val_loss, 'bo', label='Losses in the validation sample')
    plt.title('Loss graph for training and validation samples')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    history = train()
    accuracy_and_loss(history)
    # show_batch(validation_generator[0])
    # # Uncomment to visualize a batch of validation images
