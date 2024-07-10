import cv2
import os
import tensorflow as tf
from preprocessor import prepare
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def main():
    print('OpenCV version {} '.format(cv2.__version__))

    current_dir = os.path.dirname(__file__)

    author = '024' #testing with different dataset
    training_folder = os.path.join(current_dir, 'data/training/', author)
    test_folder = os.path.join(current_dir, 'data/test/', author)

    training_data = []
    training_labels = []
    for filename in os.listdir(training_folder):
        img = cv2.imread(os.path.join(training_folder, filename), 0)
        if img is not None:
            data = prepare(img)
            data = np.array(data).reshape(-1)
            training_data.append(data)
            training_labels.append([0, 1] if "genuine" in filename else [1, 0])

    test_data = []
    test_labels = []
    for filename in os.listdir(test_folder):
        img = cv2.imread(os.path.join(test_folder, filename), 0)
        if img is not None:
            data = prepare(img)
            data = np.array(data).reshape(-1)
            test_data.append(data)
            test_labels.append([0, 1] if "genuine" in filename else [1, 0])

    sgd(training_data, training_labels, test_data, test_labels)

def regression(x):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(2, activation='softmax', input_shape=(901,))
    ])
    return model

def sgd(training_data, training_labels, test_data, test_labels):
    x_train = np.array(training_data)
    y_train = np.array(training_labels)
    x_test = np.array(test_data)
    y_test = np.array(test_labels)

    model = regression(x_train)

    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=50, batch_size=32)

    loss, accuracy = model.evaluate(x_test, y_test)
    print("Test accuracy:", accuracy)

      # Get predictions
    y_pred = model.predict(x_test)
    y_pred_labels = np.argmax(y_pred, axis=1)
    y_true_labels = np.argmax(y_test, axis=1)

    # Confusion matrix
    cm = confusion_matrix(y_true_labels, y_pred_labels)
    print("Confusion Matrix:")
    print(cm)

    # Plot confusion matrix as a heatmap
    classes = ['Genuine', 'Forged']
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()

if __name__ == '__main__':
    main()