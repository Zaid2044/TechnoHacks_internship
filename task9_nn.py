import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
print("Dataset loaded. Training samples:", x_train.shape[0])

x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0

y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28, 1)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
print("Model compiled.")

history = model.fit(x_train, y_train, epochs=5, batch_size=32, validation_split=0.1)
print("Training complete.")

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=1)
print("Test Accuracy:", round(test_acc, 4))

indices = np.random.choice(x_test.shape[0], 5, replace=False)

for i, idx in enumerate(indices, 1):
    img = x_test[idx]
    true_label = np.argmax(y_test[idx])
    sample = np.expand_dims(img, axis=0)
    pred_label = np.argmax(model.predict(sample, verbose=0))
    print(f"Sample {i}: True Label = {true_label}, Predicted = {pred_label}")
    plt.imshow(img.squeeze(), cmap="gray")
    plt.title(f"True: {true_label} | Pred: {pred_label}")
    plt.axis("off")
    plt.savefig(f"prediction_{i}.png")
    plt.close()
