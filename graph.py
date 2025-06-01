import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import tensorflow as tf


# Load the saved model
loaded_model = tf.keras.models.load_model("my_model.h5")

# Function to evaluate the model
def evaluate_model(model, test_data):
    results = model.evaluate(test_data, verbose=0)
    loss = results[0]
    acc = results[1]

    print("    Test Loss: {:.5f}".format(loss))
    print("Test Accuracy: {:.2f}%".format(acc * 100))

    y_pred = np.squeeze((model.predict(test_data) >= 0.5).astype(np.int))
    cm = confusion_matrix(test_data.labels, y_pred)
    clr = classification_report(test_data.labels, y_pred, target_names=["Negative", "Positive"])

    # Plot and save confusion matrix
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='g', vmin=0, cmap='Blues', cbar=False)
    plt.xticks(ticks=np.arange(2) + 0.5, labels=["Negative", "Positive"])
    plt.yticks(ticks=np.arange(2) + 0.5, labels=["Negative", "Positive"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")  # Save the plot as an image

    # Print classification report
    print("Classification Report:\n----------------------\n", clr)

    # Plot and save accuracy graph
    fig = px.line(
        history.history,
        y=['accuracy', 'val_accuracy'],
        labels={'index': "Epoch", 'value': "Accuracy"},
        title="Training and Validation Accuracy Over Time"
    )
    fig.write_image("accuracy_graph.png", width=800, height=600, scale=2)  # Save the plot as png
