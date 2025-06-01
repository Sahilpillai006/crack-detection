import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from pathlib import Path
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report

# Define the root directory where the images are located
root_dir = Path("E:/Project/Project/Crack Detection/Crack/Model")

# Create a list to store the filepaths and labels
filepaths = []
labels = []

# Iterate over the subfolders (Negative and Positive)
for label in ["Negative", "Positive"]:
    folder = root_dir / label
    # Iterate over the files in the subfolder
    for file in folder.glob("*.jpg"):
        # Append the filepath and label to the lists
        filepaths.append(file)
        labels.append(label)

# Create a DataFrame from the lists
all_df = pd.DataFrame({
    "Filepath": filepaths,
    "Label": labels
})

# Convert Filepath column to strings
all_df['Filepath'] = all_df['Filepath'].astype(str)

# Shuffle the DataFrame
all_df = all_df.sample(frac=1, random_state=1).reset_index(drop=True)

# Display the first few rows of the DataFrame
print(all_df.head())

# Split the dataset into training and testing sets
train_df, test_df = train_test_split(
    all_df,
    train_size=0.7,
    shuffle=True,
    random_state=1
)

# Data augmentation and preprocessing for training set
train_gen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

# Data preprocessing for validation and testing sets
val_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
test_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

# Create data generators for training, validation, and testing sets
train_data = train_gen.flow_from_dataframe(
    train_df,
    x_col='Filepath',
    y_col='Label',
    target_size=(120, 120),
    color_mode='rgb',
    class_mode='binary',
    batch_size=32,
    shuffle=True,
    seed=42,
    subset='training'
)

val_data = train_gen.flow_from_dataframe(
    train_df,
    x_col='Filepath',
    y_col='Label',
    target_size=(120, 120),
    color_mode='rgb',
    class_mode='binary',
    batch_size=32,
    shuffle=True,
    seed=42,
    subset='validation'
)

test_data = test_gen.flow_from_dataframe(
    test_df,
    x_col='Filepath',
    y_col='Label',
    target_size=(120, 120),
    color_mode='rgb',
    class_mode='binary',
    batch_size=32,
    shuffle=False,
    seed=42
)

# Define the model architecture
inputs = tf.keras.Input(shape=(120, 120, 3))
x = tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu')(inputs)
x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
x = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(x)
x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

# Compile the model
model = tf.keras.Model(inputs=inputs, outputs=outputs)
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Print the model summary
print(model.summary())

# Train the model
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=10,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True
        )
    ]
)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc= 'lower right')
plt.savefig('accuracy_graph.png')

# Save the model
model.save("my_model.h5")