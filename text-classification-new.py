import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

# 1. Load and preprocess data
df = pd.read_excel('casa_data.xlsx')
df['Comments'] = df['Comments'].astype(str).fillna('')  # Handle missing values

# Split labels into separate columns (binary encoding)
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(df['Code'].str.split(','))  # Assuming ',' as separator

# Text preprocessing (adjust as needed)
vectorizer = tf.keras.layers.TextVectorization(max_tokens=10000, output_sequence_length=100)
vectorizer.adapt(df['Comments'])

# Split into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(df['Comments'], y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# 2. Build the model
model = tf.keras.Sequential([
    vectorizer,
    tf.keras.layers.Embedding(10000, 128),  # Adjust embedding dimensions
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),  # Bi-LSTM for better context
    tf.keras.layers.Dense(64, activation='relu'),  # Additional dense layer
    tf.keras.layers.Dropout(0.5),  # Dropout for regularization
    tf.keras.layers.Dense(y.shape[1], activation='sigmoid')  
])

model.compile(loss='binary_crossentropy', 
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), # Adjust learning rate
              metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

# 3. Train the model
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val), batch_size=32)

# 4. Evaluate the model
loss, accuracy, precision, recall = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")
print(f"Test Precision: {precision:.4f}")
print(f"Test Recall: {recall:.4f}")

# 5. Make predictions (example)
example_comment = ["This is a sample user comment."]
predictions = model.predict(example_comment)
predicted_labels = mlb.inverse_transform(predictions > 0.5)  # Adjust threshold
print(predicted_labels)
