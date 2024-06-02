import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.utils import class_weight
from tensorflow.keras import layers
from tensorflow.keras.optimizers.legacy import Adam # Legacy Adam optimizer
from tensorflow.keras.callbacks import ReduceLROnPlateau

# 1. Load and preprocess data
df = pd.read_excel('casa_data.xlsx')
df['Comments'] = df['Comments'].astype(str).fillna('')  # Handle missing values

# Split labels into separate columns (binary encoding)
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(df['Code'].values.reshape(-1, 1))  # Reshape to 2D array for mlb

# Text preprocessing
max_tokens = 10000
max_length = 100
vectorizer = tf.keras.layers.TextVectorization(max_tokens=max_tokens, output_sequence_length=max_length)
vectorizer.adapt(df['Comments'].values)

# Split into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(df['Comments'], y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)


'''
model = tf.keras.Sequential([
    vectorizer,
    tf.keras.layers.Embedding(10000, 128, 
                             embeddings_regularizer=tf.keras.regularizers.l2(0.01)), # L2 regularization
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dense(64, activation='relu', 
                         kernel_regularizer=tf.keras.regularizers.l2(0.01)),  # L2 regularization
    tf.keras.layers.Dropout(0.6),  # Increased dropout rate 
    tf.keras.layers.Dense(y.shape[1], activation='sigmoid') 
])
'''
model = tf.keras.Sequential([
    vectorizer,
    layers.Embedding(max_tokens + 1, 128), 
    layers.Conv1D(128, 5, activation='relu'),
    layers.MaxPooling1D(pool_size=2),    
    layers.Bidirectional(layers.LSTM(64)),
    layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    layers.Dropout(0.5),
    layers.Dense(y.shape[1], activation='sigmoid')
])

# Compile with legacy Adam optimizer
model.compile(loss='binary_crossentropy',
              optimizer=Adam(learning_rate=0.0001), # Using legacy Adam
              metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])



# 3. Early stopping callback 
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10, 
    restore_best_weights=True
)

lr_reducer = ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.1, 
    patience=5,
)

# Class weights (if you have imbalanced classes)
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train.flatten()
)
class_weights = dict(enumerate(class_weights))


# 4. Train the model (with callbacks)
history = model.fit(
    np.array(X_train), y_train,  # Convert to NumPy array
    epochs=25,  # Consider increasing epochs further if needed
    validation_data=(np.array(X_val), y_val),
    batch_size=32,  # Consider experimenting with different batch sizes (e.g., 64, 128)
    callbacks=[early_stopping, lr_reducer]
)



# 5. Evaluation of the model
loss, accuracy, precision, recall = model.evaluate(np.array(X_test), y_test)  # Convert to NumPy array
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")
print(f"Test Precision: {precision:.4f}")
print(f"Test Recall: {recall:.4f}")

# 6. Save the model
model.save('casa_model.keras')  # Save the model in HDF5 format

# 5. Make predictions (example)
example_comment = ["baggage is damaged at airport."]
predictions = model.predict([example_comment]) # Ensure a list of lists
predicted_labels = mlb.inverse_transform(predictions > 0.35)  
print(predicted_labels)


'''
Model was over fit
one epoch takes 7.2 minutes
'''
