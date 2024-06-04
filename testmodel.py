import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MultiLabelBinarizer

# 1. Load the model and tokenizer
model = tf.keras.models.load_model('casa_model.keras')

# Load the MultiLabelBinarizer (from 'mlb_classes.npy')
mlb = MultiLabelBinarizer()
mlb.classes_ = np.load('mlb_classes.npy', allow_pickle=True)

# 2. Preprocess the new input (use the same preprocessing as during training)
def preprocess_text(text):
    # Apply the same preprocessing steps you used during training
    return text

# 3. Get user input or load new data
new_comments = ["baggage is damaged at airport.", "suitecase is lost", "baggage is lost at airport", "flight delayed due to weather", "overbooked flight","baggage is wet"]  # Example comments
preprocessed_comments = [preprocess_text(comment) for comment in new_comments]

# 4. Make predictions
predictions = model.predict(np.array(preprocessed_comments))

# 5. Convert predictions to labels (adjust threshold if needed)
predicted_labels = mlb.inverse_transform(predictions > 0.3)  # Experiment with thresholds 

# 6. Print or process the predictions
for comment, labels in zip(new_comments, predicted_labels):
    print(f"Comment: '{comment}'")
    print(f"Predicted Labels: {labels}")
