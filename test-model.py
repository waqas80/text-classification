import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# from tensorflow.keras.models import load_model
# optimizer = keras.optimizers.Adam()

# Load the model
model = tf.keras.models.load_model('baggage_model.keras',compile=False)

from tensorflow.keras import backend
backend.clear_session()

# Assuming you have a new comment stored in a variable named `new_comment`
# Tokenize and pad the new comment
# Define and load the tokenizer
new_comment = "I lost baggage"
data = [new_comment]
texts = data[0]
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
X = tokenizer.texts_to_sequences(texts)

new_comment_sequence = tokenizer.texts_to_sequences([new_comment])
new_comment_padded = pad_sequences(new_comment_sequence, maxlen=100)


# Make predictions
predicted_prob = model.predict(new_comment_padded)


# Convert predicted probabilities to class labels
predicted_class = "damage" if predicted_prob > 0.5 else "lost"

print(f"The predicted class for the new comment is: {predicted_class}")
