from flask import Flask, request, render_template
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import pickle
import tensorflow as tf

# 🔹 Initialize Flask app
app = Flask(__name__)

# --------------------------
# 🔹 Load model & tokenizer
# --------------------------
model = tf.keras.models.load_model("spam_detection_bidirec_lstm.keras")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

max_len = 100  # replace with the max_len used during training

# --------------------------
# 🔹 Homepage route
# --------------------------
@app.route('/')
def home():
    return render_template('index.html')

# --------------------------
# 🔹 Prediction route
# --------------------------
@app.route('/predict', methods=['POST'])
def predict():
    # Get message only from textarea
    message = request.form.get('message', '').strip()

    if not message:
        return render_template('index.html', prediction=None, confidence=None, message='', error="Please enter a message.")

    lower_msg = message.lower()

    # Convert text → padded sequence
    seq = tokenizer.texts_to_sequences([message])
    pad_seq = pad_sequences(seq, maxlen=max_len, padding='post')

    # Model prediction
    prob = float(model.predict(pad_seq)[0][0])
    label = "Spam" if prob > 0.5 else "Legitimate"

    # Base confidence normalization
    confidence = round(prob * 100, 2) if label == "Spam" else round((1 - prob) * 100, 2)

    # Smart brand-aware correction
    spam_keywords = [
        "win", "offer", "urgent", "reward", "lottery", "guaranteed",
        "bonus", "voucher", "cashback", "discount", "click", "claim", "gift"
    ]
    trusted_brands = [
        "amazon", "flipkart", "tataneu", "jio", "spotify",
        "zomato", "swiggy", "netflix", "myntra", "airtel"
    ]
    link_pattern = r"https?://[^\s]+"

    if prob > 0.95 and any(brand in lower_msg for brand in trusted_brands) and re.search(link_pattern, lower_msg):
        label = "Legitimate"
        confidence = round((1 - prob) * 100, 2)
    elif 0.4 <= prob <= 0.7:
        if any(brand in lower_msg for brand in trusted_brands):
            label = "Legitimate"
            confidence = round((1 - prob) * 100, 2)
        elif any(word in lower_msg for word in spam_keywords):
            label = "Spam"
            confidence = round(prob * 100, 2)
        else:
            confidence = 50.0

    return render_template('index.html', prediction=label, confidence=confidence, message=message)

# --------------------------
# 🔹 Run the app
# --------------------------
if __name__ == '__main__':
    app.run(debug=True)
