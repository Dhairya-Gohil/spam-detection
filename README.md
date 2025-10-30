**Spam Detection System**

This project is a web application built with Flask that uses a deep learning model to classify messages as 'Spam' or 'Legitimate'. It features a simple user interface where a user can enter a message and receive an instant classification. The application utilizes a trained Bidirectional LSTM (Long Short-Term Memory) model, which is effective at understanding the context and sequence of words to identify spam.

**Features**

The application features a deep learning model (a Bidirectional LSTM) for sequence classification. It has a clean web interface built with Flask and HTML/CSS for easy user interaction and provides real-time predictions. A key feature is the Smart Brand-Aware Correction, which includes post-processing logic to reduce false positives. Messages that might be flagged as spam by the model are re-classified as 'Legitimate' if they contain keywords from trusted brands (e.g., 'Amazon', 'Flipkart', 'Netflix') and a URL. The system also provides a confidence percentage for each prediction.

**Technology Stack**

The backend is built with Flask, using TensorFlow (Keras) for the deep learning component. The frontend is standard HTML and CSS. Data preprocessing relies on a Keras Tokenizer loaded from a pickle file.

**Project Structure**

The project folder contains the main Flask application logic (`app.py`), the trained Keras model (`spam_detection_bidirec_lstm.keras`), the pickled tokenizer (`tokenizer.pkl`), and the training data (`spam.csv`). It also includes a `templates` folder containing `index.html` and a `static` folder for CSS.

**Setup & Installation**

To run this project, you first need to install the dependencies, which are primarily Flask and TensorFlow (`pip install flask tensorflow`). Ensure the trained model (`spam_detection_bidirec_lstm.keras`) and the tokenizer (`tokenizer.pkl`) are in the same directory as `app.py`. You can then run the application using the command `python app.py`. Once running, you can access the application in your web browser at `http://127.0.0.1:5000`.

**Usage**

To use the application, open it in your browser and type or paste a message into the text area. Click the "Check Message" button. The application will then display the result ("Spam" or "Legitimate"), the original message, and the model's confidence in the prediction. If no message is entered, an error will be displayed.

**Training Data**

The `spam.csv`(Downloaded from Kaggle) file is the dataset used to train the model. It contains two columns: `v1` for the label ("ham" or "spam") and `v2` for the raw text of the message.
