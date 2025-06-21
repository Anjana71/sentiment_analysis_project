# 🎭 Emotion Detection from Text Using Classical Machine Learning

This project implements a **text-based emotion classification system** using **classical machine learning models** such as Naive Bayes. It allows both **command-line prediction** and an interactive **Gradio-based web interface** for real-time emotion detection.

---

## 📂 Dataset

The dataset consists of `.txt` files:
- `train.txt`
- `val.txt`
- `test.txt`

Each line is formatted as:
<sentence>;<emotion_label>


Example:
I'm feeling quite sad and sorry for myself but I'll snap out of it soon;sadness



The emotions include: `sadness`, `joy`, `anger`, `fear`, `surprise`, `love`, and others depending on the dataset.

---

## 🧠 Model Overview

- **Preprocessing:** Lowercasing, punctuation removal
- **Vectorization:** TF-IDF with top 3000 features
- **Model:** Multinomial Naive Bayes
- **Evaluation:** Classification report on test split
- **Interface:** Gradio UI + CLI terminal predictions

---
## 🖼️ Demo Screenshot

Here's an example of the Gradio web interface:

![Gradio Output](gradio_demo.png)

## 🛠️ How to Run

### 📍 1. Upload and Extract Dataset

Upload your `archive.zip` containing `train.txt`, `val.txt`, and `test.txt` to Colab and extract it:
```python
from zipfile import ZipFile
zip_path = "/content/archive.zip"
with ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall("/content/emotion_data")
📍 2. Load Dataset and Train Model

# Load .txt files from extracted folder
def load_single_file(path):
    ...

train_df = load_single_file("/content/emotion_data/train.txt")
...
df = pd.concat([train_df, val_df, test_df])
📍 3. Train and Evaluate

# Preprocessing, TF-IDF, train/test split, train Naive Bayes
...
print(classification_report(y_test, model.predict(X_test)))
🧪 CLI Prediction (Colab Input Loop)

while True:
    user_input = input("Type sentence (or 'exit'): ")
    if user_input.lower() == 'exit':
        break
    ...
🌐 Web Interface (Gradio)

import gradio as gr

def predict_emotion(text):
    ...

with gr.Blocks() as demo:
    ...
demo.launch(share=True)
📊 Results
Model performs well with balanced classes and clean input. Evaluation metrics include precision, recall, and F1-score for each emotion class.

📌 Project Structure

emotion-detector/
│
├── archive.zip                # Contains train.txt, val.txt, test.txt
├── emotion_model.ipynb        # Main notebook (Colab-compatible)
├── README.md                  # Project documentation
🏁 Features
🧹 Text preprocessing (cleaning, vectorization)

🧠 ML-based classification (Naive Bayes)

🧪 CLI emotion prediction loop

🌐 Gradio interface for real-time predictions

📈 Model evaluation and performance metrics


📚 Requirements

pip install pandas scikit-learn gradio matplotlib
🚀 Future Work
Use deep learning models (LSTM, BERT)

Support more emotion categories

Export model with joblib or pickle

Add confusion matrix and visualizations

🙋‍♀️ Author
Anjana C
Student Project – Emotion Classification with Classical ML

📝 License
This project is open-source and free to use for academic and educational purposes.

