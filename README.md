# Message Classification System

## Project Description
This project is a text-based Message Classification System that automatically categorizes incoming customer messages into one of three classes: **Complaint**, **Feedback**, or **Inquiry**. It accepts a raw text message as input and returns the predicted category along with a confidence score — making it suitable for customer support pipelines, ticketing systems, or chatbot backends.

---

## Approach and Methodology
The system uses a **TF-IDF + Logistic Regression** pipeline built with scikit-learn.

- **TF-IDF Vectorizer** converts raw text into numerical features using unigrams and bigrams, capturing phrases like *"not working"* or *"how do"* that carry strong classification signals
- **Logistic Regression** performs multi-class classification using the `lbfgs` solver with L2 regularization
- Both steps are wrapped in a single `sklearn Pipeline` to prevent data leakage between training and inference
- Evaluated using an 80/20 stratified train-test split and 5-fold cross-validation

---

## Dataset
Manually curated dataset of **60 labeled messages — 20 per class**.

- **Complaint** — e.g. *"Your service has been down for two days and I'm losing money."*
- **Feedback** — e.g. *"The new dashboard is really clean and easy to navigate."*
- **Inquiry** — e.g. *"What are the pricing plans for the enterprise tier?"*
- Classes are perfectly balanced, covering diverse vocabulary and realistic customer language

---

## How to Run

1. Install dependencies → `pip install -r requirements.txt`
2. Train the model → `python classifier.py` *(saves model.pkl)*
3. Predict a single message → `python predict.py "Your message here"`
4. Interactive mode → `python predict.py` then type messages one by one

---

## Sample Inputs and Outputs

| Input Message | Category | Confidence |
|---|---|---|
| *"Your platform has been down for hours!"* | Complaint | 50.7% |
| *"The new UI looks fantastic."* | Feedback | 42.9% |
| *"What is the max users on the basic plan?"* | Inquiry | 37.8% |
| *"Waiting three weeks for a refund, no response."* | Complaint | 53.3% |
| *"Could you add dark mode to the mobile app?"* | Feedback | 40.8% |
| *"Do you support Slack integration?"* | Inquiry | 52.0% |

**Overall accuracy: 83.3%** (5-fold cross-validation)
