Live Demo: https://ai-driven-email-security.streamlit.app/

## About

AI-Driven Email Security is an intelligent system that helps manage emails by detecting spam, phishing URLs, classifying priority, and providing automatic summaries. The focus is on AI-powered email safety and productivity.

## Key Features

- **Spam Detection:** ML classifier using TF-IDF + traditional ML model.
- **Phishing URL Detection:** BERT-based model trained for phishing detection.
- **Priority Classification:** TF-IDF + ML model classifies emails as High, Medium, or Low priority.
- **Email Summarization:** T5-small transformer generates concise email summaries.

## How It Works

- User inputs an email and optional URL.
- ML models process the input:
  - BERT model predicts if the URL is phishing.
  - TF-IDF + ML model predicts if the email is spam.
  - TF-IDF + ML model classifies email priority.
- T5 generates a concise summary of the email.
- Email is stored in the appropriate folder: Inbox, Spam, or Phishing.
