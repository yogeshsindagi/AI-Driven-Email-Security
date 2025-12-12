import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize
from nltk.stem.porter import PorterStemmer
from transformers import pipeline
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import os


# Initialize the PorterStemmer
ps = PorterStemmer()

# Download required NLTK data
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

# Define text transformation function
def transform_text(text):
    text = text.lower()
    text = wordpunct_tokenize(text)
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words("english") and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

MODEL_DIR = "models"

# Load Priority Model and Vectorizer
tfidf = pickle.load(open(os.path.join(MODEL_DIR, "priority_vectorizer.pkl"), "rb"))
model = pickle.load(open(os.path.join(MODEL_DIR, "priority_model.pkl"), "rb"))

# Load Spam Model and Vectorizer
spam_tfidf = pickle.load(open(os.path.join(MODEL_DIR, "spam_vectorizer.pkl"), "rb"))
spam_model = pickle.load(open(os.path.join(MODEL_DIR, "spam_model.pkl"), "rb"))

# Load Phishing URL Detection Model
PHISHING_MODEL_DIR = "phishing"

phishing_model = BertForSequenceClassification.from_pretrained(
    PHISHING_MODEL_DIR, use_safetensors=True
)
phishing_tokenizer = BertTokenizer.from_pretrained(PHISHING_MODEL_DIR)


# Helper functions
def classify_email(input_sms):
    if input_sms.strip() == "":
        return "❗ Please enter a valid email message before classification."

    transformed_sms = transform_text(input_sms)
    vector_input = tfidf.transform([transformed_sms])
    result = model.predict(vector_input)[0]

    # Classification based on priority
    if result == 0:
        return "🔴 High Priority"
    elif result == 1:
        return "🟢 Low Priority"
    else:
        return "🟠 Medium Priority"


def detect_spam(input_sms):
    transformed_sms = transform_text(input_sms)
    vector_input = spam_tfidf.transform([transformed_sms])
    result = spam_model.predict(vector_input)[0]
    return result == 1  # Spam = 1, Not Spam = 0


def detect_phishing_url(url):
    if not url.strip():
        return "❗ Please enter a valid URL."
    try:
        # Ensure model is on CPU
        phishing_model.to("cpu")

        # Tokenize and move inputs to CPU
        inputs = phishing_tokenizer(url, return_tensors="pt", truncation=True, padding=True, max_length=128)
        inputs = {k: v.to("cpu") for k, v in inputs.items()}

        # Forward pass
        outputs = phishing_model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=-1)
        result = "🚩 Phishing URL" if prediction.item() == 1 else "✅ Legitimate URL"

        # Debug print
        print(f"[DEBUG] URL: {url} | Prediction: {result}")
        return result

    except Exception as e:
        print(f"[DEBUG] Phishing error: {e}")
        return f"❗ Error occurred: {e}"


# Initialize summarizer model
summarizer = pipeline("summarization", model="t5-small", tokenizer="t5-small", framework="pt", device=-1)

# Email Summarization Function
def summarize_email(input_sms):
    if input_sms.strip() == "":
        return "❗ Please enter a valid email message before generating the summary."

    try:
        summary = summarizer(input_sms, min_length=40, do_sample=False)
        return summary[0]["summary_text"]
    except Exception as e:
        return f"❗ Error generating summary: {e}"


# Streamlit App Configuration
st.set_page_config(page_title="AI-Driven Email Security and Management", layout="wide")

# Initialize session state for emails
if "emails" not in st.session_state:
    st.session_state.emails = []


# Helper function to filter emails by folder
def filter_emails_by_folder(folder):
    return [email for email in st.session_state.emails if email["folder"] == folder]


# Helper function to filter emails by priority
def filter_emails_by_priority(priority):
    return [email for email in st.session_state.emails if email.get("priority") == priority]


# Render header
def render_header():
    st.markdown("<h1 style='text-align: center;'>AI-Driven Email Security</h1>", unsafe_allow_html=True)


# Sidebar Buttons for Folder Selection
st.sidebar.title("Navigation")
folder_buttons = ["📥 Inbox", "Spam & Phishing", "📝 Input Mail", "ℹ️ About Us"]

if "selected_tab" not in st.session_state:
    st.session_state.selected_tab = "📥 Inbox"  # Default tab is Inbox

# Create buttons for navigation
for folder in folder_buttons:
    if st.sidebar.button(folder):
        st.session_state.selected_tab = folder

# Render header on each page
render_header()

# Inbox Tab
if st.session_state.selected_tab == "📥 Inbox":
    st.title("📥 Inbox")
    tab1, tab2, tab3 = st.tabs(["🔴 High Priority", "🟠 Medium Priority", "🟢 Low Priority"])

    for priority, tab, label in zip(["High", "Medium", "Low"], [tab1, tab2, tab3], ["High", "Medium", "Low"]):
        with tab:
            emails = filter_emails_by_priority(priority)
            if emails:
                # Create a dropdown (selectbox) for email selection
                email_list = [f"{email['subject']} ({email['sender']})" for email in emails]
                selected_email = st.selectbox(f"Select an email ({label} Priority):", email_list,
                                              key=f"select_{label.lower()}")

                # Display the selected email details
                if selected_email:
                    email_index = email_list.index(selected_email)
                    email = emails[email_index]
                    st.subheader(email["subject"])
                    st.write(f"**From:** {email['sender']}")
                    st.write(f"**Content:** {email['content']}")
                    st.write(f"**URL:** {email.get('url', 'N/A')}")
                    # Add summarize button
                    if st.button(f"📝 Summarize Email: {email['subject']}", key=f"summarize_{email['subject']}"):
                        summary = summarize_email(email["content"])
                        st.success(f"**Summary:** {summary}")
            else:
                st.info(f"No emails in {label} Priority.")

# Spam & Phishing Dropdown Tab
elif st.session_state.selected_tab == "Spam & Phishing":
    tab1, tab2 = st.tabs(["📋 Spam Folder", "🚩 Phishing Folder"])

    with tab1:
        st.title("📋 Spam Folder")
        emails = filter_emails_by_folder("Spam")
        if emails:
            email_list = [f"{email['subject']} ({email['sender']})" for email in emails]
            selected_email = st.selectbox("Select an email:", email_list, key="spam_email")

            if selected_email:
                email_index = email_list.index(selected_email)
                email = emails[email_index]
                st.subheader(email["subject"])
                st.write(f"**From:** {email['sender']}")
                st.write(f"**Content:** {email['content']}")
                st.write(f"**URL:** {email.get('url', 'N/A')}")
        else:
            st.info("No emails in Spam folder.")

    with tab2:
        st.title("🚩 Phishing Folder")
        emails = filter_emails_by_folder("Phishing")
        if emails:
            email_list = [f"{email['subject']} ({email['sender']})" for email in emails]
            selected_email = st.selectbox("Select an email:", email_list, key="phishing_email")

            if selected_email:
                email_index = email_list.index(selected_email)
                email = emails[email_index]
                st.subheader(email["subject"])
                st.write(f"**From:** {email['sender']}")
                st.write(f"**Content:** {email['content']}")
                st.write(f"**URL:** {email.get('url', 'N/A')}")
        else:
            st.info("No emails in Phishing folder.")

# Input Mail Tab
elif st.session_state.selected_tab == "📝 Input Mail":
    st.title("📝 Input Mail")
    with st.form("Email Input Form"):
        sender = st.text_input("Sender's Email ID", placeholder="example@example.com")
        subject = st.text_input("Subject", placeholder="Enter the email subject")
        content = st.text_area("Email Content", placeholder="Type the email content here...")
        url = st.text_input("Related URL (if any)", placeholder="Paste the URL here...")
        submitted = st.form_submit_button("Submit")

        if submitted:
            if sender and subject and content:
                if sender and subject and content:
                    if detect_phishing_url(url) == "🚩 Phishing URL":
                        folder = "Phishing"
                        priority = "N/A"
                    elif detect_spam(content):
                        folder = "Spam"
                        priority = "N/A"
                    else:
                        priority = classify_email(content)
                        if priority == "🔴 High Priority":
                            folder = "High"
                        elif priority == "🟠 Medium Priority":
                            folder = "Medium"
                        else:
                            folder = "Low"

                # Append email details to session state
                st.session_state.emails.append({
                    "folder": folder,
                    "priority": folder,
                    "sender": sender,
                    "subject": subject,
                    "content": content,
                    "url": url,
                })
                st.success(f"Email successfully added to {folder} Priority Folder!")
            else:
                st.error("Please fill in all fields.")

# About Us Tab
elif st.session_state.selected_tab == "ℹ️ About Us":
    st.title("ℹ️ About Us")
    st.markdown(""" 
        **AI-Driven Email** helps you:
        - Detect Spam and Phishing emails.
        - Summarize email content.
        - Categorize emails dynamically into Inbox, Spam, and Phishing folders.
        
        Developers
        **Team Members**:
        - Yogesh (Lead AI Developer)
        - Shreyas G (AI Specialist)
        - Aditya (ML and Frontend Designer)
    """)
