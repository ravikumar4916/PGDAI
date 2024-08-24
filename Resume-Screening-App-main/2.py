import streamlit as st
import pickle
import re
import nltk
import matplotlib.pyplot as plt

nltk.download('punkt')
nltk.download('stopwords')

# Loading models
try:
    clf = pickle.load(open('clf.pkl', 'rb'))
    tfidfd = pickle.load(open('tfidf.pkl', 'rb'))
    st.write("Models loaded successfully.")
except Exception as e:
    st.error(f"Error loading models: {e}")

# User and Admin credentials
users = {
    "user": "user123",
    "admin": "admin123"
}

def clean_resume(resume_text):
    clean_text = re.sub('http\S+\s*', ' ', resume_text)
    clean_text = re.sub('RT|cc', ' ', clean_text)
    clean_text = re.sub('#\S+', '', clean_text)
    clean_text = re.sub('@\S+', '  ', clean_text)
    clean_text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', clean_text)
    clean_text = re.sub(r'[^\x00-\x7f]', r' ', clean_text)
    clean_text = re.sub('\s+', ' ', clean_text)
    return clean_text

def login():
    st.sidebar.title("Login")
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")
    if st.sidebar.button("Login"):
        if username in users and users[username] == password:
            st.session_state['logged_in'] = True
            st.session_state['username'] = username
            st.success(f"Logged in as {username}")
        else:
            st.error("Invalid username or password")

def logout():
    st.sidebar.title("Logout")
    if st.sidebar.button("Logout"):
        st.session_state['logged_in'] = False
        st.session_state['username'] = None
        st.success("Logged out successfully")

# Web app
def main():
    st.title("Resume Screening App")

    # Login functionality
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False
        st.session_state['username'] = None

    if not st.session_state['logged_in']:
        login()
    else:
        logout()

        uploaded_files = st.file_uploader('Upload Resumes', type=['txt', 'pdf'], accept_multiple_files=True)
        job_description = st.text_area("Enter Job Description", height=150)

        if not uploaded_files:
            st.warning("Please upload at least one resume file.")
        if not job_description:
            st.warning("Please enter a job description.")

        if uploaded_files and job_description:
            best_resume = None
            best_score = 0
            best_category = None

            st.write(f"Processing {len(uploaded_files)} resumes...")

            for uploaded_file in uploaded_files:
                try:
                    resume_bytes = uploaded_file.read()
                    resume_text = resume_bytes.decode('utf-8')
                except UnicodeDecodeError:
                    resume_text = resume_bytes.decode('latin-1')

                st.write(f"Processing resume: {uploaded_file.name}")

                cleaned_resume = clean_resume(resume_text)
                input_features = tfidfd.transform([cleaned_resume])
                prediction_proba = clf.predict_proba(input_features)[0]
                prediction_id = clf.predict(input_features)[0]

                # Map category ID to category name
                category_mapping = {
                    15: "Java Developer",
                    23: "Testing",
                    8: "DevOps Engineer",
                    20: "Python Developer",
                    24: "Web Designing",
                    12: "HR",
                    13: "Hadoop",
                    3: "Blockchain",
                    10: "ETL Developer",
                    18: "Operations Manager",
                    6: "Data Science",
                    22: "Sales",
                    16: "Mechanical Engineer",
                    1: "Arts",
                    7: "Database",
                    11: "Electrical Engineering",
                    14: "Health and fitness",
                    19: "PMO",
                    4: "Business Analyst",
                    9: "DotNet Developer",
                    2: "Automation Testing",
                    17: "Network Security Engineer",
                    21: "SAP Developer",
                    5: "Civil Engineer",
                    0: "Advocate",
                }

                category_name = category_mapping.get(prediction_id, "Unknown")
                matching_percentage = prediction_proba[prediction_id] * 100

                st.write(f"Resume {uploaded_file.name} matched {category_name} with {matching_percentage:.2f}%")

                # Determine if this resume is the best match
                if matching_percentage > best_score:
                    best_resume = uploaded_file.name
                    best_score = matching_percentage
                    best_category = category_name

            # Display the best matching resume
            if best_resume:
                st.success(f"Best matching resume: {best_resume}")
                st.write(f"Predicted Category: {best_category}")
                st.write(f"Matching Percentage: {best_score:.2f}%")
            else:
                st.error("No resumes matched the job description well.")

# Python main
if __name__ == "__main__":
    main()
