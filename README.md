# resume-parser
A Resume Parser built on LangExtract and Streamlit


That's a great idea\! A clear setup guide is essential for any GitHub project.

Here is the markdown code for a **Setup and Installation** section of your `README.md`. This guide is structured to be clear, professional, and easy for any developer to follow.

### ⚙️ Setup and Installation

Follow these steps to get the **Gemini Resume Parser App** running locally on your machine.

#### 1\. Clone the Repository

First, clone the repository to your local machine:

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

#### 2\. Create and Activate a Virtual Environment

It is highly recommended to use a Python virtual environment to manage dependencies:

```bash
# Create the environment (using venv)
python3 -m venv .venv 

# Activate the environment
source .venv/bin/activate  # On Linux/macOS
# .venv\Scripts\activate   # On Windows (Command Prompt)
```

#### 3\. Install Required Packages

All necessary Python libraries are listed in `requirements.txt`. Install them using pip:

```bash
pip install -r requirements.txt
```

#### 4\. Configure Environment Variables

This application requires access to Google Cloud Services (Gemini API, Cloud Storage). You must set up a `.env` file to securely store your credentials.

**A. Create the `.env` file:**
Copy the example file to create your working environment file:

```bash
cp .env.example .env
```

**B. Edit the `.env` file:**
Open the newly created `.env` file and replace the placeholder values with your actual project details:

```ini
# .env

# Your Google Cloud Project ID
PROJECT_ID="your-gcp-project-id"

# The name of the GCS bucket where resumes are stored
BUCKET_NAME="your-gcs-bucket-name"

# The name of the Gemini model to use for extraction (e.g., gemini-2.5-flash)
MODEL_NAME="gemini-2.5-flash" 

# [CRITICAL] Your Gemini API Key
# Note: You need to enable the Gemini API in your GCP project.
GEMINI_API_KEY="AIzaSy...your-actual-api-key..." 
```

#### 5\. Run the Streamlit Application

With your environment activated and dependencies installed, you can start the application:

```bash
streamlit run your_main_app_file.py
```

The application will automatically open in your web browser, typically at `http://localhost:8501`.

-----

This markdown code covers the full process: cloning, environment setup, dependency installation, secure credential configuration, and launching the app.

Would you like to move on to the first section of your app tutorial?
