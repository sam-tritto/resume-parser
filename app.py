
import pandas as pd
import pandas_gbq
import json
import time
import os
import requests
import datetime
import pypdf
import textwrap
import io 
from typing import List
from pathlib import Path
import base64

from google import genai
from google.cloud import storage
from google.oauth2 import service_account 

import langextract as lx
import fitz  # PyMuPDF

import streamlit as st
from streamlit import session_state as ss # Alias for brevity


try:
    from dotenv import load_dotenv
    # Find the absolute path to the .env file 
    # This ensures it loads correctly regardless of the starting directory.
    dotenv_path = Path(__file__).parent / ".env"
    load_dotenv(dotenv_path=dotenv_path)
    
except ImportError:
    # This is fine if the user doesn't need to load a local .env file.
    pass


# ----------------------------------------------------------
# ENVIRONMENT VARIABLES (Read from OS environment)
# ----------------------------------------------------------

LANGEXTRACT_API_KEY = os.environ.get("LANGEXTRACT_API_KEY")
GOOGLE_APPLICATION_CREDENTIALS = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
BUCKET_NAME = os.environ.get("BUCKET_NAME")
PROJECT_ID = os.environ.get("PROJECT_ID")
TABLE_NAME = os.environ.get("TABLE_NAME")
MODEL_NAME = "gemini-2.5-flash"
PRICE_PER_MILLION_INPUT_TOKENS = 1.25
EXPIRATION = 30 # Signed URL expiration in minutes
# Define the target color for white text in PDF RGB (24-bit integer)
# 0xFFFFFF in hex is 16777215 in decimal. This represents pure white foreground text.
WHITE_COLOR_INT = 16777215

# ----------------------------------------------------------
# CLIENT INITIALIZATION 
# ----------------------------------------------------------
client = None
if LANGEXTRACT_API_KEY:
    try:
        client = genai.Client(api_key=LANGEXTRACT_API_KEY)
    except Exception as e:
        st.error(f"❌ Failed to initialize Gemini Client: {e}")

# ----------------------------------------------------------
# UTILITY FUNCTIONS (GCS, PDF Extraction, etc.)
# ----------------------------------------------------------

@st.cache_data(ttl=300) # Cache the list for 5 minutes
def list_gcs_pdfs(bucket_name: str) -> List[str]:
    """Lists all PDF files in the specified GCS bucket."""
    if not BUCKET_NAME:
        return []

    try:
        # Initialize client with PROJECT_ID for context
        storage_client = storage.Client(project=PROJECT_ID)
        blobs = storage_client.list_blobs(bucket_name)
        # Filter for PDF files and return only the file name (blob.name)
        pdf_files = [blob.name for blob in blobs if blob.name.lower().endswith('.pdf')]
        return sorted(pdf_files)
    except Exception as e:
        st.error(f"❌ Error listing files from GCS bucket '{BUCKET_NAME}': {e}")
        return []

# ----------------------------------------------------------

def generate_gcs_signed_url(bucket_name: str, blob_name: str, expiration_minutes: int = 15, credentials_path: str = None) -> str:
    """Generates a V4 signed URL for a Google Cloud Storage object."""
    try:
        if credentials_path:
            credentials = service_account.Credentials.from_service_account_file(credentials_path)
            storage_client = storage.Client(credentials=credentials)
        else:
            storage_client = storage.Client()

        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        expiration_time = datetime.timedelta(minutes=expiration_minutes)

        signed_url = blob.generate_signed_url(
            version="v4",
            expiration=expiration_time,
            method="GET",
        )
        return signed_url

    except Exception as e:
        # Crucial error display for debugging
        error_message = f"❌ V4 Signing Failed: Check Service Account permissions and GOOGLE_APPLICATION_CREDENTIALS path. Error details: {e}"
        print(error_message)
        st.error(error_message)
        return ""

# ----------------------------------------------------------

@st.cache_data(show_spinner=False)
def get_pdf_bytes_from_gcs(bucket_name: str, blob_name: str) -> bytes | None:
    """Fetches the content (bytes) of a PDF file from GCS."""
    try:
        storage_client = storage.Client(project=PROJECT_ID)
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        pdf_bytes = blob.download_as_bytes()
        return pdf_bytes
    except Exception as e:
        st.error(f"❌ Error fetching '{blob_name}' from GCS: {e}")
        return None

# ----------------------------------------------------------

def display_pdf_bytes(pdf_bytes: bytes, display_height: int = 700):
    """
    Renders PDF bytes directly in Streamlit using a Base64 encoded iframe.
    This avoids saving the file locally.
    """
    try:
        # Encode the PDF bytes to Base64
        base64_pdf = base64.b64encode(pdf_bytes).decode('utf-8')

        # Create the HTML iframe code
        pdf_display = f"""
        <iframe 
            src="data:application/pdf;base64,{base64_pdf}" 
            width="100%" 
            height="{display_height}px" 
            type="application/pdf"
        ></iframe>
        """
        # Display the iframe in Streamlit
        st.markdown(pdf_display, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"❌ Could not render PDF for display. Error: {e}")
# ----------------------------------------------------------

def extract_text_from_url(pdf_url: str) -> str:
    """Reads a PDF file from a given URL and extracts all text content."""
    pdf_bytes = None
    try:
        response = requests.get(pdf_url)
        response.raise_for_status()
        pdf_bytes = response.content
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Error fetching PDF from URL: {e}")
        return ""
    
    if not pdf_bytes:
        return ""
    
    try:
        pdf_stream = io.BytesIO(pdf_bytes)
        reader = pypdf.PdfReader(pdf_stream)
        full_text = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                full_text.append(text)
        return "\n\n".join(full_text)
    except Exception as e:
        st.error(f"❌ An error occurred during PDF processing (pypdf error): {e}")
        return ""

# ----------------------------------------------------------

def count_tokens_and_estimate_cost(clean_resume_text: str):
    """Counts the tokens in the cleaned resume text and estimates the cost.""" 
    if not client:
        st.error("Cannot count tokens: Gemini Client is not initialized.")
        return

    try:
        token_count = client.models.count_tokens(model=MODEL_NAME, contents=clean_resume_text)
        total_tokens = token_count.total_tokens
        estimated_cost = (total_tokens / 1_000_000) * PRICE_PER_MILLION_INPUT_TOKENS
        
        st.info(f"Estimated Tokens: **{total_tokens:,}**. Estimated Cost (Gemini-2.5-flash-input): **${estimated_cost:.6f}**")

    except Exception as e:
        st.error(f"An error occurred while counting tokens. Error: {e}")

# ----------------------------------------------------------

def detect_white_text(pdf_file_path_or_bytes: str | bytes) -> dict:
    """
    Scans a PDF document for text spans where the foreground color is white
    (0xFFFFFF), indicating potentially hidden text against a white background.

    Args:
        pdf_file_path_or_bytes: The file path (str) or file bytes (bytes)
                                of the PDF resume.

    Returns:
        A dictionary containing a list of findings and a summary flag.
    """
    findings = []
    
    # Handle both file paths (for local testing) and bytes (for Streamlit upload)
    try:
        if isinstance(pdf_file_path_or_bytes, bytes):
            # Open from memory (e.g., Streamlit uploaded file)
            doc = fitz.open(stream=pdf_file_path_or_bytes, filetype="pdf")
        else:
            # Open from file path
            doc = fitz.open(pdf_file_path_or_bytes)
    except Exception as e:
        print(f"Error opening PDF: {e}")
        return {"flagged": False, "findings": [], "error": str(e)}

    for page_num, page in enumerate(doc):
        # Use 'dict' output for maximum detail, down to the span level
        text_data = page.get_text('dict')
        
        for block in text_data.get('blocks', []):
            if 'lines' in block:
                for line in block['lines']:
                    for span in line['spans']:
                        # PyMuPDF stores color as a 24-bit integer
                        span_color = span.get('color')
                        
                        # Check if the text color is white
                        if span_color == WHITE_COLOR_INT:
                            # Extract the text and its bounding box for reporting
                            hidden_text = span['text'].strip()
                            bbox = span['bbox'] # (x0, y0, x1, y1)
                            
                            if hidden_text: # Only log non-empty strings
                                findings.append({
                                    "page": page_num + 1,
                                    "text": hidden_text,
                                    "bbox": bbox,
                                    "reason": "Foreground text color is white (0xFFFFFF)."
                                })

    doc.close()

    return {
        "flagged": len(findings) > 0,
        "findings": findings,
        "total_findings": len(findings)
    }


# ----------------------------------------------------------

def process_lx_output_to_dataframe(lx_result: dict, file_name: str, white_text_results: dict) -> pd.DataFrame:
    # ... (function body remains the same) ...
    """
    Takes the structured JSON output from a LangExtract (lx.extract) call 
    and converts it into a clean pandas DataFrame.

    This function handles both simple fields (like name, email, skills) and 
    nested fields (like work_experience with attributes).
    
    Args:
        lx_result: The dictionary output representing the full document extraction.

    Returns:
        A pandas DataFrame where each row is a single extracted entity.
    """
    
    records = []
    
    # ---------------------------------------------------
    # Process the LangExtract data
    # ---------------------------------------------------

    # Iterate through all extracted entities
    for extraction in lx_result.get("extractions", []):
        class_name = extraction.get("extraction_class")
        text_value = extraction.get("extraction_text")
        attributes = extraction.get("attributes", {})
        
        # Start the record with basic details
        record = {
            "entity_type": class_name,
            "raw_text_value": text_value,
        }
        
        # If there are nested attributes, add them to the record.
        # This is where we flatten the nested structure.
        if attributes:
            # For complex entities, the desired value is usually in the attributes
            # rather than the raw text. We merge the attributes directly.
            record.update(attributes)
        else:
            # For simple entities (Name, Email, Skills), the value is the raw_text_value itself.
            record["value"] = text_value

        records.append(record)

    # ---------------------------------------------------
    # Process the hidden white text findings
    # ---------------------------------------------------

    try:
        parsed_json = white_text_results
        findings = parsed_json.get("findings", [])
        for finding in findings:
            # Create a standardized record for each finding
            record = {
                # New entity type for the final structured finding
                "entity_type": "hidden_text_finding",
                "raw_text_value": finding.get("text", "N/A"),
                "hidden_page": finding.get("page", "N/A"),
                "hidden_reason": finding.get("reason", "N/A"),
                # Convert list bbox to string for easier display in DataFrame
                "hidden_bbox": str(finding.get("bbox", "N/A"))
            }
            records.append(record)
        if len(findings) > 0:
            st.warning(f"⚠️ Warning there are {len(findings)} structured hidden text findings.")
    except json.JSONDecodeError as e:
        # Handle case where LLM or mock injected malformed JSON
        st.error(f"❌ Error parsing Raw Hidden Text JSON: {e}")
        records.append({
            "entity_type": "raw_hidden_text_json_error", 
            "raw_text_value": white_text_results, 
            "error": str(e)
        })


    # Convert the list of dictionaries into a DataFrame
    df = pd.DataFrame(records)
    
    # Optional: Clean up and reorder columns for a clearer view
    # This ensures columns like 'role', 'company', etc., show up nicely.
    df = df.sort_values(by='entity_type').reset_index(drop=True)

    df['file_name'] = file_name 
    
    return df
# ----------------------------------------------------------

def load_lx_results_from_jsonl(filepath: str) -> list[dict]:
    # ... (function body remains the same) ...
    """Loads one or more LangExtract document results from a JSON Lines (.jsonl) file."""
    results = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip(): results.append(json.loads(line))
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading JSONL: {e}")
        
    return results

# ----------------------------------------------------------


def sanitize_text(text: str) -> str:
    # ... (function body remains the same) ...
    """
    Performs aggressive cleaning to remove characters that break JSON 
    serialization or are non-printable, often originating from PDF extraction.
    """
    if not isinstance(text, str):
        return ""
        
    # 1. Strip non-ASCII/control characters and convert to standard ASCII.
    # This is often necessary for data extracted from PDFs.
    sanitized = text.encode('ascii', 'ignore').decode('ascii')

    # 2. Handle newlines and null characters 
    sanitized = sanitized.replace('\x00', ' ').replace('\u0000', ' ') 
    sanitized = sanitized.replace('\r\n', '\n').replace('\r', '\n')

    # 3. CRITICAL: Replace double quotes (") and backslashes (\) with spaces.
    # An unescaped quote is the most common cause of "Unterminated string" in JSON.
    sanitized = sanitized.replace('"', ' ').replace('\\', ' ')
    
    # 4. Collapse multiple spaces and trim
    sanitized = ' '.join(sanitized.split())

    return sanitized

# ----------------------------------------------------------

def save_to_bigquery(df):
    """Saving to BigQuery."""
    df.to_gbq(destination_table=TABLE_NAME, if_exists='append')
    st.success(f"Successfully saved {len(df)} records to BigQuery table **{TABLE_NAME}**.")


# ----------------------------------------------------------

def process_file_from_gcs(file_name: str):
    """Handles the processing pipeline for a file selected from GCS."""
    
    # Reset all processing state variables
    ss.resume_text = None
    ss.lx_result = None
    ss.white_text_results = None
    
    ss.uploaded_file_name = file_name # Set name for display/downstream

    # 1. Get Signed URL
    try:
        with st.spinner(f"Generating signed URL for **{file_name}**..."):
            signed_url = generate_gcs_signed_url(BUCKET_NAME, file_name, EXPIRATION, GOOGLE_APPLICATION_CREDENTIALS)
            if not signed_url:
                st.error("❌ Signed URL generation failed. See error above.")
                return

            ss.signed_url = signed_url
    except Exception as e:
        st.error(f"❌ **Signed URL Pipeline Failed**: {e}")
        return

    # 2. Extract Text from URL
    try:
        with st.spinner("Extracting text from PDF via signed URL..."):
            resume_text = extract_text_from_url(signed_url)
            
            if not resume_text:
                st.error("❌ **PDF Text Extraction Failed**. The PDF might be empty or unreadable.")
                return

            ss.resume_text = resume_text
            st.success("Text extraction complete.")

    except Exception as e:
        st.error(f"❌ **Text Extraction Pipeline Failed**: An unexpected error occurred. Error: {e}")
        return

    # 3. White Text Detection (Requires PDF bytes)
    try:
        with st.spinner("Running white text detection..."):
            pdf_bytes = get_pdf_bytes_from_gcs(BUCKET_NAME, file_name)
            
            if pdf_bytes:
                ss.pdf_bytes = pdf_bytes
                white_text_results = detect_white_text(pdf_bytes)
                ss.white_text_results = white_text_results
            else:
                st.warning("⚠️ Could not retrieve PDF bytes for white text check.")
                ss.white_text_results = {"flagged": False, "findings": [], "total_findings": 0}

    except Exception as e:
        st.error(f"❌ **White Text Detection Failed**: An unexpected error occurred. Error: {e}")


# ----------------------------------------------------------

def handle_file_upload(uploaded_file):
    """Handles GCS upload, sets session state, and triggers reprocessing of the new file."""
    file_name = uploaded_file.name
    
    # 1. GCS Upload
    try:
        with st.spinner(f"Uploading {file_name} to GCS..."):
            storage_client = storage.Client(project=PROJECT_ID)
            bucket = storage_client.bucket(BUCKET_NAME)
            blob = bucket.blob(file_name)
            # Ensure pointer is at the start and upload
            uploaded_file.seek(0) 
            blob.upload_from_file(uploaded_file)
            st.success(f"Uploaded **{file_name}** to GCS bucket **{BUCKET_NAME}**.")

            # Invalidate the cache for the file list to reflect the new upload
            list_gcs_pdfs.clear() 

            # Trigger immediate processing of the newly uploaded file
            process_file_from_gcs(file_name)
            
            # Update the dropdown selection to the new file
            ss.selected_gcs_file = file_name
            
            # Rerun to update the dropdown list and re-render the app with the new selection
            st.rerun() 

    except Exception as e:
        st.error(f"❌ **GCS Upload Failed**: Check your bucket name/permissions. Error: {e}")
        # Reset file-specific state if upload failed
        ss.uploaded_file_name = None
        return


# ----------------------------------------------------------
# PROMPT AND EXAMPLES
# ----------------------------------------------------------

prompt = textwrap.dedent("""
    Extract the following entities from the text:
    - Candidate Name
    - Email Address
    - Phone Number
    - Skills (list)
    - Education (nested structure)
    - Work Experience (nested structure)
    - Certifications (list)
    - Projects (list)
    - Languages Known (list)
    - Awards and Honors (list)
    
    Provide the results in a structured JSON format. For Work Experience and Education, 
    extract each instance (e.g., a single job or degree) as a separate entity 
    using the 'attributes' field to capture details like dates, titles, and descriptions.
""").strip()  

examples: List[lx.data.ExampleData] = [
    lx.data.ExampleData(
        text="""
        Sam Tritto | sam.tritto@example.com | 555-123-4567

        Experience
        Data Scientist at The Home Depot (2019-2023). Led ML model deployment for inventory optimization.
        Research Associate at University Y (2018-2019). Published 2 papers on NLP model efficiency.

        Education
        M.S. in Data Science, University X, Graduated 2019. Thesis on predictive analysis.

        Skills: Python, TensorFlow, SQL, PyTorch.
        Certifications: AWS Certified Data Analytics, Google Cloud Professional Data Engineer.
        Projects: Recommendation Engine (GitHub link).
        Languages: English (Fluent), Spanish (Conversational).
        Awards: Employee of the Year 2021 (THD).
        """,
        extractions=[
            # --- Simple Contact & Identifier Entities ---
            lx.data.Extraction(extraction_class="candidate_name", extraction_text="Sam Tritto"),
            lx.data.Extraction(extraction_class="email_address", extraction_text="sam.tritto@example.com"),
            lx.data.Extraction(extraction_class="phone_number", extraction_text="555-123-4567"),

            # --- List Entities (Repeat extraction_class for each item) ---
            lx.data.Extraction(extraction_class="skill", extraction_text="Python"),
            lx.data.Extraction(extraction_class="skill", extraction_text="TensorFlow"),
            lx.data.Extraction(extraction_class="skill", extraction_text="SQL"),
            lx.data.Extraction(extraction_class="skill", extraction_text="PyTorch"),
            
            lx.data.Extraction(extraction_class="language", extraction_text="English (Fluent)"),
            lx.data.Extraction(extraction_class="language", extraction_text="Spanish (Conversational)"),

            lx.data.Extraction(extraction_class="certification", extraction_text="AWS Certified Data Analytics"),
            lx.data.Extraction(extraction_class="certification", extraction_text="Google Cloud Professional Data Engineer"),

            lx.data.Extraction(extraction_class="project", extraction_text="Recommendation Engine (GitHub link)"),

            lx.data.Extraction(extraction_class="award", extraction_text="Employee of the Year 2021 (THD)"),

            # --- Complex/Nested Entity: Work Experience (Instance 1) ---
            lx.data.Extraction(
                extraction_class="work_experience",
                extraction_text="Data Scientist at The Home Depot (2019-2023). Led ML model deployment for inventory optimization.",
                attributes={
                    "role": "Data Scientist",
                    "company": "The Home Depot",
                    "start_year": "2019",
                    "end_year": "2023",
                    "description": "Led ML model deployment for inventory optimization."
                }
            ),
            # --- Complex/Nested Entity: Work Experience (Instance 2) ---
            lx.data.Extraction(
                extraction_class="work_experience",
                extraction_text="Research Associate at University Y (2018-2019). Published 2 papers on NLP model efficiency.",
                attributes={
                    "role": "Research Associate",
                    "company": "University Y",
                    "start_year": "2018",
                    "end_year": "2019",
                    "description": "Published 2 papers on NLP model efficiency."
                }
            ),
            
            # --- Complex/Nested Entity: Education ---
            lx.data.Extraction(
                extraction_class="education",
                extraction_text="M.S. in Data Science, University X, Graduated 2019. Thesis on predictive analysis.",
                attributes={
                    "degree": "M.S. in Data Science",
                    "institution": "University X",
                    "graduation_year": "2019",
                    "description": "Thesis on predictive analysis."
                }
            ),
            
        ]
    )
]


# ----------------------------------------------------------

def handle_extraction(resume_text: str, white_text_results: dict):
    # ... (function body remains the same) ...
    """Performs the LLM extraction and subsequent visualization/processing."""
    OUTPUT_FILE_JSONL = "extraction_results.jsonl"
    OUTPUT_FILE_HTML = "visualization.html"

    if not client or not LANGEXTRACT_API_KEY:
        st.error("Cannot start extraction: LangExtract Client is not properly initialized.")
        return
    
    # Clean the text before sending to LLM
    resume_text = sanitize_text(resume_text)

    # 1. LLM Extraction
    try:
        with st.spinner("Starting LangExtract process (this may take 10-30 seconds)..."):
            lx_result = lx.extract(
                                    text_or_documents=resume_text,
                                    prompt_description=prompt,
                                    examples=examples,
                                    model_id=MODEL_NAME,
                                    extraction_passes=1,
                                    max_workers=5,
                                    max_char_buffer=5000,
                                )
            ss.lx_result = lx_result

            print('LLM extraction result:', lx_result)

            # Save result for visualization
            lx.io.save_annotated_documents([lx_result], output_name=OUTPUT_FILE_JSONL, output_dir=".")
            st.success("LLM extraction completed.")
    
    except Exception as e:
        st.error(f"❌ **LangExtract/Gemini Extraction Failed**: Check your LANGEXTRACT_API_KEY or if the Gemini API is enabled for your project. Error: {e}")
        return

# ----------------------------------------------------------

    st.divider()
    st.subheader("👀 Visualization")
    html_content = lx.visualize(OUTPUT_FILE_JSONL)
    html_data = getattr(html_content, 'data', html_content)

    with open(OUTPUT_FILE_HTML, "w") as f:
        f.write(html_data)

    st.html(html_data)

    #   Pandas DataFrame Summary
    lx_result_reloaded = load_lx_results_from_jsonl(OUTPUT_FILE_JSONL)[0]
    extraction_df = process_lx_output_to_dataframe(lx_result_reloaded, ss.uploaded_file_name, white_text_results)

    st.divider()
    st.subheader("💾 Structured Data Summary")
    st.dataframe(extraction_df, width='stretch')

    if st.button("Save Results to BigQuery", key="save_bq_button", type="primary"):
        save_to_bigquery(extraction_df)

    try:
        if os.path.exists(OUTPUT_FILE_HTML): os.remove(OUTPUT_FILE_HTML)
        if os.path.exists(OUTPUT_FILE_JSONL): os.remove(OUTPUT_FILE_JSONL)
    except Exception as e:
        print(f"Cleanup failed: {e}")

# ----------------------------------------------------------

def check_environment_vars():
    # ... (function body remains the same) ...
    """Checks for critical environment variables and displays errors if missing."""
    critical_vars = {
        "LANGEXTRACT_API_KEY": LANGEXTRACT_API_KEY,
        "PROJECT_ID": PROJECT_ID,
        "BUCKET_NAME": BUCKET_NAME,
        "GOOGLE_APPLICATION_CREDENTIALS": GOOGLE_APPLICATION_CREDENTIALS,
    }
    
    missing_vars = [name for name, value in critical_vars.items() if not value]
    
    if missing_vars:
        st.error(f"🚨 **CRITICAL CONFIGURATION ERROR** 🚨")
        st.markdown(f"The following environment variables are **missing or empty**: **{', '.join(missing_vars)}**.")
        st.markdown("Please ensure they are correctly defined in your `.env` file or environment settings.")
        return False
    return True

# ----------------------------------------------------------
# MAIN FUNCTION
# ----------------------------------------------------------

def main():
    """Main Streamlit app function."""
    # Set page config to wide for a better two-column view
    st.set_page_config(layout="wide", page_title="Gemini Resume Parser")

    st.title("📄 Resume Parser App")
    st.write(f"Uses **{MODEL_NAME}** and GCS bucket **{BUCKET_NAME}** (Project: **{PROJECT_ID}**).")
    st.divider()

    # CRITICAL CHECK: Stop execution if environment variables are not set
    if not check_environment_vars():
        return

    # Initialize session state (using alias ss for st.session_state)
    if 'uploaded_file_name' not in ss:
        ss.uploaded_file_name = None
        ss.signed_url = None
        ss.resume_text = None
        ss.lx_result = None
        ss.white_text_results = None
        ss.pdf_bytes = None  # Ensure PDF bytes are initialized
        # Initialize the control variable for the dropdown to the default
        ss.selected_gcs_file = "Select an existing file..."


    # --- DEFINE COLUMNS (e.g., 60% width for controls/results, 40% for PDF) ---
    col_left, col_right = st.columns([0.6, 0.4])

    # --------------------------------------------------------------------------
    # LEFT COLUMN: INPUT CONTROLS, EXTRACTION TRIGGER, AND TEXT RESULTS
    # --------------------------------------------------------------------------
    with col_left:
        
        st.subheader("🔍 Select and Process Resume")
        
        # 1. Get the list of PDFs from GCS
        gcs_files = list_gcs_pdfs(BUCKET_NAME)
        options_list = ["Select an existing file..."] + gcs_files

        # 2. Dropdown Selector (for existing files)
        selected_file = st.selectbox(
            "1. Choose a file from GCS:",
            options=options_list,
            # Use key to link to ss.selected_gcs_file (resolves the previous warning)
            key="selected_gcs_file" 
        )

        # 3. File Uploader (to add a new file)
        with st.expander("2. Upload a new file to the bucket", expanded=False):
            uploaded_file = st.file_uploader("Choose a Resume PDF to upload", type="pdf")
            
            # --- UPLOAD/REPROCESS LOGIC ---
            if uploaded_file:
                # Use file name check to prevent reprocessing if already handled
                if ss.uploaded_file_name != uploaded_file.name:
                    handle_file_upload(uploaded_file)
            
        # --- FILE SELECTION CHANGE LOGIC ---
        # Check if a file was selected/changed AND it's not the default option
        # AND it's a different file from the one currently being processed
        if (selected_file != "Select an existing file..." and 
            ss.uploaded_file_name != selected_file):
            
            with st.spinner(f"Processing **{selected_file}** from GCS..."):
                process_file_from_gcs(selected_file)
                # Store the selected name
                ss.uploaded_file_name = selected_file
                
            # Rerun to display the results and make the extraction button visible
            st.rerun() 
            
        st.divider()

        # --- EXTRACTION TRIGGER ---
        if ss.resume_text:
            st.subheader("⚒️ Extraction")
            
            if st.button(f"Start LLM Extraction for: **{ss.uploaded_file_name}**", type="primary", use_container_width=True):
                # The session state variables are passed to handle_extraction
                handle_extraction(ss.resume_text, ss.white_text_results)

        # --- HIDDEN TEXT RESULT DISPLAY ---
        if ss.white_text_results is not None:
            white_text_results = ss.white_text_results
            
            st.divider()
            
            if white_text_results.get('flagged'):
                st.subheader("🚨 Hidden Text Analysis Result")
                st.error("🚨 **WHITE TEXT DETECTED (ATS MANIPULATION FLAG)** 🚨")
                
                with st.expander(f"Total hidden text segments found: {white_text_results['total_findings']}", expanded=True):
                    for finding in white_text_results['findings']:
                        st.markdown(f"> **Page {finding['page']}** (BBox: {finding['bbox']})")
                        st.markdown(f"**REVEALED TEXT**: `{finding['text']}`")
                        st.markdown("---")
                
            elif ss.uploaded_file_name:
                st.success(f"✅ **Hidden Text Check for {ss.uploaded_file_name}**: No white text detected. Resume seems clean.")

    # --------------------------------------------------------------------------
    # RIGHT COLUMN: PDF PREVIEW
    # --------------------------------------------------------------------------
    with col_right:
        st.subheader("Resume Preview")
        
        if ss.pdf_bytes:
            # Display the PDF using the function defined in the previous response
            # Note: We use the full height of the column for better viewing
            display_pdf_bytes(ss.pdf_bytes, display_height=850) 
        else:
            st.info("Select a file from the left column to view the PDF here.")

# ----------------------------------------------------------
# RUN THE APP
# ----------------------------------------------------------


if __name__ == '__main__':

    # Run: streamlit run resume_parser_app.py
    main()


