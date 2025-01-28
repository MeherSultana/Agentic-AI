import os
import re
import json
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List, Optional
from langchain_community.document_loaders.pdf import UnstructuredPDFLoader
from tenacity import retry, wait_random_exponential, stop_after_attempt
from together import Together

# 1. Load environment and Together API key
load_dotenv()
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
client = Together(api_key=TOGETHER_API_KEY)

# 2. Model name for JSON mode
MODEL_ID = "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo"

############################################################################
# Hard-coded prompt with your instructions
############################################################################
SYSTEM_PROMPT = """\
####
Scientific research paper:

{document)

You are an expert in analyzing scientific research papers. Please carefully read the provided research paper above and extract the following key information:

Extract these six (6) properties from the research paper:
- Paper Title: The full title of the research paper
- Publication Year: The year the paper was published
- Authors: The full names of all authors of the paper
- Author Contact: A list of dictionaries, where each dictionary contains the following keys for each author:
    - Name: The full name of the author
    - Institution: The institutional affiliation of the author
    - Email: The email address of the author (if provided)
- Abstract: The full text of the paper's abstract
- Summary Abstract: A concise summary of the abstract in 2-3 sentences, highlighting the key points

Guidelines:
- The extracted information should be factual and accurate to the document.
- Be extremely concise, except for the Abstract which should be copied in full.
- The extracted entities should be self-contained and easily understood without the rest of the paper.
- If any property is missing from the paper, please leave the field empty rather than guessing.
- For the Summary Abstract, focus on the main objectives, methods, and key findings of the research.
- For Author Contact, create an entry for each author, even if some information is missing. If an email or institution is not provided for an author, leave that

Answer in JSON format. The JSON should contain 6 keys: "PaperTitle", "PublicationYear", "Authors", "AuthorContact", "Abstract", and "SummaryAbstract".
"""

############################################################################
# Define a Pydantic model to represent the final JSON structure
############################################################################
class AuthorContactItem(BaseModel):
    Name: str = Field(..., description="Full name of the author")
    Institution: str = Field("", description="The institutional affiliation of the author")
    Email: str = Field("", description="The email address of the author, if provided")

class PaperMetadata(BaseModel):
    PaperTitle: str = Field("", description="The full title of the research paper")
    PublicationYear: str = Field("", description="The year the paper was published")
    Authors: List[str] = Field(default_factory=list, description="List of author names")
    AuthorContact: List[AuthorContactItem] = Field(
        default_factory=list, 
        description="List of dictionaries with details for each author"
    )
    Abstract: str = Field("", description="The full abstract text")
    SummaryAbstract: str = Field("", description="2-3 sentence summary of the abstract")

############################################################################
# PDF extraction
############################################################################
def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text content from a PDF using LangChain's UnstructuredPDFLoader.
    """
    loader = UnstructuredPDFLoader(pdf_path)
    documents = loader.load()
    text_content = "\n".join(doc.page_content for doc in documents)
    return text_content

############################################################################
# Together chat completions with exponential backoff
############################################################################
@retry(wait=wait_random_exponential(min=1, max=120), stop=stop_after_attempt(10))
def chat_completion_with_backoff(**kwargs):
    """Calls Together's chat.completions.create() with tenacity for exponential backoff."""
    return client.chat.completions.create(**kwargs)

############################################################################
# Main extraction logic
############################################################################
def extract_metadata(content: str, model_id: str) -> dict:
    """
    Use JSON Mode to extract structured metadata based on the system prompt
    plus the PDF text as user content.
    """
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": (
                f"Here is the paper content:\n\n{content}\n\n"
                "Please extract the relevant metadata strictly as valid JSON."
            )
        }
    ]

    try:
        response = chat_completion_with_backoff(
            model=model_id,
            messages=messages,
            # Using JSON Mode
            response_format={
                "type": "json_object",
                "schema": PaperMetadata.model_json_schema(),
            },
            temperature=0.2,
            max_tokens=1000
        )

        # Parse the returned content from the model
        if not response or "choices" not in response or len(response["choices"]) == 0:
            print("Empty response from the model.")
            return {}

        response_content = response["choices"][0]["message"]["content"].strip()
        if not response_content:
            print("Empty response from the model.")
            return {}

        # Attempt direct JSON parse
        try:
            return json.loads(response_content)
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON: {e}")
            print(f"Raw response:\n{response_content}")
            return {}

    except Exception as e:
        print(f"Error calling Together API: {e}")
        return {}

def process_research_paper(pdf_path: str, output_folder: str, model_id: str):
    """
    Process a single research paper through the entire pipeline.
    """
    print(f"Processing research paper: {pdf_path}")

    try:
        # 1. Extract text from the PDF
        content = extract_text_from_pdf(pdf_path)
        print(f"Extracted text content from: {pdf_path}")

        # 2. Extract metadata from the PDF text
        metadata = extract_metadata(content, model_id)
        if not metadata:
            print(f"Failed to extract metadata for {pdf_path}")
            return
        print(f"Extracted metadata using {model_id} for {pdf_path}")

        # 3. Save the result as a JSON file
        output_filename = Path(pdf_path).stem + '.json'
        output_path = os.path.join(output_folder, output_filename)
        with open(output_path, 'w', encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        print(f"Saved metadata to {output_path}")

    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")

def process_directory(directory_path: str, output_folder: str, model_id: str):
    """
    Process all PDF files in the given directory.
    """
    for filename in os.listdir(directory_path):
        if filename.lower().endswith('.pdf'):
            pdf_path = os.path.join(directory_path, filename)
            process_research_paper(pdf_path, output_folder, model_id)

############################################################################
# Usage
############################################################################
if __name__ == "__main__":
    # Example for a single PDF
    pdf_path = "data/1706.03762v7.pdf"
    output_folder = "extracted_metadata"
    os.makedirs(output_folder, exist_ok=True)

    # Process one PDF
    process_research_paper(pdf_path, output_folder, MODEL_ID)

    # Or process an entire directory
    directory_path = "data"
    process_directory(directory_path, output_folder, MODEL_ID)
