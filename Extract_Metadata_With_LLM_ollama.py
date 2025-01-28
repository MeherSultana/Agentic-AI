import os
import re
import json
from pathlib import Path
from dotenv import load_dotenv

# LangChain + Unstructured
from langchain_community.document_loaders.pdf import UnstructuredPDFLoader

# Retry library
from tenacity import retry, wait_random_exponential, stop_after_attempt

# Ollama Python bindings
import ollama

# Load environment variables if needed (optional)
load_dotenv()

# Example model name for Ollama. Must match what you've pulled.
MODEL_ID = "llama-3.2-8b"

def read_prompt(prompt_path: str) -> str:
    """
    Read the prompt for research paper parsing from a text file.
    """
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text content from a PDF using LangChain's UnstructuredPDFLoader.
    """
    loader = UnstructuredPDFLoader(pdf_path)
    documents = loader.load()
    text_content = "\n".join(doc.page_content for doc in documents)
    return text_content

@retry(wait=wait_random_exponential(min=1, max=120), stop=stop_after_attempt(10))
def ollama_completion_with_backoff(model_id: str, prompt: str) -> str:
    """
    Call Ollama locally, and assemble output from streaming chunks.
    Wrapped with tenacity for exponential backoff retries.
    """
    generated_text = []
    for chunk in ollama.generate(model=model_id, prompt=prompt):
        # chunk is a dict like: {"text": "..."} that streams partial responses
        generated_text.append(chunk["text"])
    return "".join(generated_text)

def extract_metadata(content: str, prompt_path: str, model_id: str) -> dict:
    """
    Use a local Llama model (via Ollama) to extract metadata 
    from the research paper content based on the given prompt.
    """
    # 1. Read the prompt
    prompt_data = read_prompt(prompt_path)

    # 2. Combine your instructions + the paper content into one string
    combined_prompt = (
        f"{prompt_data}\n\n"
        f"Here is the paper content:\n\n"
        f"{content}\n\n"
        "Please extract the relevant metadata in JSON format."
    )

    try:
        # 3. Call Ollama to get completion text
        response_content = ollama_completion_with_backoff(model_id, combined_prompt).strip()

        if not response_content:
            print("Empty response from the model.")
            return {}

        # 4. Clean up any markdown code block indicators
        response_content = re.sub(r'```json\s*', '', response_content)
        response_content = re.sub(r'\s*```', '', response_content)

        # 5. Parse JSON if possible
        try:
            return json.loads(response_content)
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON: {e}")
            print(f"Raw response: {response_content}")

            # Attempt to extract a JSON object if there's extra text
            match = re.search(r'\{.*\}', response_content, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except json.JSONDecodeError as jde:
                    print(f"Failed to extract valid JSON from the response: {jde}")
            return {}

    except Exception as e:
        print(f"Error calling Ollama: {e}")
        return {}

def process_research_paper(pdf_path: str, prompt_path: str,
                           output_folder: str, model_id: str):
    """
    Process a single research paper through the entire pipeline.
    """
    print(f"Processing research paper: {pdf_path}")

    try:
        # Step 1: Extract text from PDF
        content = extract_text_from_pdf(pdf_path)
        print(f"Extracted text content from: {pdf_path}")

        # Step 2: Extract metadata via local Llama (Ollama)
        metadata = extract_metadata(content, prompt_path, model_id)
        if not metadata:
            print(f"Failed to extract metadata for {pdf_path}")
            return
        print(f"Extracted metadata using {model_id} for {pdf_path}")

        # Step 3: Save the result as a JSON file
        output_filename = Path(pdf_path).stem + '.json'
        output_path = os.path.join(output_folder, output_filename)
        with open(output_path, 'w', encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        print(f"Saved metadata to {output_path}")

    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")

def process_directory(prompt_path: str, directory_path: str, output_folder: str, model_id: str):
    """
    Process all PDF files in the given directory.
    """
    for filename in os.listdir(directory_path):
        if filename.lower().endswith('.pdf'):
            pdf_path = os.path.join(directory_path, filename)
            process_research_paper(pdf_path, prompt_path, output_folder, model_id)

if __name__ == "__main__":
    # Example usage
    pdf_path = "data/1706.03762v7.pdf"
    prompt_path = "prompt.txt"
    output_folder = "extracted_metadata"

    process_research_paper(pdf_path, prompt_path, output_folder, MODEL_ID)

    # Or process a directory of PDFs
    directory_path = "data"
    process_directory(prompt_path, directory_path, output_folder, MODEL_ID)
