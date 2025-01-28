This Python script "code.py" demonstrates how to use the Together API for generating responses via a chat model. Here's a step-by-step explanation of the code:
   1. Load API Key
The script uses the dotenv library to load environment variables from a .env file.
TOGETHER_API_KEY is the API key needed to authenticate requests to the Together API.
If the API key is not set, an error is raised to prevent the script from proceeding.
  2. Initialize the Together Client
The script creates a client object (client) for the Together API, initialized using the API key.
  3. Make a Chat Completion Request
The script attempts to generate a response using the Together API by invoking the chat.completions.create() method.
It specifies:
Model: meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo (a variant of Meta's Llama language model).
Prompt: The user prompt, "What are the top 3 things to do in New York?"
Streaming: Enabled via stream=True to receive the response incrementally.
The script loops through the streamed response, printing it chunk by chunk
Error Handling: If an exception occurs, it prints the error message.

Extract Metadata from LLM 
1. Environment Setup
Loading Environment Variables: The dotenv library is used to load the TOGETHER_API_KEY, which is required to interact with the Together API (a chat-based AI model API).
Together API Client: A client is initialized using this API key to access Together's services.

2. Model and Prompt Configuration
Model ID: The NLP model used here is "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo".
System Prompt: A detailed instruction is crafted to guide the NLP model in extracting specific metadata fields (e.g., title, authors, abstract) from research papers.
The system prompt specifies:

Key Information to extract (e.g., title, year, authors, contact details, etc.).
Guidelines for extraction (e.g., accuracy, conciseness, and format).

3. Data Model
Pydantic Models: These are used to define the structure of the extracted metadata in Python. The PaperMetadata model includes:
PaperTitle, PublicationYear, Authors, AuthorContact (author details), Abstract, and SummaryAbstract

4. PDF Text Extraction
extract_text_from_pdf Function:
Uses LangChain's UnstructuredPDFLoader to read the content of a PDF.
Outputs the text content, which is later processed.

5. Metadata Extraction
extract_metadata Function:
Sends the extracted PDF text and system prompt to the Together API.
Uses exponential backoff (via tenacity) to handle retries if the API call fails.
The API response is parsed into JSON format, structured according to the PaperMetadata schema.

Key Libraries and Concepts Used
LangChain: For text extraction from PDFs.
Pydantic: To validate and enforce JSON schema.
Tenacity: For retrying API calls with exponential backoff.
Together API: An NLP service for extracting structured information from unstructured text.


Extract datameta with LLM ollama 
Input:

PDF file containing a research paper (e.g., 1706.03762v7.pdf).
A prompt file (prompt.txt) with instructions for metadata extraction.
The Llama model is hosted locally via Ollama.
Process:

Text content is extracted from the PDF.
Metadata (e.g., title, authors, abstract) is extracted using the NLP model.
Metadata is saved as a JSON file.
Output:

For 1706.03762v7.pdf, a JSON file like 1706.03762v7.json is created with extracted metadata.


