from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import io
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from tortoise.contrib.fastapi import register_tortoise
import numpy as np
from models import TextChunk, Embedding

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

app = FastAPI()

# CORS configuration
origins = [
    "http://localhost:3000",
    "https://magnificent-ganache-b464dc.netlify.app/",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Tortoise ORM
register_tortoise(
    app,
    db_url="sqlite://db.sqlite3",
    modules={"models": ["models"]},
    generate_schemas=True,
    add_exception_handlers=True,
)

# Temporary storage for uploaded file
uploaded_file = None

class Document:
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata if metadata is not None else {}

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        if isinstance(pdf, bytes):
            # Handle the case when a single file (bytes object) is uploaded
            pdf_reader = PdfReader(io.BytesIO(pdf))
        else:
            # Handle the case when multiple files (file paths) are uploaded
            pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

async def store_embeddings(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    for chunk in text_chunks:
        vector = list(embeddings.embed_query(chunk))  # Convert to list
        text_chunk_obj = await TextChunk.create(content=chunk)
        await Embedding.create(text_chunk=text_chunk_obj, vector=vector)

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

@app.post("/upload-file")
async def upload_file(file: UploadFile = File(...)):
    global uploaded_file
    uploaded_file = await file.read()
    return JSONResponse(content={"message": "File uploaded successfully. What would you like to do next?", "options": ["Ask a question", "Generate questions"]})

@app.post("/ask-question")
async def ask_question(question: str = Form(...)):
    if uploaded_file is None:
        return JSONResponse(content={"error": "No file uploaded."}, status_code=400)
    
    pdf_files = [uploaded_file]
    raw_text = get_pdf_text(pdf_files)
    text_chunks = get_text_chunks(raw_text)
    await store_embeddings(text_chunks)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    question_embedding = list(embeddings.embed_query(question))  # Convert to list
    
    # Retrieve all text chunks and their embeddings
    text_chunks = await TextChunk.all().prefetch_related('embeddings')
    similarities = []
    
    for chunk in text_chunks:
        for embedding in chunk.embeddings:
            vector = np.array(embedding.vector)
            similarity = np.dot(question_embedding, vector) / (np.linalg.norm(question_embedding) * np.linalg.norm(vector))
            similarities.append((chunk.content, similarity))
    
    # Get the most relevant text chunk
    most_relevant_chunk = max(similarities, key=lambda item: item[1])[0]
    
    # Create a Document object with the most relevant chunk
    relevant_document = Document(page_content=most_relevant_chunk)

    chain = get_conversational_chain()
    response = chain({"input_documents": [relevant_document], "question": question}, return_only_outputs=True)
    return JSONResponse(content={"answer": response["output_text"]})

@app.post("/generate-questions")
async def generate_questions(question_type: str = Form(...)):
    if uploaded_file is None:
        return JSONResponse(content={"error": "No file uploaded."}, status_code=400)
    
    pdf_files = [uploaded_file]
    raw_text = get_pdf_text(pdf_files)
    text_chunks = get_text_chunks(raw_text)

    question_chain = get_question_generation_chain(question_type)
    
    questions = []
    for chunk in text_chunks:
        relevant_document = Document(page_content=chunk)
        response = question_chain({"input_documents": [relevant_document]}, return_only_outputs=True)
        questions.append(response["output_text"])

    return JSONResponse(content={"questions": questions})

def get_question_generation_chain(question_type: str):
    if question_type == "choice":
        prompt_template = """
        Generate a multiple-choice question based on the following text. Provide four options and indicate the correct answer.\n\n
        Context:\n {context}\n
        Question:
        """
    elif question_type == "true/false":
        prompt_template = """
        Generate a true/false question based on the following text. Indicate the correct answer.\n\n
        Context:\n {context}\n
        Question:
        """
    elif question_type == "shortanswer":
        prompt_template = """
        Generate a short answer question based on the following text. Provide the correct answer.\n\n
        Context:\n {context}\n
        Question:
        """
    else:
        raise ValueError("Invalid question type")

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)

    prompt = PromptTemplate(template=prompt_template, input_variables=["context"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
