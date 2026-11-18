
import os
from pathlib import Path
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# ---------------- Settings ----------------
BASE_DIR = Path(__file__).resolve().parent

PDF_PATH = BASE_DIR / "Python.pdf"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_FOLDER = str(BASE_DIR / "saved_model")
# VECTOR_DB_FOLDER = str(BASE_DIR / "chroma_db_1")
VECTOR_DB_FOLDER = str(BASE_DIR / "chroma_db_python_v2")

# -------- Step 1: Load Embedding Model --------
embedding_model = HuggingFaceEmbeddings(
    model_name=MODEL_NAME,
    cache_folder=MODEL_FOLDER
)

# -------- Step 2: Read PDF and Split into Chunks --------
def read_pdf_into_chunks():
    reader = PdfReader(str(PDF_PATH))

    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        if text.strip():
            pages.append(
                Document(
                    page_content=text,
                    metadata={"page": page_number}
                )
            )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(pages)

    return chunks

# -------- Step 3: Create Vector Database --------
def create_vector_db():
    chunks = read_pdf_into_chunks()

    if not chunks:
        raise ValueError("No extractable text found in PDF.")

    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=VECTOR_DB_FOLDER
    )

    return vector_db

# -------- Load Existing Vector Database --------
def load_vector_db():
    return Chroma(
        persist_directory=VECTOR_DB_FOLDER,
        embedding_function=embedding_model
    )

# -------- Load Everything Once --------
if os.path.exists(VECTOR_DB_FOLDER):
    print("Loading saved embeddings...")
    vector_db = load_vector_db()
else:
    print("Creating embeddings (first time only)...")
    vector_db = create_vector_db()

# -------- Step 4: Similarity Search --------
def get_relevant_info(question, top_k=3):
    results = vector_db.similarity_search(
        question,
        k=top_k
    )

    texts = []

    for doc in results:
        texts.append(doc.page_content)

    return texts

# -------- Step 5: Chat Loop --------
if __name__ == "__main__":
    print("\nPDF chatbot is ready! Type 'exit' or 'quit' to stop\n")

    while True:
        question = input("You: ").strip()

        if question.lower() in ["exit", "quit"]:
            print("Bye!")
            break

        if question == "":
            continue

        answers = get_relevant_info(question)

        for i, text in enumerate(answers, start=1):
            print(f"\n----- Result {i} -----")
            print(text)

        print()
