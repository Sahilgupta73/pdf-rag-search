# PDF RAG Search

A local Streamlit app that finds the **top 3 relevant text chunks** from a selected PDF. Choose Python, Machine Learning, or Data Visualization, then enter a question to see matching passages from the document.

The app displays the retrieved PDF text directly. No LLM or Groq API key is required.

## How it works

1. **pypdf** extracts text from the PDF.
2. **LangChain** splits the text into smaller chunks.
3. **Hugging Face MiniLM** converts each chunk into an embedding.
4. **Chroma** stores the embeddings and finds the closest matches to your question.
5. **Streamlit** displays up to three results.

## Run locally on Windows

Clone the repository and create a virtual environment:

```powershell
git clone https://github.com/Sahilgupta73/pdf-rag-search.git
cd pdf-rag-search
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install streamlit pypdf langchain-core langchain-text-splitters langchain-huggingface langchain-chroma sentence-transformers
```

### Add your PDFs

**PDFs are not included in this repository.** Place your own text-readable PDFs in the same folder as `app2.py`. Use the exact filenames below. You only need the PDF for the category you want to search.

| Streamlit option | Required filename | Python module |
| --- | --- | --- |
| Python | `Python.pdf` | `rag_1.py` |
| Machine Learning | `ML.pdf` | `rag_2.py` |
| Data Visualization | `DV.pdf` | `rag_3.py` |

### Start the app

```powershell
.\.venv\Scripts\python.exe -m streamlit run app2.py --server.address 127.0.0.1
```

Open [http://localhost:8501](http://localhost:8501). Choose a PDF from the sidebar and try a question such as **“What is Python?”** or **“What is linear regression?”**

The first search may take longer while the app downloads the embedding model and builds the selected PDF’s Chroma index. Later searches reuse that index.

## Troubleshooting

- **Wrong or outdated results after replacing a PDF:** Stop the app and rename the Chroma folder configured in that PDF’s `rag_*.py` file. Restart the app to build a fresh index.
- **No extractable text:** Use a PDF with selectable text. Scanned pages need OCR before indexing.
