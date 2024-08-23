import os
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import dotenv
from sqlalchemy import Column, Integer, JSON, create_engine, inspect
from sqlalchemy.orm import Session, declarative_base
from tidb_vector.sqlalchemy import VectorType
from langchain.document_loaders import PyPDFLoader
from tidb_vector.integrations import TiDBVectorClient
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tempfile

dotenv.load_dotenv()

app = FastAPI()

# TiDB and Jina AI setup
TIDB_DATABASE_URL = os.getenv('TIDB_DATABASE_URL')
JINAAI_API_KEY = os.getenv('JINAAI_API_KEY')

assert TIDB_DATABASE_URL is not None
assert JINAAI_API_KEY is not None

engine = create_engine(url=TIDB_DATABASE_URL, pool_recycle=300)
Base = declarative_base()

# Global dictionary to store table models
table_models = {}

def generate_embeddings(text: str):
    JINAAI_API_URL = 'https://api.jina.ai/v1/embeddings'
    JINAAI_HEADERS = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {JINAAI_API_KEY}'
    }
    JINAAI_REQUEST_DATA = {
        'input': [text],
        'model': 'jina-embeddings-v2-base-en'
    }
    response = requests.post(JINAAI_API_URL, headers=JINAAI_HEADERS, json=JINAAI_REQUEST_DATA)
    return response.json()['data'][0]['embedding']

# Dynamic table creation
def create_document_table(table_name):
    if table_name in table_models:
        return table_models[table_name]

    class Document(Base):
        __tablename__ = table_name

        id = Column(Integer, primary_key=True)
        content_vec = Column(
            VectorType(dim=768),
            comment="hnsw(distance=cosine)"
        )
        mdata = Column(JSON)

    # Check if the table already exists in the database
    if not inspect(engine).has_table(table_name):
        Base.metadata.create_all(engine, tables=[Document.__table__])

    table_models[table_name] = Document
    return Document

# Pydantic model for request body
class PDFRequest(BaseModel):
    filekey: str
    fileurl: str

@app.post("/process_pdf")
async def process_pdf(request: PDFRequest):
    try:
        # Create a new table for this file
        
        DocumentModel = create_document_table(request.filekey)

        # Download and process the PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            response = requests.get(request.fileurl)
            temp_file.write(response.content)
            temp_file_path = temp_file.name
            print(temp_file_path)

        loader = PyPDFLoader(temp_file_path)
        documents = loader.load()

        # Split the documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(documents)

        # Process and store chunks
        with Session(engine) as session:
            for i, split in enumerate(splits):
                content = split.page_content
                mdata = {
                    "pageNumber": split.metadata.get("page", None),
                    "text": content,  # Store full content in mdata
                }
                embedding = generate_embeddings(content)

                doc = DocumentModel(
                    content_vec=embedding,
                    mdata=mdata
                )
                session.add(doc)
            session.commit()

        # Clean up the temporary file
        os.unlink(temp_file_path)

        return {"message": f"PDF processed and stored in table {request.filekey}"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
# New Pydantic model for the query request
class QueryRequest(BaseModel):
    query: str
    filekey: str

@app.post("/search")
async def search(request: QueryRequest):
    try:
        # Generate embedding for the query
        print(request)
        query_embedding = generate_embeddings(request.query)

        # Get or create the table class
        DocumentModel = create_document_table(request.filekey)

        with Session(engine) as session:
            doc = session.query(
                DocumentModel,
                DocumentModel.content_vec.cosine_distance(query_embedding).label('distance')
            ).order_by(
                'distance'
            ).limit(3).all()
            search_results = []
            for row, distance in doc:
                search_results.append({
                    "content": row.mdata,
                })

        return {"results": search_results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)