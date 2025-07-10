# main.py
import nest_asyncio
nest_asyncio.apply()

from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from ingest import ingest_file, chunk_documents
from chroma_client import get_vectorstore
from rag_pipeline import get_rag_chain
import os
from raga_eval import evaluate_single_sample
import uvicorn

from dotenv import load_dotenv

load_dotenv()


app = FastAPI()
qa_chain = get_rag_chain()

class QueryRequest(BaseModel):
    question: str


@app.post("/query")
def query_rag(request: QueryRequest):
    result = qa_chain(request.question)
    answer = result["result"]
    sources = [doc.page_content for doc in result["source_documents"]]

    metrics = evaluate_single_sample(
        question=request.question,
        contexts=sources,
        answer=answer,
    )

    return {
        "answer": answer,
        "sources": sources,
        "raga_metrics": metrics,
    }


@app.post("/ingest/drugbank")
async def ingest_drugbank(file: UploadFile = File(...)):
    contents = await file.read()
    temp_path = f"./data/{file.filename}"
    with open(temp_path, "wb") as f:
        f.write(contents)

    docs = ingest_file(temp_path)
    chunks = chunk_documents(docs)

    vectorstore = get_vectorstore()
    vectorstore.add_documents(chunks)

    os.remove(temp_path)
    return {"message": f"Indexed {len(chunks)} chunks"}


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    ) 