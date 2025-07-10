from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from chroma_client import get_vectorstore
from langchain_core.prompts import ChatPromptTemplate

def get_rag_chain():
    retriever = get_vectorstore().as_retriever()

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0,
        max_tokens=None,
        max_retries=2,
    )

    prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a medical assistant. Use the context below to answer the question accurately.",
        ),
        ("human", "Context: {context}"),
        ("human", "Question: {question}"),
        ("assistant", "Answer:"),
    ]
)
    # Create the RetrievalQA chain with the Gemini model and prompt
    # chain = prompt | llm 
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

    return qa_chain
