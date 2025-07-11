import asyncio
from typing import Dict, Any, List
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from chroma_client import get_vectorstore
from config import settings
import structlog

logger = structlog.get_logger(__name__)


async def get_async_rag_chain():
    """Get async RAG chain for financial intelligence queries"""
    try:
        # Get retriever
        vectorstore = get_vectorstore()
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )
        
        # Initialize LLM
        llm = ChatGoogleGenerativeAI(
            model=settings.MODEL_NAME,
            temperature=settings.MODEL_TEMPERATURE,
            max_tokens=settings.MAX_TOKENS,
            max_retries=3,
            google_api_key=settings.GOOGLE_API_KEY
        )
        
        # Create financial-specific prompt
        financial_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """You are an expert financial analyst AI assistant specializing in corporate financial data analysis. 
                Your role is to provide accurate, insightful analysis of financial reports, earnings data, and corporate metrics.
                
                Guidelines:
                - Focus on financial metrics, trends, and comparative analysis
                - Provide specific numbers and percentages when available
                - Explain financial concepts clearly
                - Highlight key insights and potential risks
                - Use the provided context to support your analysis
                - If information is not available in the context, clearly state this limitation
                
                Context: {context}
                """
            ),
            ("human", "Question: {question}"),
            ("assistant", "Based on the financial data provided, here's my analysis:")
        ])
        
        # Create async RAG chain
        rag_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": financial_prompt}
        )
        
        logger.info("Async RAG chain initialized successfully")
        return rag_chain
        
    except Exception as e:
        logger.error(f"Failed to initialize async RAG chain: {e}")
        raise


def get_rag_chain():
    """Synchronous wrapper for backward compatibility"""
    try:
        vectorstore = get_vectorstore()
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )
        
        llm = ChatGoogleGenerativeAI(
            model=settings.MODEL_NAME,
            temperature=settings.MODEL_TEMPERATURE,
            max_tokens=settings.MAX_TOKENS,
            max_retries=3,
            google_api_key=settings.GOOGLE_API_KEY
        )
        
        financial_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """You are an expert financial analyst AI assistant specializing in corporate financial data analysis. 
                Your role is to provide accurate, insightful analysis of financial reports, earnings data, and corporate metrics.
                
                Guidelines:
                - Focus on financial metrics, trends, and comparative analysis
                - Provide specific numbers and percentages when available
                - Explain financial concepts clearly
                - Highlight key insights and potential risks
                - Use the provided context to support your analysis
                - If information is not available in the context, clearly state this limitation
                
                Context: {context}
                """
            ),
            ("human", "Question: {question}"),
            ("assistant", "Based on the financial data provided, here's my analysis:")
        ])
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": financial_prompt}
        )
        
        return qa_chain
        
    except Exception as e:
        logger.error(f"Failed to initialize RAG chain: {e}")
        raise


async def process_financial_query(query: str, context_docs: List[str] = None) -> Dict[str, Any]:
    """Process financial query with optional context documents"""
    try:
        rag_chain = await get_async_rag_chain()
        
        # Process query
        result = await rag_chain.ainvoke({"query": query})
        
        return {
            "answer": result["result"],
            "sources": [doc.page_content for doc in result["source_documents"]],
            "source_count": len(result["source_documents"])
        }
        
    except Exception as e:
        logger.error(f"Financial query processing failed: {e}")
        raise


async def extract_financial_metrics(company: str, metrics: List[str]) -> Dict[str, Any]:
    """Extract specific financial metrics for a company"""
    try:
        metrics_query = f"""
        Extract the following financial metrics for {company}:
        {', '.join(metrics)}
        
        Please provide specific values, percentages, and time periods where available.
        If any metric is not available, clearly state this.
        """
        
        result = await process_financial_query(metrics_query)
        
        return {
            "company": company,
            "requested_metrics": metrics,
            "analysis": result["answer"],
            "sources": result["sources"]
        }
        
    except Exception as e:
        logger.error(f"Financial metrics extraction failed: {e}")
        raise


async def compare_companies(companies: List[str], metrics: List[str]) -> Dict[str, Any]:
    """Compare financial metrics between multiple companies"""
    try:
        comparison_query = f"""
        Compare the financial performance of the following companies: {', '.join(companies)}
        
        Focus on these metrics: {', '.join(metrics)}
        
        Please provide:
        1. Side-by-side comparison of key metrics
        2. Relative performance analysis
        3. Strengths and weaknesses of each company
        4. Investment implications
        
        Use specific numbers and percentages where available.
        """
        
        result = await process_financial_query(comparison_query)
        
        return {
            "companies": companies,
            "compared_metrics": metrics,
            "comparison_analysis": result["answer"],
            "sources": result["sources"]
        }
        
    except Exception as e:
        logger.error(f"Company comparison failed: {e}")
        raise


async def analyze_financial_trends(company: str, time_period: str = "quarterly") -> Dict[str, Any]:
    """Analyze financial trends for a company over time"""
    try:
        trends_query = f"""
        Analyze the financial trends for {company} over the {time_period} period.
        
        Please provide:
        1. Revenue trends and growth rates
        2. Profitability trends
        3. Key financial ratios over time
        4. Notable changes or inflection points
        5. Forward-looking indicators
        
        Use specific numbers, percentages, and time periods.
        """
        
        result = await process_financial_query(trends_query)
        
        return {
            "company": company,
            "time_period": time_period,
            "trend_analysis": result["answer"],
            "sources": result["sources"]
        }
        
    except Exception as e:
        logger.error(f"Financial trend analysis failed: {e}")
        raise
