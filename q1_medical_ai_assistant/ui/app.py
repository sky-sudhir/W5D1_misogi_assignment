import streamlit as st
import requests

API_URL = "http://localhost:8000"  # FastAPI base URL

st.set_page_config(page_title="🩺 Medical Knowledge Assistant", layout="wide")
st.title("🩺 Medical RAG Assistant with RAGAS Monitoring")

st.markdown("Query clinical guidelines, drug interactions, and medical literature safely.")

# --- Ingestion Section ---
st.sidebar.header("📥 Upload DrugBank CSV")
upload_file = st.sidebar.file_uploader("Upload CSV or PDF", type=["csv", "pdf"])

if upload_file is not None:
    with st.spinner("Uploading and indexing..."):
        files = {"file": upload_file.getvalue()}
        res = requests.post(f"{API_URL}/ingest/drugbank", files={"file": upload_file})
        if res.status_code == 200:
            st.sidebar.success("Uploaded and indexed!")
        else:
            st.sidebar.error("Upload failed.")

# --- Query Section ---
st.subheader("🔎 Ask a Medical Question")

query = st.text_input("Enter your question", placeholder="e.g. Can I take ibuprofen with warfarin?")
submit = st.button("Generate Answer")

if submit and query:
    with st.spinner("Retrieving answer..."):
        response = requests.post(f"{API_URL}/query", json={"question": query})
        if response.status_code == 200:
            result = response.json()

            # Answer display
            st.markdown("### 💬 Answer")
            st.success(result["answer"])

            # Source display
            st.markdown("### 📚 Source Chunks")
            for idx, src in enumerate(result["sources"]):
                st.markdown(f"**Chunk {idx+1}:**\n```{src}```")

            # RAGAS Metrics
            st.markdown("### 📊 RAGAS Metrics")
            metrics = result["raga_metrics"]
            cols = st.columns(4)

            def highlight(metric, threshold):
                return "🟢" if metric >= threshold else "🔴"

            cols[0].metric("Faithfulness", f"{metrics['faithfulness']:.2f}", highlight(metrics['faithfulness'], 0.90))
            cols[1].metric("Relevancy", f"{metrics['answer_relevancy']:.2f}", highlight(metrics['answer_relevancy'], 0.90))
            cols[2].metric("Context Precision", f"{metrics['context_precision']:.2f}", highlight(metrics['context_precision'], 0.85))
            cols[3].metric("Context Recall", f"{metrics['context_recall']:.2f}", highlight(metrics['context_recall'], 0.85))

            # Warning if poor faithfulness
            if metrics["faithfulness"] < 0.9:
                st.error("⚠️ Warning: This response may not be clinically faithful.")
        else:
            st.error("Failed to get response from backend.")
