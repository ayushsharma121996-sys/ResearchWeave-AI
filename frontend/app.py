"""
Minimal Streamlit UI for ResearchPilot AI.

Deliberately thin -- the project's value is in the pipeline, not the UI.
Talks to the FastAPI backend over HTTP so the two stay decoupled (you could
swap this for a React frontend later without touching the pipeline code).

Run:
    uvicorn app.api:app --reload          # in one terminal
    streamlit run frontend/app.py         # in another
"""
import requests
import streamlit as st

API_BASE = "http://localhost:8000"

st.set_page_config(page_title="ResearchPilot AI", layout="wide")
st.title("📚 ResearchPilot AI")
st.caption("Upload research papers → ask grounded questions → get a structured, cited comparison report.")

with st.sidebar:
    st.header("1. Upload papers")
    uploaded = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
    if uploaded and st.button("Index papers"):
        files = [("files", (f.name, f.getvalue(), "application/pdf")) for f in uploaded]
        with st.spinner("Parsing, chunking, and indexing..."):
            resp = requests.post(f"{API_BASE}/papers/upload", files=files)
        if resp.ok:
            data = resp.json()
            st.success(f"Indexed {len(data['papers_ingested'])} papers, {data['total_chunks']} chunks.")
            st.session_state["papers_ready"] = True
        else:
            st.error(resp.text)

tab1, tab2 = st.tabs(["💬 Ask a question", "📊 Compare papers"])

with tab1:
    question = st.text_input("Ask a grounded question about the uploaded papers")
    if st.button("Ask") and question:
        with st.spinner("Retrieving + answering..."):
            resp = requests.post(f"{API_BASE}/ask", json={"question": question})
        if resp.ok:
            data = resp.json()
            st.markdown(data["answer"])
            with st.expander("Sources"):
                for s in data["sources"]:
                    st.write(f"- **{s['paper_id']}** ({s['section']}, p.{s['page']})")
        else:
            st.error(resp.text)

with tab2:
    topic = st.text_input("Topic for the report", value="the uploaded papers")
    if st.button("Generate comparison report"):
        with st.spinner("Extracting structured data, cross-checking claims, writing report... this can take a minute for several papers."):
            resp = requests.post(f"{API_BASE}/compare", json={"topic": topic})
        if resp.ok:
            data = resp.json()
            st.caption(f"{data['num_papers']} papers · {data['num_claims_analyzed']} claims cross-checked")
            st.markdown(data["report_markdown"])
        else:
            st.error(resp.text)
