"""Qazaq Guide - Streamlit web UI for the RAG travel assistant.

Run:  streamlit run app.py
"""
import streamlit as st

from src.config import get_settings
from src.dataset import load_records
from src.embeddings_client import EmbeddingsClient
from src.llm_client import LLMClient
from src.rag import RAGPipeline
from src.vector_db import VectorStore

st.set_page_config(page_title="Qazaq Guide", page_icon="🏔️", layout="wide")

EXAMPLES = [
    "Куда поехать из Алматы на один день летом, чтобы увидеть горное озеро?",
    "Which historical places should I visit in Turkistan region?",
    "Хочу на море или пляж в Казахстане. Какие есть варианты?",
    "What should I know about ticks and safety in the mountains?",
    "Что посмотреть зимой недалеко от Алматы?",
]


@st.cache_resource
def get_pipeline():
    settings = get_settings()
    store = VectorStore(settings.qdrant_url, settings.collection_name)
    return settings, store, RAGPipeline(EmbeddingsClient(settings), store, LLMClient(settings))


@st.cache_data
def get_filter_options():
    records = load_records()
    return (
        sorted({r["category"] for r in records}),
        sorted({r["region"] for r in records}),
    )


st.title("🏔️ Qazaq Guide")
st.caption("AI travel assistant for Kazakhstan · answers are grounded in a curated knowledge base (RAG)")

try:
    settings, store, pipeline = get_pipeline()
    points = store.count()
except Exception as exc:  # show a clear error instead of a crash
    st.error(f"Cannot start the pipeline: {exc}")
    st.info("Check that Qdrant is running (docker compose up -d), that .env has your API key, "
            "and that you ran: python -m scripts.ingest")
    st.stop()

categories, regions = get_filter_options()

with st.sidebar:
    st.header("⚙️ Search settings")
    top_k = st.slider("How many documents to retrieve (top-k)", 1, 8, settings.top_k)
    threshold = st.slider("Minimum similarity score", 0.0, 0.9, 0.0, 0.05,
                          help="Documents less similar than this are ignored. 0 = no limit.")
    st.subheader("Filters (metadata)")
    season = st.selectbox("Season", ["any", "spring", "summer", "autumn", "winter"])
    budget = st.selectbox("Budget", ["any", "low", "medium", "high"])
    category = st.selectbox("Category", ["any", *categories])
    region = st.selectbox("Region", ["any", *regions])
    show_debug = st.toggle("Show the prompt sent to the LLM", value=False)
    st.divider()
    st.markdown(
        f"**Provider:** {settings.provider}  \n"
        f"**LLM:** `{settings.chat_model}`  \n"
        f"**Embeddings:** `{settings.embedding_model}`  \n"
        f"**Vector DB:** Qdrant · `{settings.collection_name}` · {points} vectors"
    )
    if st.button("🗑️ Clear chat"):
        st.session_state.messages = []
        st.rerun()

filters = {
    "season": None if season == "any" else season,
    "budget": None if budget == "any" else budget,
    "category": None if category == "any" else category,
    "region": None if region == "any" else region,
}

if "messages" not in st.session_state:
    st.session_state.messages = []


def render_sources(sources):
    with st.expander(f"📚 Sources found by vector search ({len(sources)})"):
        for i, s in enumerate(sources, 1):
            st.markdown(
                f"**[{i}] {s['name']}** · similarity `{s['score']:.3f}`  \n"
                f"{s['region']} · {s['category']} · seasons: {', '.join(s['best_seasons'])} · "
                f"budget: {s['budget']} · [source]({s['source']})"
            )


# Show the conversation so far.
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            render_sources(msg["sources"])

# Example questions for a quick demo.
if not st.session_state.messages:
    st.markdown("**Try an example question:**")
    cols = st.columns(len(EXAMPLES))
    for col, example in zip(cols, EXAMPLES):
        if col.button(example, use_container_width=True):
            st.session_state.pending = example
            st.rerun()

question = st.chat_input("Ask about places to visit in Kazakhstan...") or st.session_state.pop("pending", None)

if question:
    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Searching the knowledge base and asking the LLM..."):
            try:
                result = pipeline.answer(question, top_k=top_k, filters=filters,
                                         score_threshold=threshold or None, history=history)
            except Exception as exc:
                st.error(f"Error: {exc}")
                st.stop()
        st.markdown(result.answer)
        if result.sources:
            render_sources(result.sources)
        if show_debug and result.prompt_messages:
            with st.expander("🔍 Prompt sent to the LLM"):
                for m in result.prompt_messages:
                    st.markdown(f"**{m['role']}**")
                    st.code(m["content"], language="markdown")
    st.session_state.messages.append(
        {"role": "assistant", "content": result.answer, "sources": result.sources}
    )
