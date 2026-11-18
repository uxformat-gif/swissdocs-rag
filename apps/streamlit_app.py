"""SwissDocs demo UI. Talks only to the FastAPI service."""

import html
import os
import time

import httpx
import streamlit as st

API_URL = os.getenv("SWISSDOCS_API_URL", "http://localhost:8000").rstrip("/")

EXAMPLES = {
    "fr": (
        "Quelles prestations l'assurance obligatoire des soins "
        "prend-elle en charge en cas de maladie ?"
    ),
    "de": "Wie beteiligen sich die Versicherten an den Kosten der Leistungen?",
    "en": "What does the insured person have to disclose before concluding the contract?",
}

st.set_page_config(page_title="SwissDocs", page_icon="▣", layout="wide")
st.markdown(
    """
    <style>
      .stApp { background: #f4f5f7; }
      .block-container { padding-top: 2rem; max-width: 1080px; }
      .sd-mark { display:flex; gap:0.7rem; align-items:center; }
      .sd-square { width:1.1rem; height:1.1rem; background:#c8102e; display:inline-block; }
      .sd-title { font-size:1.8rem; font-weight:650; letter-spacing:-0.03em;
                  margin:0; color:#1c1c1c; }
      .sd-sub { color:#4b5563; margin:0.2rem 0 0.8rem 0; }
      .sd-card { background:white; border:1px solid #e5e7eb; border-radius:12px;
                 padding:1rem 1.1rem; }
      .sd-warn { background:#fff7ed; border:1px solid #fdba74; border-radius:12px;
                 padding:0.8rem 1rem; }
      .sd-src { background:white; border:1px solid #e5e7eb; border-left:3px solid #c8102e;
                border-radius:10px; padding:0.75rem 0.9rem; margin-bottom:0.6rem; }
      .sd-meta { color:#6b7280; font-size:0.85rem; }
      a { color:#9f1239; }
    </style>
    """,
    unsafe_allow_html=True,
)


def _get(path: str, timeout: float = 5.0) -> httpx.Response | None:
    try:
        return httpx.get(f"{API_URL}{path}", timeout=timeout)
    except httpx.HTTPError:
        return None


def wait_for_api() -> dict[str, object] | None:
    placeholder = st.empty()
    for attempt in range(1, 201):
        response = _get("/health", timeout=3)
        if response is not None and response.status_code == 200:
            placeholder.empty()
            return response.json()
        placeholder.info(
            "The API is still starting. The first launch downloads the multilingual "
            f"embedding model and indexes the corpus. Attempt {attempt}."
        )
        time.sleep(3)
    placeholder.error(f"Could not reach the API at {API_URL}.")
    return None


st.markdown(
    """
    <div class="sd-mark">
      <span class="sd-square"></span>
      <p class="sd-title">SwissDocs</p>
    </div>
    <p class="sd-sub">Question answering over Swiss insurance and regulatory texts,
    with the article, the page and the Fedlex source.</p>
    """,
    unsafe_allow_html=True,
)

health = wait_for_api()
if health is None:
    st.stop()

with st.sidebar:
    st.header("Corpus")
    st.write(f"Chunks: **{health.get('chunks', 0)}**")
    st.write("Languages: " + ", ".join(health.get("languages", [])))
    st.write("SR: " + ", ".join(health.get("laws", [])))
    st.write(f"Embeddings: `{health.get('embedding_model')}`")
    st.write(f"Answer backend: `{health.get('llm_backend')}`")
    st.caption(
        "Default deployment keeps documents and questions on this machine "
        "(Ollama). The compose demo uses extractive quotes so it runs without "
        "a model download beyond the embedder. Set LLM_BACKEND=ollama to generate."
    )

language = st.radio(
    "Answer language",
    options=["auto", "fr", "de", "en"],
    horizontal=True,
    format_func=lambda code: {
        "auto": "Detect",
        "fr": "Français",
        "de": "Deutsch",
        "en": "English",
    }[code],
)
if st.button("Use an example"):
    key = "fr" if language == "auto" else language
    st.session_state["question"] = EXAMPLES[key]

question = st.text_area(
    "Question",
    key="question",
    height=110,
    placeholder="Ask about compulsory health insurance or the insurance contract…",
)
cross = st.checkbox("Also search the other languages", value=False)
submitted = st.button("Ask", type="primary")

if submitted:
    if not question or len(question.strip()) < 3:
        st.warning("Write a slightly longer question.")
        st.stop()
    payload = {
        "question": question.strip(),
        "cross_lingual": cross,
    }
    if language != "auto":
        payload["language"] = language
    with st.spinner("Searching the consolidations…"):
        try:
            response = httpx.post(f"{API_URL}/ask", json=payload, timeout=180)
        except httpx.HTTPError as exc:
            st.error(f"The API request failed: {exc}")
            st.stop()
    if response.status_code != 200:
        st.error(response.text)
        st.stop()
    data = response.json()
    answer_html = html.escape(data["answer"]).replace("\n", "<br>")
    if data["insufficient_context"]:
        st.markdown(
            f'<div class="sd-warn"><strong>Insufficient context.</strong><br>{answer_html}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(f'<div class="sd-card">{answer_html}</div>', unsafe_allow_html=True)
    st.caption(f"Language {data['language']} · backend `{data['backend']}`")
    st.subheader("Sources")
    if not data["citations"]:
        st.write("No source was attached.")
    for citation in data["citations"]:
        page = f", p. {citation['page']}" if citation["page"] else ""
        binding = "" if citation["binding"] else " · non-binding translation"
        path = " · ".join(citation.get("heading_path") or [])
        st.markdown(
            f"""
            <div class="sd-src">
              <strong>[{citation["n"]}] {html.escape(str(citation["law_abbr"]))}
              art. {html.escape(str(citation["article"]))}</strong>
              {html.escape(str(citation["article_title"]))}<br>
              <span class="sd-meta">SR {html.escape(str(citation["law_sr"]))}{page}
              · version {html.escape(str(citation["version_date"]))}
              {binding}</span><br>
              <span class="sd-meta">{html.escape(path)}</span>
              <p>{html.escape(str(citation["snippet"]))}</p>
              <a href="{html.escape(str(citation["source_url"]))}" target="_blank">
                Open on Fedlex
              </a>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.caption(data["disclaimer"])
