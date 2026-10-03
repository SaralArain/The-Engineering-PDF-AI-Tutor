""" The Engineering PDF AI Tutor (v2) ---------------------------------- Upload technical PDFs -> chat, formulas, flashcards, quizzes, summaries. Stack: Streamlit + LangChain + FAISS + (OpenAI or Google Gemini) Optional pictures (put them in an assets/ folder next to app.py): assets/background.jpg -> full-page background assets/logo.png -> faint watermark, bottom-right Run: streamlit run app.py """

import base64
import os
import tempfile
from pathlib import Path

import pdfplumber
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
ASSETS = Path(__file__).parent / "assets"
ANSWER_MARK = "<<<ANSWERS>>>"   # LLM puts this line before quiz answers

PROVIDERS = {
    "OpenAI": {
        "env": "OPENAI_API_KEY",
        "chat_model": "gpt-4o-mini",
        "embed_model": "text-embedding-3-small",
    },
    "Google Gemini": {
        "env": "GOOGLE_API_KEY",
        "chat_model": "gemini-2.0-flash",
        "embed_model": "models/gemini-embedding-001",
    },
}

# Study modes change HOW the tutor explains (grounding rules never change)
STUDY_MODES = {
    "🎓 Tutor (step by step)": "Explain step by step in a clear, student-friendly way.",
    "🧒 Simple (beginner)": "Use very simple language and everyday analogies, but only for ideas that appear in the context.",
    "📝 Exam prep (concise)": "Be concise: key points, formulas, and common exam mistakes, based only on the context.",
    "🔬 Deep dive (detailed)": "Give a detailed, derivation-style explanation using the context.",
}

SYSTEM_PROMPT = """You are an expert engineering tutor (thermodynamics, aerodynamics, AI, avionics and similar subjects). Follow these rules strictly: 1. Answer ONLY using the CONTEXT below, which comes from the user's uploaded document. 2. If the answer is not in the context, reply exactly: "I cannot find this in the document." Do not guess and do not use outside knowledge. 3. Format every equation in LaTeX: inline as $...$ and standalone as $$...$$. After an equation, list what each symbol means (with units, if the text gives them). 4. {style} 5. Mention the source page like (p. 12) when you use a fact. CONTEXT: {context} """

HUMAN_PROMPT = """Recent chat history: {history} Task / question: {question}"""

# Quick actions: (button label, chat label, retrieval query, task prompt)
ACTIONS = {
    "flash": (
        "🃏 Flashcards",
        "Generate 5 Flashcards",
        "important concepts definitions formulas principles",
        "Generate exactly 5 flashcards from the context. Format each as:\n"
        "**Card N**\n**Q:** ...\n**A:** ...\n"
        "Mix definitions, formula-recall and one short application question. "
        "Use LaTeX for equations. Use only the context.",
    ),
    "formulas": (
        "📐 Key Formulas",
        "Summarize Key Formulas",
        "key formulas equations definitions",
        "List the key formulas found in the context. For each: the equation in LaTeX, "
        "what every symbol means, and when it is used. Use only formulas in the context.",
    ),
    "quiz": (
        "🧪 Practice Quiz",
        "Create a 5-question practice quiz",
        "important concepts calculations examples",
        "Create 5 multiple-choice questions (options A-D) from the context. "
        "Write all questions and options first. Then output one line containing exactly "
        f"{ANSWER_MARK} followed by the answer key with a one-line explanation per question.",
    ),
    "summary": (
        "📄 Study Summary",
        "Give me a structured study summary",
        "main topics overview introduction conclusion summary",
        "Write a structured study summary of the main topics in the context: use headings, "
        "short bullet points, and put key equations in LaTeX. Use only the context.",
    ),
}


# ----------------------------------------------------------------------
# 1. LOOK & FEEL
# ----------------------------------------------------------------------
def load_image_b64(names):
    """Return (mime, base64) for the first existing image in assets/, else None."""
    for name in names:
        path = ASSETS / name
        if path.exists():
            ext = path.suffix.lower().strip(".")
            mime = "image/png" if ext == "png" else "image/jpeg"
            return mime, base64.b64encode(path.read_bytes()).decode()
    return None


CSS = """ <style> .stApp { background: __BACKGROUND__; background-size: cover; background-attachment: fixed; } /* faint watermark logo, bottom-right */ .stApp::after { content: ""; __WATERMARK__ position: fixed; right: 28px; bottom: 90px; width: 240px; height: 240px; background-repeat: no-repeat; background-size: contain; background-position: center; opacity: 0.07; pointer-events: none; z-index: 0; } [data-testid="stHeader"] { background: transparent; } [data-testid="stBottom"] > div { background: transparent; } [data-testid="stSidebar"] { background: rgba(10, 16, 36, 0.88); border-right: 1px solid rgba(56, 189, 248, 0.25); } .hero { padding: 1.2rem 1.6rem; border-radius: 16px; margin-bottom: 1rem; background: linear-gradient(135deg, rgba(56,189,248,0.18), rgba(168,85,247,0.18)); border: 1px solid rgba(255,255,255,0.12); } .hero h1 { margin: 0; font-size: 2rem; background: linear-gradient(90deg, #38bdf8, #a78bfa); -webkit-background-clip: text; -webkit-text-fill-color: transparent; } .hero p { margin: 0.3rem 0 0 0; color: #cbd5e1; } .stat-row { display: flex; gap: 0.7rem; margin: 0.4rem 0 1rem 0; } .stat { flex: 1; text-align: center; padding: 0.6rem; border-radius: 12px; background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.1); } .stat b { display: block; font-size: 1.4rem; color: #38bdf8; } .stat span { font-size: 0.75rem; color: #94a3b8; } [data-testid="stChatMessage"] { background: rgba(15, 23, 42, 0.72); border: 1px solid rgba(255,255,255,0.08); border-radius: 14px; backdrop-filter: blur(6px); } .stButton > button { width: 100%; border-radius: 12px; font-weight: 600; border: 1px solid rgba(56,189,248,0.45); background: rgba(56,189,248,0.10); transition: all 0.2s; } .stButton > button:hover { background: rgba(56,189,248,0.30); border-color: #38bdf8; transform: translateY(-2px); } </style> """


def inject_css():
    bg = load_image_b64(["background.jpg", "background.jpeg", "background.png"])
    logo = load_image_b64(["logo.png", "logo.jpg", "logo.jpeg"])

    if bg:  # your picture, darkened so text stays readable
        background = (
            "linear-gradient(rgba(8,12,28,0.82), rgba(8,12,28,0.92)), "
            f'url("data:{bg[0]};base64,{bg[1]}")'
        )
    else:   # built-in fallback: blueprint grid + glow
        background = (
            "radial-gradient(circle at 15% 10%, rgba(56,189,248,0.18), transparent 40%), "
            "radial-gradient(circle at 85% 90%, rgba(168,85,247,0.18), transparent 40%), "
            "repeating-linear-gradient(0deg, rgba(255,255,255,0.03) 0 1px, transparent 1px 40px), "
            "repeating-linear-gradient(90deg, rgba(255,255,255,0.03) 0 1px, transparent 1px 40px), "
            "linear-gradient(135deg, #0b1020, #111a35)"
        )
    watermark = (
        f'background-image: url("data:{logo[0]};base64,{logo[1]}");' if logo else ""
    )
    st.markdown(
        CSS.replace("__BACKGROUND__", background).replace("__WATERMARK__", watermark),
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------------------
# 2. MODELS
# ----------------------------------------------------------------------
def get_models(provider: str, api_key: str):
    """Return (embeddings, llm) for the chosen provider."""
    cfg = PROVIDERS[provider]
    if provider == "OpenAI":
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings

        embeddings = OpenAIEmbeddings(model=cfg["embed_model"], api_key=api_key)
        llm = ChatOpenAI(model=cfg["chat_model"], temperature=0.2, api_key=api_key)
    else:
        from langchain_google_genai import (
            ChatGoogleGenerativeAI,
            GoogleGenerativeAIEmbeddings,
        )

        embeddings = GoogleGenerativeAIEmbeddings(
            model=cfg["embed_model"], google_api_key=api_key
        )
        llm = ChatGoogleGenerativeAI(
            model=cfg["chat_model"], temperature=0.2, google_api_key=api_key
        )
    return embeddings, llm


# ----------------------------------------------------------------------
# 3. PDF -> TEXT -> CHUNKS -> VECTOR STORE
# ----------------------------------------------------------------------
def extract_pdf(uploaded_file) -> list[Document]:
    """Extract page-wise text. pdfplumber first, PyPDFLoader as fallback."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getvalue())
        path = tmp.name

    docs = []
    try:
        with pdfplumber.open(path) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                if text.strip():
                    docs.append(
                        Document(
                            page_content=text,
                            metadata={"source": uploaded_file.name, "page": i},
                        )
                    )
        if not docs:
            docs = PyPDFLoader(path).load()
            for d in docs:
                d.metadata["source"] = uploaded_file.name
                d.metadata["page"] = d.metadata.get("page", 0) + 1
    finally:
        os.remove(path)
    return docs


def build_vector_store(files, embeddings, chunk_size, chunk_overlap):
    """extract -> split -> embed -> FAISS. Returns (store, n_pages, n_chunks)."""
    all_docs = []
    for f in files:
        all_docs.extend(extract_pdf(f))
    if not all_docs:
        raise ValueError("No text found. The PDF may be scanned images (needs OCR).")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],  # paragraph > line > sentence
    )
    chunks = splitter.split_documents(all_docs)
    return FAISS.from_documents(chunks, embeddings), len(all_docs), len(chunks)


# ----------------------------------------------------------------------
# 4. RAG
# ----------------------------------------------------------------------
def format_context(docs: list[Document]) -> str:
    return "\n\n---\n\n".join(
        f"[{d.metadata.get('source')} | p. {d.metadata.get('page')}]\n{d.page_content}"
        for d in docs
    )


def format_history(messages: list[dict], max_turns: int = 6) -> str:
    recent = messages[-max_turns:]
    if not recent:
        return "(none)"
    # keep history short: only the first 600 characters of each message
    return "\n".join(f"{m['role'].capitalize()}: {m['content'][:600]}" for m in recent)


def ask(question, retrieval_query, k, style, task=None):
    """Retrieve chunks -> prompt -> LLM. Returns (answer, sources)."""
    retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": k})
    docs = retriever.invoke(retrieval_query)

    prompt = ChatPromptTemplate.from_messages(
        [("system", SYSTEM_PROMPT), ("human", HUMAN_PROMPT)]
    )
    chain = prompt | st.session_state.llm | StrOutputParser()
    answer = chain.invoke(
        {
            "context": format_context(docs),
            "history": format_history(st.session_state.messages),
            "question": task or question,
            "style": style,
        }
    )
    sources = [
        {
            "source": d.metadata.get("source"),
            "page": d.metadata.get("page"),
            "snippet": d.page_content[:350],
        }
        for d in docs
    ]
    return answer, sources


# ----------------------------------------------------------------------
# 5. UI
# ----------------------------------------------------------------------
def init_state():
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("vector_store", None)
    st.session_state.setdefault("llm", None)
    st.session_state.setdefault("signature", None)
    st.session_state.setdefault("stats", None)


def chat_as_markdown() -> str:
    lines = ["# Engineering PDF AI Tutor - Chat Export\n"]
    for m in st.session_state.messages:
        who = "You" if m["role"] == "user" else "Tutor"
        lines.append(f"## {who}\n\n{m['content'].replace(ANSWER_MARK, '---')}\n")
    return "\n".join(lines)


def sidebar():
    """Returns (provider, api_key, files, style, settings dict)."""
    with st.sidebar:
        st.markdown("### ⚙️ Setup")
        provider = st.selectbox("LLM provider", list(PROVIDERS))
        api_key = st.text_input(
            "API key",
            value=os.getenv(PROVIDERS[provider]["env"], ""),
            type="password",
            help="Or put it in a .env file.",
        )
        files = st.file_uploader("Upload PDF(s)", type=["pdf"], accept_multiple_files=True)

        st.markdown("### 🎛️ Study mode")
        mode = st.radio("Explanation style", list(STUDY_MODES), label_visibility="collapsed")

        with st.expander("🔧 Advanced (chunking & retrieval)"):
            chunk_size = st.slider("Chunk size (characters)", 600, 2000, 1200, 100)
            overlap = st.slider("Chunk overlap", 0, 500, 250, 50)
            top_k = st.slider("Chunks retrieved per question", 3, 10, 5)

        st.markdown("### 💾 Session")
        if st.session_state.messages:
            st.download_button(
                "⬇️ Download chat (.md)",
                chat_as_markdown(),
                file_name="tutor_chat.md",
                mime="text/markdown",
            )
        if st.button("🗑️ Clear chat"):
            st.session_state.messages = []
            st.rerun()

    settings = {"chunk_size": chunk_size, "overlap": min(overlap, chunk_size - 100), "top_k": top_k}
    return provider, api_key, files, STUDY_MODES[mode], settings


def process_files_if_new(provider, api_key, files, settings):
    """Rebuild the vector store only when files / key / chunk settings change."""
    if not files or not api_key:
        return
    signature = (
        provider, api_key[-6:], tuple((f.name, f.size) for f in files),
        settings["chunk_size"], settings["overlap"],
    )
    if signature == st.session_state.signature:
        return

    with st.spinner("📖 Reading PDFs, chunking and creating embeddings..."):
        try:
            embeddings, llm = get_models(provider, api_key)
            store, n_pages, n_chunks = build_vector_store(
                files, embeddings, settings["chunk_size"], settings["overlap"]
            )
        except Exception as e:
            st.sidebar.error(f"Processing failed: {e}")
            return
    st.session_state.vector_store = store
    st.session_state.llm = llm
    st.session_state.signature = signature
    st.session_state.stats = (len(files), n_pages, n_chunks)
    st.session_state.messages = []


def render_message(m):
    """Show a message; quiz answers and sources go into expanders."""
    parts = m["content"].split(ANSWER_MARK)
    st.markdown(parts[0])
    if len(parts) > 1:
        with st.expander("✅ Show answer key"):
            st.markdown(parts[1])
    if m.get("sources"):
        with st.expander("📚 Sources used"):
            for s in m["sources"]:
                st.markdown(f"**{s['source']} — page {s['page']}**")
                st.caption(s["snippet"] + "...")


def run_and_store(user_text, retrieval_query, k, style, task=None):
    st.session_state.messages.append({"role": "user", "content": user_text})
    with st.chat_message("user"):
        st.markdown(user_text)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                answer, sources = ask(user_text, retrieval_query, k, style, task)
            except Exception as e:
                answer, sources = f"⚠️ Error: {e}", []
        msg = {"role": "assistant", "content": answer, "sources": sources}
        render_message(msg)
    st.session_state.messages.append(msg)


def show_hero_and_stats():
    st.markdown(
        '<div class="hero"><h1>🎓 The Engineering PDF AI Tutor</h1>'
        "<p>Upload lecture notes — ask questions, extract formulas, "
        "make flashcards and quizzes. Answers come only from your document.</p></div>",
        unsafe_allow_html=True,
    )
    if st.session_state.stats:
        files, pages, chunks = st.session_state.stats
        st.markdown(
            f'<div class="stat-row">'
            f'<div class="stat"><b>{files}</b><span>PDF files</span></div>'
            f'<div class="stat"><b>{pages}</b><span>Pages read</span></div>'
            f'<div class="stat"><b>{chunks}</b><span>Chunks indexed</span></div>'
            f"</div>",
            unsafe_allow_html=True,
        )


def main():
    st.set_page_config(page_title="Engineering PDF AI Tutor", page_icon="🎓", layout="wide")
    init_state()
    inject_css()

    provider, api_key, files, style, settings = sidebar()
    process_files_if_new(provider, api_key, files, settings)
    show_hero_and_stats()

    if st.session_state.vector_store is None:
        st.info("👈 Add your API key and upload at least one PDF to begin.")
        return

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            render_message(m)

    # Quick action buttons
    clicked = None
    cols = st.columns(len(ACTIONS))
    for col, (key, action) in zip(cols, ACTIONS.items()):
        if col.button(action[0], key=f"btn_{key}"):
            clicked = key

    question = st.chat_input("Ask something about your document...")

    if clicked:
        _, label, query, task = ACTIONS[clicked]
        run_and_store(label, query, settings["top_k"] + 3, style, task)
    elif question:
        run_and_store(question, question, settings["top_k"], style)


if __name__ == "__main__":
    main()
