"""
The Engineering PDF AI Tutor
----------------------------
Upload technical PDFs -> chat, extract formulas, generate flashcards.
Stack: Streamlit + LangChain + FAISS + (OpenAI or Google Gemini)

Run:  streamlit run app.py
"""

import os
import tempfile

import pdfplumber
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()  # reads OPENAI_API_KEY / GOOGLE_API_KEY from a .env file

# ----------------------------------------------------------------------
# CONFIG  (change model names here if your provider updates them)
# ----------------------------------------------------------------------
CHUNK_SIZE = 1200      # characters per chunk (big, so equations stay with context)
CHUNK_OVERLAP = 250    # characters shared between neighbouring chunks
TOP_K = 5              # chunks retrieved for a normal question
TOP_K_ACTIONS = 8      # chunks retrieved for flashcards / formula summary

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

SYSTEM_PROMPT = """You are an expert engineering tutor (thermodynamics, aerodynamics,
AI, avionics and similar subjects). Follow these rules strictly:

1. Answer ONLY using the CONTEXT below, which comes from the user's uploaded document.
2. If the answer is not in the context, reply exactly: "I cannot find this in the document."
   Do not guess and do not use outside knowledge.
3. Format every equation in LaTeX: inline as $...$ and standalone as $$...$$.
   After an equation, list what each symbol means (with units, if the text gives them).
4. Explain step by step in a clear, student-friendly way.
5. Mention the source page like (p. 12) when you use a fact.

CONTEXT:
{context}
"""

HUMAN_PROMPT = """Recent chat history:
{history}

Task / question: {question}"""

FLASHCARD_TASK = (
    "Generate exactly 5 flashcards from the context. Format each as:\n"
    "**Card N**\n**Q:** ...\n**A:** ...\n"
    "Mix definitions, formula-recall and one short application question. "
    "Use LaTeX for equations. Use only the context."
)
FORMULA_TASK = (
    "List the key formulas found in the context. For each: the equation in LaTeX, "
    "what every symbol means, and when it is used. Use only formulas that appear in the context."
)
FORMULA_QUERY = "key formulas equations definitions"
FLASHCARD_QUERY = "important concepts definitions formulas principles"


# ----------------------------------------------------------------------
# 1. MODELS
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
# 2. PDF -> TEXT -> CHUNKS -> VECTOR STORE
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
        if not docs:  # pdfplumber found nothing -> try the other parser
            docs = PyPDFLoader(path).load()
            for d in docs:
                d.metadata["source"] = uploaded_file.name
                d.metadata["page"] = d.metadata.get("page", 0) + 1
    finally:
        os.remove(path)
    return docs


def split_documents(docs: list[Document]) -> list[Document]:
    """Split into chunks. Separators try paragraph breaks first, then lines,
    so an equation line is rarely cut in the middle."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(docs)


def build_vector_store(files, embeddings):
    """Full pipeline: extract -> split -> embed -> FAISS."""
    all_docs = []
    for f in files:
        all_docs.extend(extract_pdf(f))
    if not all_docs:
        raise ValueError(
            "No text found. The PDF may be scanned images (needs OCR)."
        )
    chunks = split_documents(all_docs)
    return FAISS.from_documents(chunks, embeddings), len(chunks)


# ----------------------------------------------------------------------
# 3. RAG
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
    return "\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in recent)


def ask(question: str, retrieval_query: str, k: int, task: str | None = None) -> str:
    """Retrieve chunks -> build prompt -> call LLM."""
    retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": k})
    docs = retriever.invoke(retrieval_query)

    prompt = ChatPromptTemplate.from_messages(
        [("system", SYSTEM_PROMPT), ("human", HUMAN_PROMPT)]
    )
    chain = prompt | st.session_state.llm | StrOutputParser()
    return chain.invoke(
        {
            "context": format_context(docs),
            "history": format_history(st.session_state.messages),
            "question": task or question,
        }
    )


# ----------------------------------------------------------------------
# 4. UI
# ----------------------------------------------------------------------
def init_state():
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("vector_store", None)
    st.session_state.setdefault("llm", None)
    st.session_state.setdefault("files_signature", None)


def sidebar():
    """Sidebar: provider, API key, file upload. Returns (provider, key, files)."""
    with st.sidebar:
        st.header("⚙️ Configuration")
        provider = st.selectbox("LLM provider", list(PROVIDERS))
        default_key = os.getenv(PROVIDERS[provider]["env"], "")
        api_key = st.text_input(
            "API key", value=default_key, type="password",
            help="Or put it in a .env file.",
        )
        files = st.file_uploader(
            "Upload PDF(s)", type=["pdf"], accept_multiple_files=True
        )
        if st.button("🗑️ Clear chat"):
            st.session_state.messages = []
            st.rerun()
    return provider, api_key, files


def process_files_if_new(provider, api_key, files):
    """Rebuild the vector store only when files/provider/key change."""
    if not files or not api_key:
        return
    signature = (provider, api_key[-6:], tuple((f.name, f.size) for f in files))
    if signature == st.session_state.files_signature:
        return

    with st.spinner("Reading PDFs, chunking and creating embeddings..."):
        try:
            embeddings, llm = get_models(provider, api_key)
            store, n_chunks = build_vector_store(files, embeddings)
        except Exception as e:
            st.sidebar.error(f"Processing failed: {e}")
            return
    st.session_state.vector_store = store
    st.session_state.llm = llm
    st.session_state.files_signature = signature
    st.session_state.messages = []
    st.sidebar.success(f"Ready! {len(files)} file(s), {n_chunks} chunks indexed.")


def run_and_store(user_text: str, retrieval_query: str, k: int, task: str | None = None):
    """Show the user message, get the answer, save both to history."""
    st.session_state.messages.append({"role": "user", "content": user_text})
    with st.chat_message("user"):
        st.markdown(user_text)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                answer = ask(user_text, retrieval_query, k, task)
            except Exception as e:
                answer = f"⚠️ Error: {e}"
        st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})


def main():
    st.set_page_config(page_title="Engineering PDF AI Tutor", page_icon="🎓", layout="wide")
    st.title("🎓 The Engineering PDF AI Tutor")
    st.caption("Upload lecture notes, then ask questions, pull out formulas, or make flashcards.")

    init_state()
    provider, api_key, files = sidebar()
    process_files_if_new(provider, api_key, files)

    if st.session_state.vector_store is None:
        st.info("👈 Add your API key and upload at least one PDF to begin.")
        return

    # Chat history
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    # Quick action buttons
    col1, col2, _ = st.columns([1, 1, 3])
    flash = col1.button("🃏 Generate 5 Flashcards", use_container_width=True)
    formulas = col2.button("📐 Summarize Key Formulas", use_container_width=True)

    # Chat input
    question = st.chat_input("Ask something about your document...")

    if flash:
        run_and_store("Generate 5 Flashcards", FLASHCARD_QUERY, TOP_K_ACTIONS, FLASHCARD_TASK)
    elif formulas:
        run_and_store("Summarize Key Formulas", FORMULA_QUERY, TOP_K_ACTIONS, FORMULA_TASK)
    elif question:
        run_and_store(question, question, TOP_K)


if __name__ == "__main__":
    main()
