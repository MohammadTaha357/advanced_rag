import math

import re
import shutil
from collections import Counter
from dataclasses import dataclass
from io import BytesIO

import cohere
import pymupdf
import pytesseract
import streamlit as st
from groq import Groq
from PIL import Image


APP_TITLE = "Advanced PDF RAG Chatbot"


@dataclass
class Chunk:
    id: str
    source: str
    page: int
    text: str


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


def chunk_text(text: str, source: str, page: int, chunk_size: int, overlap: int) -> list[Chunk]:
    words = text.split()
    if not words:
        return []

    chunks = []
    step = max(1, chunk_size - overlap)
    for idx, start in enumerate(range(0, len(words), step), start=1):
        piece = " ".join(words[start : start + chunk_size]).strip()
        if len(piece) < 40:
            continue
        chunks.append(Chunk(f"{source}-p{page}-c{idx}", source, page, piece))
    return chunks


def ocr_page(page: pymupdf.Page, zoom: float) -> str:
    matrix = pymupdf.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=matrix, alpha=False)
    image = Image.open(BytesIO(pix.tobytes("png")))
    return pytesseract.image_to_string(image, config="--psm 6").strip()


def extract_pdf_chunks(
    uploaded_file,
    chunk_size: int,
    overlap: int,
    ocr_zoom: float,
    force_ocr: bool,
    min_native_chars: int,
) -> tuple[list[Chunk], list[str]]:
    chunks: list[Chunk] = []
    warnings: list[str] = []
    pdf_bytes = uploaded_file.getvalue()

    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
        for page_index, page in enumerate(doc, start=1):
            native_text = page.get_text("text").strip()
            should_ocr = force_ocr or len(native_text) < min_native_chars
            page_text = native_text

            if should_ocr:
                try:
                    ocr_text = ocr_page(page, ocr_zoom)
                    if len(ocr_text) > len(native_text):
                        page_text = ocr_text
                except pytesseract.TesseractNotFoundError:
                    warnings.append(
                        "Tesseract OCR is not installed or not on PATH, so image-only pages could not be OCR'd."
                    )
                except Exception as exc:
                    warnings.append(f"OCR failed for {uploaded_file.name}, page {page_index}: {exc}")

            chunks.extend(chunk_text(page_text, uploaded_file.name, page_index, chunk_size, overlap))

    return chunks, warnings


class BM25Index:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.doc_tokens = [tokenize(chunk.text) for chunk in chunks]
        self.doc_lengths = [len(tokens) for tokens in self.doc_tokens]
        self.avg_doc_length = sum(self.doc_lengths) / max(1, len(self.doc_lengths))
        self.term_freqs = [Counter(tokens) for tokens in self.doc_tokens]
        self.doc_freq: Counter[str] = Counter()

        for tokens in self.doc_tokens:
            self.doc_freq.update(set(tokens))

    def search(self, query: str, limit: int) -> list[tuple[Chunk, float]]:
        query_terms = tokenize(query)
        total_docs = max(1, len(self.chunks))
        scored: list[tuple[Chunk, float]] = []

        for chunk, freqs, doc_len in zip(self.chunks, self.term_freqs, self.doc_lengths):
            score = 0.0
            for term in query_terms:
                if term not in freqs:
                    continue
                df = self.doc_freq[term]
                idf = math.log(1 + (total_docs - df + 0.5) / (df + 0.5))
                tf = freqs[term]
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / max(1, self.avg_doc_length))
                score += idf * (tf * (self.k1 + 1)) / denominator

            if score > 0:
                scored.append((chunk, score))

        return sorted(scored, key=lambda item: item[1], reverse=True)[:limit]


def rerank_with_cohere(api_key: str, query: str, candidates: list[tuple[Chunk, float]], top_n: int):
    if not api_key or not candidates:
        return candidates[:top_n]

    client = cohere.Client(api_key)
    docs = [chunk.text for chunk, _ in candidates]
    response = client.rerank(
        model="rerank-english-v3.0",
        query=query,
        documents=docs,
        top_n=min(top_n, len(docs)),
    )

    reranked = []
    for item in response.results:
        chunk, bm25_score = candidates[item.index]
        reranked.append((chunk, float(item.relevance_score), bm25_score))
    return reranked


def build_context(results) -> str:
    context_parts = []
    for rank, result in enumerate(results, start=1):
        chunk = result[0]
        context_parts.append(
            f"[Context {rank}] Source: {chunk.source}, page {chunk.page}, chunk {chunk.id}\n{chunk.text}"
        )
    return "\n\n".join(context_parts)


def answer_with_groq(
    api_key: str,
    model: str,
    question: str,
    context: str,
    temperature: float,
    chat_history: list[dict],
) -> str:
    client = Groq(api_key=api_key)
    messages = [
        {
            "role": "system",
            "content": (
                "You are a precise RAG assistant. Answer only from the provided context. "
                "Cite sources inline as [filename, page N]. If the context is insufficient, say what is missing."
            ),
        }
    ]
    messages.extend(chat_history[-6:])
    messages.append(
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {question}",
        }
    )

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=1200,
    )
    return response.choices[0].message.content


def show_retrieval(results):
    if not results:
        st.info("No retrieval has run yet.")
        return

    for rank, result in enumerate(results, start=1):
        chunk = result[0]
        rerank_score = result[1]
        bm25_score = result[2] if len(result) > 2 else result[1]
        with st.expander(f"{rank}. {chunk.source} · page {chunk.page} · score {rerank_score:.3f}"):
            st.caption(f"Chunk ID: {chunk.id} · BM25: {bm25_score:.3f}")
            st.write(chunk.text)


def main():
    st.set_page_config(page_title=APP_TITLE, page_icon="📚", layout="wide")
    st.title(APP_TITLE)

    with st.sidebar:
        st.header("API Keys")
        groq_api_key = st.text_input("Groq API key", type="password")
        cohere_api_key = st.text_input("Cohere API key", type="password")

        st.header("Model")
        groq_model = "openai/gpt-oss-120b"
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.05)

        st.header("Retrieval")
        retrieval_k = st.slider("BM25 candidates", 5, 30, 12)
        rerank_top_n = st.slider("Final retrieved chunks", 2, 10, 5)
        chunk_size = st.slider("Chunk size", 180, 900, 420, step=20)
        overlap = st.slider("Chunk overlap", 0, 250, 80, step=10)

        st.header("OCR")
        force_ocr = st.checkbox("Force OCR on every page", value=False)
        min_native_chars = st.slider("OCR pages below native text chars", 0, 500, 80, step=10)
        ocr_zoom = st.slider("OCR render zoom", 1.5, 4.0, 2.5, step=0.25)
        if shutil.which("tesseract"):
            st.success("Tesseract OCR detected.")
        else:
            st.warning("Tesseract executable not found on PATH.")

        if st.button("Clear chat and index", use_container_width=True):
            st.session_state.clear()
            st.rerun()

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chunks" not in st.session_state:
        st.session_state.chunks = []
    if "last_retrieval" not in st.session_state:
        st.session_state.last_retrieval = []

    left, right = st.columns([0.58, 0.42], gap="large")

    with left:
        uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

        if uploaded_files and st.button("Process PDFs", type="primary"):
            with st.spinner("Extracting text and OCR'ing image-only pages..."):
                all_chunks: list[Chunk] = []
                all_warnings: list[str] = []
                for uploaded_file in uploaded_files:
                    file_chunks, file_warnings = extract_pdf_chunks(
                        uploaded_file,
                        chunk_size=chunk_size,
                        overlap=overlap,
                        ocr_zoom=ocr_zoom,
                        force_ocr=force_ocr,
                        min_native_chars=min_native_chars,
                    )
                    all_chunks.extend(file_chunks)
                    all_warnings.extend(file_warnings)

                st.session_state.chunks = all_chunks
                st.session_state.last_retrieval = []

            if all_chunks:
                st.success(f"Indexed {len(all_chunks)} chunks from {len(uploaded_files)} PDF file(s).")
            else:
                st.error("No text was extracted. Try enabling Force OCR or check that Tesseract is installed.")

            for warning in sorted(set(all_warnings)):
                st.warning(warning)

        st.subheader("Chat")
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        question = st.chat_input("Ask a question about your PDFs")
        if question:
            st.session_state.messages.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)

            if not groq_api_key:
                st.error("Add your Groq API key in the sidebar.")
                return
            if not st.session_state.chunks:
                st.error("Upload and process at least one PDF first.")
                return

            with st.spinner("Retrieving, reranking, and asking Groq..."):
                bm25 = BM25Index(st.session_state.chunks)
                candidates = bm25.search(question, retrieval_k)
                if cohere_api_key:
                    retrieved = rerank_with_cohere(cohere_api_key, question, candidates, rerank_top_n)
                else:
                    retrieved = candidates[:rerank_top_n]

                st.session_state.last_retrieval = retrieved
                context = build_context(retrieved)

                st.write("DEBUG MODEL:", groq_model)
                st.write("DEBUG KEY:", bool(groq_api_key))
                answer = answer_with_groq(
                    groq_api_key,
                    groq_model,
                    question,
                    context,
                    temperature,
                    st.session_state.messages[:-1],
                )

            st.session_state.messages.append({"role": "assistant", "content": answer})
            with st.chat_message("assistant"):
                st.markdown(answer)

    with right:
        st.subheader("Index")
        st.metric("Chunks", len(st.session_state.chunks))
        if st.session_state.chunks:
            sources = sorted({chunk.source for chunk in st.session_state.chunks})
            st.caption(", ".join(sources))

        st.subheader("Last Retrieval")
        show_retrieval(st.session_state.last_retrieval)


if __name__ == "__main__":
    main()
