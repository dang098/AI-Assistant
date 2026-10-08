import uuid

import chromadb
import tiktoken
from chromadb.utils import embedding_functions
from flashrank import Ranker, RerankRequest
from google import genai
from tenacity import retry, stop_after_attempt, wait_exponential

from core.config import settings
from core.logger import get_logger

logger = get_logger(__name__)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=4),
    reraise=True,
)
def _call_gemini_with_retry(prompt: str) -> str:
    client = genai.Client(api_key=settings.gemini_api_key)
    response = client.models.generate_content(
        model=settings.gemini_model,
        contents=prompt,
    )
    return response.text.strip() if response.text else ""


def chunk_text(text: str) -> list[str]:
    encoder = tiktoken.get_encoding("cl100k_base")
    tokens = encoder.encode(text)

    chunks = []
    start = 0
    step = settings.chunk_size - settings.chunk_overlap

    while start < len(tokens):
        end = start + settings.chunk_size
        chunk_tokens = tokens[start:end]
        chunks.append(encoder.decode(chunk_tokens))
        start += step

    logger.info(f"Chunking: {len(tokens)} tokens -> {len(chunks)} chunks")
    return chunks


def get_chroma_collection() -> chromadb.Collection:
    client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
    embedding_fn = embedding_functions.DefaultEmbeddingFunction()

    return client.get_or_create_collection(
        name=settings.collection_name,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )


def ingest_document(file_content: str, filename: str) -> dict:
    if not file_content.strip():
        raise ValueError("File content is empty.")

    chunks = chunk_text(file_content)
    ids = [str(uuid.uuid4()) for _ in chunks]
    metadatas = [{"source": filename, "chunk_index": i} for i in range(len(chunks))]

    collection = get_chroma_collection()
    collection.upsert(documents=chunks, ids=ids, metadatas=metadatas)

    logger.info(
        f"Ingested '{filename}': {len(chunks)} chunks (Total in DB: {collection.count()})"
    )

    return {
        "status": "success",
        "filename": filename,
        "chunks_created": len(chunks),
        "total_chunks_in_db": collection.count(),
    }


def query_knowledge(question: str) -> dict:
    cleaned_question = question.strip()

    if not cleaned_question:
        raise ValueError("Question cannot be empty.")

    alphanumeric_chars = [c for c in cleaned_question if c.isalnum()]
    if len(alphanumeric_chars) < 3:
        return {
            "answer": "Câu hỏi quá ngắn hoặc chưa rõ nghĩa. Vui lòng hỏi rõ hơn về nội dung tài liệu.",
            "sources": [],
            "chunks_used": 0,
            "rerank_scores": [],
        }

    collection = get_chroma_collection()
    if collection.count() == 0:
        return {
            "answer": "Chưa có tài liệu trong hệ thống. Vui lòng tải lên tài liệu trước.",
            "sources": [],
            "chunks_used": 0,
            "rerank_scores": [],
        }

    n_results = min(settings.top_k_retrieval, collection.count())
    search_results = collection.query(
        query_texts=[cleaned_question], n_results=n_results
    )

    retrieved_docs = (
        search_results["documents"][0] if search_results["documents"] else []
    )
    retrieved_metas = (
        search_results["metadatas"][0] if search_results["metadatas"] else []
    )

    if not retrieved_docs:
        return {
            "answer": "Không tìm thấy thông tin phù hợp trong tài liệu.",
            "sources": [],
            "chunks_used": 0,
            "rerank_scores": [],
        }

    passages = [
        {"id": i, "text": doc, "meta": meta}
        for i, (doc, meta) in enumerate(zip(retrieved_docs, retrieved_metas))
    ]

    ranker = Ranker(
        model_name="ms-marco-MiniLM-L-12-v2", cache_dir="./chroma_db/flashrank_cache"
    )
    rerank_results = ranker.rerank(
        RerankRequest(query=cleaned_question, passages=passages)
    )

    top_results = rerank_results[: settings.top_k_rerank]
    final_docs = [item["text"] for item in top_results]
    final_metas = [item["meta"] for item in top_results]
    rerank_scores = [round(float(item["score"]), 4) for item in top_results]

    logger.info(
        f"Rerank: {len(retrieved_docs)} -> {len(final_docs)} chunks | Scores: {rerank_scores}"
    )

    context_text = "\n\n---\n\n".join(final_docs)
    sources = list({meta.get("source", "Không rõ") for meta in final_metas})

    prompt = f"""Bạn là trợ lý ảo trả lời dựa trên tài liệu nội bộ được cung cấp.

TÀI LIỆU:
====================
{context_text}
====================

CÂU HỎI: {cleaned_question}

QUY TẮC:
1. Chỉ trả lời dựa trên thông tin trong TÀI LIỆU ở trên.
2. Trả lời rõ ràng, ngắn gọn bằng tiếng Việt.
3. Nếu tài liệu không đề cập, hãy trả lời trung thực: "Tài liệu nội bộ không có thông tin về vấn đề này." Tuyệt đối không tự suy diễn.
"""

    answer = _call_gemini_with_retry(prompt)

    return {
        "answer": answer if answer else "Không nhận được phản hồi từ mô hình AI.",
        "sources": sources,
        "chunks_used": len(final_docs),
        "rerank_scores": rerank_scores,
        "contexts": final_docs,
    }
