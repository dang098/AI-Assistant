import io

from fastapi import APIRouter, File, HTTPException, UploadFile
from pypdf import PdfReader

from core.logger import get_logger
from services.rag_service import ingest_document

logger = get_logger(__name__)
router = APIRouter()

MAX_FILE_SIZE = 20 * 1024 * 1024
ALLOWED_EXTENSIONS = (".txt", ".pdf")


@router.post("/upload", summary="Tải lên tài liệu")
async def upload_document(file: UploadFile = File(...)):
    filename = file.filename or ""

    if not any(filename.lower().endswith(ext) for ext in ALLOWED_EXTENSIONS):
        raise HTTPException(
            status_code=400,
            detail=f"Định dạng không hỗ trợ. Chỉ chấp nhận {', '.join(ALLOWED_EXTENSIONS)}.",
        )

    if file.size and file.size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400, detail="Dung lượng tập tin vượt quá giới hạn 20MB."
        )

    file_bytes = await file.read()

    if len(file_bytes) == 0:
        raise HTTPException(status_code=400, detail="Tập tin rỗng (0 bytes).")

    if len(file_bytes) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400, detail="Dung lượng tập tin vượt quá giới hạn 20MB."
        )

    if filename.lower().endswith(".txt"):
        try:
            content = file_bytes.decode("utf-8")
        except UnicodeDecodeError:
            content = file_bytes.decode("latin-1")
    else:
        pdf_reader = PdfReader(io.BytesIO(file_bytes))
        pages_text = [page.extract_text() or "" for page in pdf_reader.pages]
        content = "\n\n".join(pages_text)

        if not content.strip():
            raise HTTPException(
                status_code=400,
                detail="Không đọc được nội dung văn bản từ tập tin PDF.",
            )

    result = ingest_document(file_content=content, filename=filename)

    return {
        "status": "success",
        "filename": filename,
        "chunks_created": result["chunks_created"],
        "total_chunks_in_db": result["total_chunks_in_db"],
    }
