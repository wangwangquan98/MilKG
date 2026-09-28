"""Read supported documents and chunk at document/paragraph boundaries."""

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Chunk:
    id: str
    source: str
    text: str


def read_document(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8-sig")
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise RuntimeError("PDF support requires: pip install '.[documents]'") from exc
        return "\n\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
    if suffix == ".docx":
        try:
            from docx import Document
        except ImportError as exc:
            raise RuntimeError("DOCX support requires: pip install '.[documents]'") from exc
        return "\n\n".join(p.text for p in Document(path).paragraphs if p.text.strip())
    raise ValueError(f"Unsupported document: {path}")


def preprocess_text(text: str) -> str:
    """Join OCR-split Chinese words while retaining paragraph boundaries."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00a0", " ").replace("\u3000", " ")
    paragraphs = []
    for part in re.split(r"\n[ \t]*\n+", text):
        line = re.sub(r"\s+", " ", part).strip()
        if not line:
            continue
        # OCR commonly inserts spaces inside Chinese words and at wrapped lines.
        # Keep spaces between Latin words and keep paragraph breaks for chunking.
        line = re.sub(r"(?<=[\u3400-\u9fff]) +(?=[\u3400-\u9fffA-Za-z0-9])", "", line)
        line = re.sub(r"(?<=[A-Za-z0-9]) +(?=[\u3400-\u9fff])", "", line)
        line = re.sub(r"(?<![A-Za-z])([A-Z]{1,4}) +(?=\d)", r"\1", line)
        line = re.sub(r"(?<=[。！？]) +(?=[\u3400-\u9fff])", "", line)
        paragraphs.append(line)
    return "\n\n".join(paragraphs)


def _split_long_paragraph(text: str, max_chars: int, overlap: int) -> list[str]:
    """Prefer sentence ends without exceeding the requested chunk length."""
    blocks = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            endings = [match.end() for match in re.finditer(r"[。！？!?；;]", text[start:end])
                       if match.end() >= int(max_chars * 0.65)]
            if endings:
                end = start + endings[-1]
        blocks.append(text[start:end].strip())
        if end == len(text):
            break
        start = max(start + 1, end - overlap)
    return blocks


def chunk_text(text: str, source: str, max_chars: int = 1800, overlap: int = 180) -> list[Chunk]:
    if max_chars < 100 or not 0 <= overlap < max_chars:
        raise ValueError("Require max_chars >= 100 and 0 <= overlap < max_chars")
    text = preprocess_text(text)
    # Keep short paragraphs together; split long OCR paragraphs near sentence
    # endings with overlap so cross-boundary relations remain visible.
    paragraphs = [re.sub(r"\s+", " ", part).strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        if len(para) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(_split_long_paragraph(para, max_chars, overlap))
            continue
        candidate = f"{current}\n\n{para}" if current else para
        if current and len(candidate) > max_chars:
            chunks.append(current)
            current = f"{current[-overlap:]}\n\n{para}" if overlap else para
            if len(current) > max_chars:
                current = para
        else:
            current = candidate
    if current:
        chunks.append(current)
    return [Chunk(f"{Path(source).stem}:{i:04d}", source, part) for i, part in enumerate(chunks)]


def load_chunks(paths: list[Path], max_chars: int = 1800, overlap: int = 180) -> list[Chunk]:
    files: list[Path] = []
    for path in paths:
        files.extend(sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in {".txt", ".md", ".pdf", ".docx"}) if path.is_dir() else [path])
    chunks: list[Chunk] = []
    for path in files:
        chunks.extend(chunk_text(read_document(path), str(path), max_chars, overlap))
    return chunks
