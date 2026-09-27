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


def chunk_text(text: str, source: str, max_chars: int = 1800, overlap: int = 180) -> list[Chunk]:
    if max_chars < 100 or not 0 <= overlap < max_chars:
        raise ValueError("Require max_chars >= 100 and 0 <= overlap < max_chars")
    # Keep section headings attached to their following paragraph; long blocks
    # are split with overlap so cross-boundary relations remain visible.
    paragraphs = [re.sub(r"\s+", " ", part).strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    blocks: list[str] = []
    for para in paragraphs:
        if len(para) <= max_chars:
            blocks.append(para)
        else:
            step = max_chars - overlap
            blocks.extend(para[i:i + max_chars] for i in range(0, len(para), step))
    chunks: list[str] = []
    current = ""
    for block in blocks:
        candidate = f"{current}\n\n{block}" if current else block
        if current and len(candidate) > max_chars:
            chunks.append(current)
            # A short context tail helps connect adjacent military sections.
            current = f"{current[-overlap:]}\n\n{block}" if overlap else block
            if len(current) > max_chars:
                current = block
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
