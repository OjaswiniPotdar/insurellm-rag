from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator

import nltk
import tiktoken
from docling.document_converter import DocumentConverter
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download("punkt", quiet=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)



_embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
_encoding = tiktoken.get_encoding("cl100k_base")


@dataclass
class ChunkConfig:
    similarity_threshold: float = 0.82
    min_chunk_tokens: int = 50
    max_chunk_tokens: int = 512

    recursive_size: int = 512
    recursive_overlap: int = 64

    context_window: int = 1



@dataclass
class Chunk:
    text: str
    chunk_id: str
    source: str
    page: int = 0
    section: str = ""
    token_count: int = 0
    strategy: str = ""
    metadata: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


_RE_CITATION = re.compile(r"\[\d+\]")
_RE_URL = re.compile(r"https?://\S+")
_RE_BOLD = re.compile(r"\*{1,2}(.*?)\*{1,2}")
_RE_HEADING = re.compile(r"^#{1,6}\s+", re.MULTILINE)
_RE_SOFT_NL = re.compile(r"(?<!\n)\n(?!\n)")
_RE_MULTI_NL = re.compile(r"\n{3,}")
_RE_WHITESPACE = re.compile(r"[ \t]+")
_RE_PAGE_NUM = re.compile(r"^\s*\d+\s*$", re.MULTILINE)

LIST_PATTERN = re.compile(r"\n\s*(\d+\.|-|\*)\s+")



def clean_text(text: str) -> str:

    text = _RE_CITATION.sub("", text)
    text = _RE_URL.sub("", text)
    text = _RE_BOLD.sub(r"\1", text)
    text = _RE_HEADING.sub("", text)
    text = _RE_PAGE_NUM.sub("", text)

    text = _RE_SOFT_NL.sub(" ", text)
    text = _RE_MULTI_NL.sub("\n\n", text)
    text = _RE_WHITESPACE.sub(" ", text)

    return text.strip()



def extract_sections(text: str):

    pattern = re.compile(r"^(#{1,6}\s+.+)$", re.MULTILINE)
    matches = list(pattern.finditer(text))

    if not matches:
        return [("", text)]

    sections = []

    for idx, match in enumerate(matches):

        title = match.group(1).lstrip("#").strip()

        start = match.end()

        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)

        body = text[start:end].strip()

        if body:
            sections.append((title, body))

    return sections


def token_count(text: str):
    return len(_encoding.encode(text))


def is_list_block(text: str):
    return bool(LIST_PATTERN.search(text))


def stable_id(text: str, source: str, idx: int):
    payload = f"{source}::{idx}::{text[:100]}"
    return hashlib.sha256(payload.encode()).hexdigest()[:16]



def recursive_chunk(text: str, cfg: ChunkConfig):

    tokens = _encoding.encode(text)

    chunks = []

    start = 0

    while start < len(tokens):

        end = min(start + cfg.recursive_size, len(tokens))

        chunk = _encoding.decode(tokens[start:end]).strip()

        chunks.append(chunk)

        if end == len(tokens):
            break

        start = end - cfg.recursive_overlap

    return chunks


def semantic_chunk(text: str, cfg: ChunkConfig):

    sentences = nltk.sent_tokenize(text)

    if len(sentences) <= 1:
        return sentences

    embeddings = _embedding_model.encode(
        sentences,
        batch_size=64,
        show_progress_bar=False,
    )

    chunks = []

    current = [sentences[0]]

    for i in range(1, len(sentences)):

        sim = cosine_similarity(
            [embeddings[i - 1]],
            [embeddings[i]]
        )[0][0]

        tokens = token_count(" ".join(current))

        boundary = (
            sim < cfg.similarity_threshold
            and tokens >= cfg.min_chunk_tokens
        ) or tokens >= cfg.max_chunk_tokens

        if boundary:

            chunks.append(" ".join(current))

            current = [sentences[i]]

        else:

            current.append(sentences[i])

    if current:
        chunks.append(" ".join(current))

    return chunks


def add_context_window(chunks, window):

    if window <= 0 or len(chunks) < 2:
        return chunks

    result = []

    for i, chunk in enumerate(chunks):

        prev = chunks[i - 1] if i > 0 else ""

        nxt = chunks[i + 1] if i < len(chunks) - 1 else ""

        text = " ".join([prev, chunk, nxt]).strip()

        result.append(text)

    return result



def create_chunks(text: str, source: str, cfg: ChunkConfig | None = None):

    cfg = cfg or ChunkConfig()

    cleaned = clean_text(text)

    sections = extract_sections(cleaned)

    all_chunks = []

    idx = 0

    for section_title, section_text in sections:

        if is_list_block(section_text):

            sem_chunks = [section_text]

            strategy = "section"

        else:

            sem_chunks = semantic_chunk(section_text, cfg)

            strategy = "semantic"

        split_chunks = []

        for chunk in sem_chunks:

            if token_count(chunk) > cfg.max_chunk_tokens:

                for sub in recursive_chunk(chunk, cfg):

                    split_chunks.append((sub, "recursive"))

            else:

                split_chunks.append((chunk, strategy))

        texts = [c[0] for c in split_chunks]

        strategies = [c[1] for c in split_chunks]

        if cfg.context_window > 0:

            texts = add_context_window(texts, cfg.context_window)

        for text_chunk, strat in zip(texts, strategies):

            tc = token_count(text_chunk)

            if tc < cfg.min_chunk_tokens:
                continue

            text_chunk = f"{section_title}\n\n{text_chunk}".strip()

            chunk = Chunk(

                text=text_chunk,
                chunk_id=stable_id(text_chunk, source, idx),
                source=source,
                section=section_title,
                token_count=tc,
                strategy=strat,
            )

            all_chunks.append(chunk)

            idx += 1

    return all_chunks



def load_pdf(path):

    converter = DocumentConverter()

    result = converter.convert(path)

    return result.document.export_to_markdown()


def load_all_files(root):

    supported = {".pdf", ".txt", ".md"}

    for root_dir, _, files in os.walk(root):

        for file in files:

            path = os.path.join(root_dir, file)

            ext = Path(file).suffix.lower()

            if ext not in supported:
                continue

            try:

                if ext == ".pdf":

                    text = load_pdf(path)

                else:

                    text = Path(path).read_text(encoding="utf-8")

                logger.info(f"Loaded {path}")

                yield path, text

            except Exception as e:

                logger.warning(f"Skipping {path}: {e}")


if __name__ == "__main__":

    cfg = ChunkConfig()

    dataset_folder = "knowledge-base"

    all_chunks = []

    file_count = 0

    for path, text in load_all_files(dataset_folder):

        chunks = create_chunks(text, path, cfg)

        all_chunks.extend(c.to_dict() for c in chunks)

        file_count += 1

        logger.info(f"{len(chunks)} chunks from {path}")

    logger.info(f"Files processed: {file_count}")

    logger.info(f"Total chunks: {len(all_chunks)}")

    with open("chunks.json", "w", encoding="utf-8") as f:

        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    logger.info("Saved chunks.json")