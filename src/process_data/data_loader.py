import json
import re
import unicodedata
from pathlib import Path

from bs4 import BeautifulSoup


def normalize_title(title):
    if not title:
        return ""

    # Chuyển Unicode về dạng chuẩn
    title = unicodedata.normalize("NFKC", title)

    # Loại bỏ HTML
    title = BeautifulSoup(title, "html.parser").get_text(" ")

    # Xử lý một số lệnh LaTeX thông dụng
    title = re.sub(r"\\(?:textbf|textit|emph|mathrm)\{([^{}]*)\}", r"\1", title)
    title = re.sub(r"\\(?:alpha|beta|gamma)", " ", title)

    # Loại bỏ các ký tự điều khiển LaTeX còn lại
    title = re.sub(r"\\[a-zA-Z]+", " ", title)

    # Chuyển thành chữ thường
    title = title.lower()

    # Loại bỏ dấu câu
    title = re.sub(r"[^\w\s]", " ", title)

    # Chuẩn hóa khoảng trắng
    title = " ".join(title.split())

    return title


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue

            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"JSON không hợp lệ tại {path}, dòng {line_number}"
                ) from exc


def load_papers(path):
    papers = {}

    for record in read_jsonl(path):
        paper_id = record.get("paperId")

        if not paper_id:
            continue

        year = record.get("year")

        try:
            year = int(year) if year is not None else None
        except (ValueError, TypeError):
            year = None

        title = record.get("title") or ""
        abstract = record.get("abstract") or ""

        keywords = record.get("raw_keywords")
        if not isinstance(keywords, list):
            keywords = []

        papers[paper_id] = {
            "paper_id": paper_id,
            "title": title,
            "normalized_title": normalize_title(title),
            "abstract": abstract,
            "venue": record.get("conference")
                     or record.get("venue")
                     or "",
            "year": year,
            "raw_keywords": keywords,

            # Chỉ dùng trong giai đoạn entity resolution
            "external_ids": record.get("externalIds") or {}
        }

    return papers


def load_references(path):
    references = {}

    for record in read_jsonl(path):
        paper_id = record.get("paperId")

        if not paper_id:
            continue

        refs = record.get("references") or []

        if not isinstance(refs, list):
            refs = []

        # Gộp các dòng cùng paperId nếu có
        references.setdefault(paper_id, set()).update(
            ref for ref in refs if isinstance(ref, str)
        )

    return references