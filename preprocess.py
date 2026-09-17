"""
WISDOM-AI preprocessing pipeline

Input:
    The scanned "The Noble Qur'an in the English Language.pdf"
    supplied for this project.

What this script does:
    1. Downloads the PDF from Google Drive if needed.
    2. Renders PDF pages to images.
    3. OCRs the English text with Tesseract.
    4. Detects Surah headings and Ayah numbers.
    5. Removes the surrounding introductory material and
       page-level footnotes from the searchable corpus.
    6. Creates Quran passage records.
    7. Groups nearby Ayahs into retrieval chunks.
    8. Creates Sentence Transformer embeddings.
    9. Builds a normalized FAISS inner-product index.
    10. Saves:
          data/quran_chunks.json
          data/quran.index

Recommended environment:
    Google Colab or a local machine.

For Colab:
    !apt-get update -qq
    !apt-get install -y -qq tesseract-ocr
    !pip install pymupdf pytesseract pillow sentence-transformers \
        faiss-cpu tqdm gdown
"""

import argparse
import json
import re
from pathlib import Path

import pymupdf
import gdown
import numpy as np
import pytesseract
from PIL import Image
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm
import faiss


DRIVE_FILE_ID = "1AKrKXmXKtfuWeFNkuTMY4xt3ilvubIr-"
DEFAULT_PDF = Path("The Noble Qur'an in the ENGLISH Language.pdf")

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Render resolution. 2.0 corresponds to roughly 144 DPI.
RENDER_SCALE = 2.0

# Number of Ayahs combined into one retrieval unit.
AYAH_GROUP_SIZE = 3

SURAH_RE = re.compile(
    r"(?:Sirah|Surah)\s+(\d{1,3})\.\s+(.+?)(?:\s+\d+\s+Part\b.*)?$",
    re.IGNORECASE,
)

AYAH_RE = re.compile(
    r"(?m)^\s*(\d{1,3})\s*\.\s*"
)

FOOTNOTE_START_RE = re.compile(
    r"^\s*(?:\[\d+\]|\(\s*V\.\s*\d+\s*:\s*\d+\s*\))"
)


def download_pdf(output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        return output_path

    url = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"

    print("Downloading Quran resource from Google Drive...")
    gdown.download(
        url,
        str(output_path),
        quiet=False,
    )

    if not output_path.exists():
        raise FileNotFoundError(
            "Google Drive download failed."
        )

    return output_path


def clean_ocr_text(text: str) -> str:
    text = text.replace("\x0c", "\n")
    text = text.replace("\r", "\n")

    # Normalize common OCR whitespace without destroying paragraph lines.
    lines = []
    for line in text.splitlines():
        line = re.sub(r"[ \t]+", " ", line).strip()

        if line:
            lines.append(line)

    return "\n".join(lines)


def detect_surah(lines):
    """
    Look for OCR forms such as:
        Sirah 2. Al-Baqarah 6 Part 1
    """
    for line in lines:
        match = SURAH_RE.search(line)
        if match:
            number = int(match.group(1))
            name = match.group(2).strip()
            return number, name

    return None


def remove_footnote_tail(text: str) -> str:
    """
    Footnotes in this resource often begin with [1], [2], etc.
    Once a footnote block starts after an Ayah, stop collecting it.
    """

    lines = text.splitlines()
    kept = []

    for line in lines:
        if FOOTNOTE_START_RE.match(line):
            break

        # Common footnote form:
        # (V.2:22) ...
        if re.match(r"^\s*\(\s*V\.\s*\d+\s*:\s*\d+\s*\)", line):
            break

        kept.append(line)

    return "\n".join(kept).strip()


def parse_page(text, page_number, current_surah, current_ayah):
    """
    Parse one OCR page into verse records.

    The parser deliberately relies on the resource's recurring
    Surah heading + numbered Ayah layout rather than treating
    arbitrary PDF page chunks as semantic units.
    """

    text = clean_ocr_text(text)
    lines = text.splitlines()

    detected = detect_surah(lines)

    if detected:
        current_surah = detected

    if current_surah is None:
        return [], current_surah, current_ayah

    # Find numbered Ayah starts.
    matches = list(AYAH_RE.finditer(text))

    if not matches:
        # Continuation of the previous Ayah.
        if current_ayah is not None:
            continuation = remove_footnote_tail(text)
            if continuation:
                current_ayah["text"] += " " + continuation

        return [], current_surah, current_ayah

    records = []

    # Text before the first numbered Ayah is usually a continuation
    # from the previous page, unless a new Surah starts on this page.
    prefix = text[:matches[0].start()].strip()

    if prefix and current_ayah is not None and not detected:
        prefix = remove_footnote_tail(prefix)
        if prefix:
            current_ayah["text"] += " " + prefix

    for i, match in enumerate(matches):
        ayah_number = int(match.group(1))

        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)

        ayah_text = text[start:end].strip()
        ayah_text = remove_footnote_tail(ayah_text)

        # Ignore obvious non-Ayah numbering from introductory material.
        if not ayah_text:
            continue

        # If a new Surah heading occurs on the page, its first
        # numbered verse belongs to the newly detected Surah.
        current_ayah = {
            "surah_number": current_surah[0],
            "surah_name": current_surah[1],
            "ayah_number": ayah_number,
            "text": ayah_text,
            "page": page_number,
        }

        records.append(current_ayah.copy())

    return records, current_surah, current_ayah


def ocr_pdf(pdf_path: Path):
    """
    OCR all pages.

    Tesseract is intentionally used here only during preprocessing.
    It is NOT needed by the deployed Streamlit application.
    """

    doc = pymupdf.open(pdf_path)

    all_records = []
    current_surah = None
    current_ayah = None

    for page_number in tqdm(
        range(len(doc)),
        desc="OCR + Quran parsing",
    ):
        page = doc.load_page(page_number)

        matrix = pymupdf.Matrix(RENDER_SCALE, RENDER_SCALE)
        pix = page.get_pixmap(
            matrix=matrix,
            alpha=False,
        )

        image = Image.frombytes(
            "RGB",
            [pix.width, pix.height],
            pix.samples,
        )

        text = pytesseract.image_to_string(
            image,
            lang="eng",
            config="--psm 6",
        )

        records, current_surah, current_ayah = parse_page(
            text,
            page_number + 1,
            current_surah,
            current_ayah,
        )

        all_records.extend(records)

    doc.close()

    return all_records


def clean_records(records):
    """
    Remove duplicates and obvious OCR garbage.
    """

    cleaned = []
    seen = set()

    for record in records:
        text = re.sub(r"\s+", " ", record["text"]).strip()

        # Remove repeated OCR punctuation noise.
        text = re.sub(r"\s+([,.;:!?])", r"\1", text)

        key = (
            record["surah_number"],
            record["ayah_number"],
        )

        if len(text) < 10:
            continue

        if key in seen:
            continue

        seen.add(key)

        cleaned.append(
            {
                "surah_number": record["surah_number"],
                "surah_name": record["surah_name"],
                "ayah_number": record["ayah_number"],
                "reference": (
                    f"Surah {record['surah_number']}. "
                    f"{record['surah_name']} "
                    f"{record['ayah_number']}"
                ),
                "text": text,
                "page": record["page"],
            }
        )

    return cleaned


def group_ayahs(records, group_size=AYAH_GROUP_SIZE):
    """
    Group adjacent Ayahs from the same Surah.

    Keeping the individual Ayah metadata makes source attribution
    possible while giving semantic retrieval enough surrounding
    context to work well.
    """

    grouped = []

    for i in range(0, len(records), group_size):
        batch = records[i:i + group_size]

        if not batch:
            continue

        # Never combine two different Surahs.
        if len({
            item["surah_number"]
            for item in batch
        }) != 1:
            # Process the first record alone and let the next group
            # start cleanly.
            for item in batch:
                grouped.append(
                    {
                        "reference": item["reference"],
                        "surah_number": item["surah_number"],
                        "surah_name": item["surah_name"],
                        "ayah_start": item["ayah_number"],
                        "ayah_end": item["ayah_number"],
                        "ayah_numbers": [item["ayah_number"]],
                        "text": item["text"],
                        "page": item["page"],
                    }
                )
            continue

        first = batch[0]
        last = batch[-1]

        references = [
            item["reference"]
            for item in batch
        ]

        grouped.append(
            {
                "reference": (
                    f"Surah {first['surah_number']}. "
                    f"{first['surah_name']} "
                    f"{first['ayah_number']}-{last['ayah_number']}"
                    if len(batch) > 1
                    else first["reference"]
                ),
                "surah_number": first["surah_number"],
                "surah_name": first["surah_name"],
                "ayah_start": first["ayah_number"],
                "ayah_end": last["ayah_number"],
                "ayah_numbers": [
                    item["ayah_number"]
                    for item in batch
                ],
                "text": " ".join(
                    item["text"]
                    for item in batch
                ),
                "page": first["page"],
                "ayah_references": references,
            }
        )

    return grouped


def build_faiss(chunks, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    model = SentenceTransformer(EMBEDDING_MODEL)

    texts = [
        item["text"]
        for item in chunks
    ]

    print(
        f"Creating embeddings for {len(texts):,} retrieval chunks..."
    )

    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    index_path = output_dir / "quran.index"
    metadata_path = output_dir / "quran_chunks.json"

    faiss.write_index(index, str(index_path))

    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            chunks,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("\nKnowledge base created:")
    print(f"FAISS index: {index_path}")
    print(f"Metadata:    {metadata_path}")
    print(f"Chunks:      {len(chunks):,}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pdf",
        type=str,
        default=str(DEFAULT_PDF),
        help="Path to the Quran PDF.",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="data",
        help="Directory for FAISS + metadata.",
    )

    args = parser.parse_args()

    pdf_path = Path(args.pdf)

    if not pdf_path.exists():
        pdf_path = download_pdf(pdf_path)

    print(f"Using PDF: {pdf_path}")

    records = ocr_pdf(pdf_path)

    print(f"\nRaw Ayah records detected: {len(records):,}")

    records = clean_records(records)

    print(f"Clean Ayah records: {len(records):,}")

    chunks = group_ayahs(records)

    print(f"Retrieval chunks: {len(chunks):,}")

    build_faiss(
        chunks,
        Path(args.output),
    )


if __name__ == "__main__":
    main()
