import PyPDF2
from pathlib import Path
from typing import List, Dict, Union, Optional
import re
import math
import uuid
from datetime import datetime
import json
import simple_parsing as sp
from dataclasses import dataclass
import tiktoken
import statistics  # Add to imports at top
from rich.table import Table  # Add to imports at top
from multiprocessing import Pool, cpu_count  # Add to imports at top
from functools import partial  # Add to imports at top
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn, BarColumn, TextColumn, TaskProgressColumn

from researcher.console import console


@dataclass
class PreprocessingArgs:
    """Arguments for preprocessing PDF documents"""

    data_dir: Path = Path("my_data/")  # Directory containing PDF files to process
    output_file: Path = None  # Output JSONL file path. Defaults to 'processed_documents.jsonl' in data_dir
    chunk_size: int = 512  # Target size of each chunk in characters
    max_workers: int = 6  # Maximum number of worker processes. Defaults to CPU count
    max_tokens_len: int = 100000  # Maximum number of tokens allowed per document
    skipped_docs_file: Path = (
        None  # Output file for skipped documents. Defaults to 'skipped_documents.jsonl' in data_dir
    )


def generate_chunks(doc_id: str, document_content: str, chunk_size: int = 2048) -> List[Dict]:
    """
    Splits a single document into chunks at natural break points (sentences or words).

    Args:
        doc_id (str): The identifier of the document.
        document_content (str): The content of the document as a single string.
        chunk_size (int): The target size of each chunk in characters.

    Returns:
        list: A list of dictionaries, where each dictionary represents a chunk.
    """
    chunks = []

    # Split into sentences first
    sentences = re.split(r"(?<=[.!?])\s+", document_content)

    current_chunk = []
    current_size = 0

    for sentence in sentences:
        sentence = sentence.strip()
        sentence_len = len(sentence)

        # If a single sentence is longer than chunk_size, split on word boundaries
        if sentence_len > chunk_size:
            if current_chunk:  # Store the current chunk before processing long sentence
                chunks.append(
                    {
                        "chunk_id": f"{doc_id}_chunk_{len(chunks)}",
                        "original_index": len(chunks),
                        "content": " ".join(current_chunk).strip(),
                    }
                )
                current_chunk = []
                current_size = 0

            # Split long sentence into words
            words = sentence.split()
            temp_chunk = []
            temp_size = 0

            for word in words:
                word_len = len(word) + 1  # +1 for space
                if temp_size + word_len > chunk_size and temp_chunk:
                    chunks.append(
                        {
                            "chunk_id": f"{doc_id}_chunk_{len(chunks)}",
                            "original_index": len(chunks),
                            "content": " ".join(temp_chunk).strip(),
                        }
                    )
                    temp_chunk = [word]
                    temp_size = word_len
                else:
                    temp_chunk.append(word)
                    temp_size += word_len

            if temp_chunk:  # Don't forget the last chunk
                chunks.append(
                    {
                        "chunk_id": f"{doc_id}_chunk_{len(chunks)}",
                        "original_index": len(chunks),
                        "content": " ".join(temp_chunk).strip(),
                    }
                )

        # Normal case: sentence is shorter than chunk_size
        elif current_size + sentence_len + 1 <= chunk_size:
            current_chunk.append(sentence)
            current_size += sentence_len + 1
        else:
            # Current chunk is full, store it and start a new one
            if current_chunk:
                chunks.append(
                    {
                        "chunk_id": f"{doc_id}_chunk_{len(chunks)}",
                        "original_index": len(chunks),
                        "content": " ".join(current_chunk).strip(),
                    }
                )
            current_chunk = [sentence]
            current_size = sentence_len + 1

    # Don't forget the last chunk
    if current_chunk:
        chunks.append(
            {
                "chunk_id": f"{doc_id}_chunk_{len(chunks)}",
                "original_index": len(chunks),
                "content": " ".join(current_chunk).strip(),
            }
        )

    return chunks


def extract_text_from_pdf(file_path: Union[str, Path]) -> str:
    """
    Extract text content from a PDF file.

    Args:
        file_path: Path to the PDF file

    Returns:
        Extracted text content as a string
    """
    try:
        with open(file_path, "rb") as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        return text
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return ""


def clean_text(text: str) -> str:
    """
    Clean extracted text by removing extra whitespace, special characters, etc.

    Args:
        text: Input text to clean

    Returns:
        Cleaned text
    """
    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text)
    # Remove special characters
    text = re.sub(r"[^\w\s.,!?-]", "", text)
    # Strip leading/trailing whitespace
    text = text.strip()
    return text


def extract_date_from_filename(filename: str) -> str:
    """
    Extract date from filename if it matches the pattern YYYYMMDD
    Returns None if no date is found
    """
    date_pattern = r"^(\d{8})"
    match = re.match(date_pattern, filename)
    if match:
        date_str = match.group(1)
        try:
            date = datetime.strptime(date_str, "%Y%m%d")
            return date.isoformat()[:10]  # Returns YYYY-MM-DD format
        except ValueError:
            return None
    return None


def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    """
    Count the number of tokens in a text string using tiktoken.

    Args:
        text (str): The text to count tokens for
        model (str): The model to use for token counting (default: gpt-3.5-turbo)

    Returns:
        int: Number of tokens in the text
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))


def process_file(file_path: Union[str, Path]) -> Dict[str, Union[str, List[Dict[str, str]], int, Optional[str]]]:
    """
    Process a single file and return its metadata, content, and chunks.

    Args:
        file_path: Path to the file

    Returns:
        Dictionary containing file metadata, processed content, chunks, and token counts
    """
    file_path = Path(file_path)
    doc_id = file_path.stem.lower().replace(" ", "_")
    result = {
        "doc_id": doc_id,
        "original_doc_name": file_path.name,
        "original_uuid": str(uuid.uuid4()),
        "date": extract_date_from_filename(file_path.name),
        "content": "",
        "chunks": [],
        "total_tokens": 0,
        "error": None,
    }

    if file_path.suffix.lower() == ".pdf":
        text = extract_text_from_pdf(file_path)
        result["content"] = clean_text(text)
        result["chunks"] = generate_chunks(doc_id, result["content"])
        result["total_tokens"] = count_tokens(result["content"])

        # Add token count to each chunk
        for chunk in result["chunks"]:
            chunk["tokens"] = count_tokens(chunk["content"])
    else:
        result["error"] = f"Unsupported file type: {file_path.suffix}"

    return result


def batch_process_files(
    file_paths: List[Union[str, Path]], max_workers: int = None, max_tokens_len: int = 100000
) -> List[Dict]:
    """
    Process multiple files in parallel.

    Args:
        file_paths: List of file paths to process
        max_workers: Maximum number of worker processes (defaults to CPU count)
        max_tokens_len: Maximum number of tokens allowed per document

    Returns:
        List of dictionaries containing processed file data
    """
    if max_workers is None:
        max_workers = cpu_count()

    results = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=None),
        TaskProgressColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        console=console,
        expand=True,
    ) as progress:
        task = progress.add_task(f"[green]Processing files using {max_workers} workers", total=len(file_paths))

        with Pool(processes=max_workers) as pool:
            for result in pool.imap(process_file, file_paths):
                if result["total_tokens"] <= max_tokens_len:
                    results.append(result)
                else:
                    console.print(
                        f"[yellow]Skipping {result['original_doc_name']} - exceeds token limit ({result['total_tokens']:,} tokens)"
                    )
                progress.advance(task)

    return results


def get_pdf_files(directory: Union[str, Path]) -> List[Path]:
    """
    Recursively get all PDF files from a directory and its subdirectories.

    Args:
        directory: Root directory to search for PDFs

    Returns:
        List of Path objects for all PDF files found
    """
    directory = Path(directory)
    pdf_files = []

    for file_path in directory.rglob("*.pdf"):
        pdf_files.append(file_path)

    return pdf_files


def process_directory(directory: Union[str, Path]) -> List[Dict[str, str]]:
    """
    Process all PDF files in a directory and its subdirectories.

    Args:
        directory: Root directory containing PDF files

    Returns:
        List of dictionaries containing processed file data
    """
    pdf_files = get_pdf_files(directory)
    print(f"Found {len(pdf_files)} PDF files in {directory}")
    return batch_process_files(pdf_files)


def save_to_jsonl(documents: List[Dict], output_path: Union[str, Path]) -> None:
    """
    Save processed documents to a JSONL file.

    Args:
        documents: List of processed document dictionaries
        output_path: Path where to save the JSONL file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for doc in documents:
            json_line = json.dumps(doc, ensure_ascii=False)
            f.write(json_line + "\n")


def calculate_token_stats(documents: List[Dict], chunk_size: int) -> Dict[str, Union[int, float, str]]:
    """Calculate statistics about token usage across all documents."""
    # Get document stats with filenames
    doc_tokens_with_names = [(doc["total_tokens"], doc["original_doc_name"]) for doc in documents]

    # Find min/max with filenames
    min_doc = min(doc_tokens_with_names, key=lambda x: x[0])
    max_doc = max(doc_tokens_with_names, key=lambda x: x[0])

    # Get just tokens for other calculations
    doc_tokens = [t[0] for t in doc_tokens_with_names]
    chunk_tokens = [chunk["tokens"] for doc in documents for chunk in doc["chunks"]]

    return {
        "documents": {
            "min": min_doc[0],
            "min_file": min_doc[1],
            "max": max_doc[0],
            "max_file": max_doc[1],
            "avg": round(statistics.mean(doc_tokens), 2),
            "median": round(statistics.median(doc_tokens), 2),
            "total": sum(doc_tokens),
        },
        "chunks": {
            "min": chunk_size,  # Fixed size
            "max": chunk_size,  # Fixed size
            "avg": chunk_size,  # Fixed size
            "median": chunk_size,  # Fixed size
            "total": sum(chunk_tokens),
            "count": len(chunk_tokens),
        },
    }


def truncate_filename(filename: str, max_length: int = 50) -> str:
    """Truncate filename if longer than max_length and add ellipsis."""
    return filename if len(filename) <= max_length else filename[: max_length - 3] + "..."


def main():
    args = sp.parse(PreprocessingArgs)

    # Set default output file if not provided
    if args.output_file is None:
        args.output_file = args.data_dir / "processed_documents.jsonl"

    console.print(f"Processing directory: {args.data_dir}")
    pdf_files = get_pdf_files(args.data_dir)
    console.print(f"Found {len(pdf_files)} PDF files in {args.data_dir}")

    # Process all files with specified chunk size and worker count
    processed_files = batch_process_files(pdf_files, max_workers=args.max_workers, max_tokens_len=args.max_tokens_len)

    # Save to JSONL
    save_to_jsonl(processed_files, args.output_file)
    console.print(f"\n[green]Saved {len(processed_files)} documents to {args.output_file}")

    # Calculate token statistics
    token_stats = calculate_token_stats(processed_files, args.chunk_size)

    console.rule("[bold blue]Token Statistics")

    # Create and populate the table
    table = Table(title="Token Usage Statistics", show_header=True, header_style="bold magenta")

    # Add columns
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")
    table.add_column("Document", justify="left")

    # Add rows
    table.add_row(
        "Minimum Tokens",
        f"{token_stats['documents']['min']:,} tokens",
        truncate_filename(token_stats["documents"]["min_file"]),
    )
    table.add_row(
        "Maximum Tokens",
        f"{token_stats['documents']['max']:,} tokens",
        truncate_filename(token_stats["documents"]["max_file"]),
    )
    table.add_row("Average Tokens", f"{token_stats['documents']['avg']:,} tokens", "")
    table.add_row("Median Tokens", f"{token_stats['documents']['median']:,} tokens", "")
    table.add_row("Total Tokens", f"{token_stats['documents']['total']:,} tokens", "")
    table.add_row("Document Count", f"{len(processed_files):,} documents", "")
    table.add_row("Chunk Count", f"{token_stats['chunks']['count']:,} chunks", "")

    # Print the table
    console.print(table)

    # Show details of first file as example
    processed_file = processed_files[0]
    console.rule("[bold blue]First Document Details")
    console.print(f"Doc ID: {processed_file['doc_id']}")
    console.print(f"Original Name: {processed_file['original_doc_name']}")
    console.print(f"UUID: {processed_file['original_uuid']}")
    console.print(f"Date: {processed_file['date']}")

    console.rule("[bold blue]Content Preview")
    console.print(f"{processed_file['content'][:200]}...")

    console.rule("[bold blue]Chunks")
    for chunk in processed_file["chunks"][:2]:
        console.print(chunk)
        console.print("---")


if __name__ == "__main__":
    main()
