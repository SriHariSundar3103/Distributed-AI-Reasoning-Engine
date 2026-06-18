"""
PDF Export Module

Converts markdown reports to PDF format using fpdf2.
Handles basic markdown formatting (headers, bold, lists).
"""

import re
from pathlib import Path
from typing import Optional

from fpdf import FPDF

from .logger import get_logger

logger = get_logger(__name__)


class ReportPDF(FPDF):
    """Custom PDF class for research reports."""
    
    def __init__(self, title: str = "Research Report"):
        super().__init__()
        self.title = title
        self.set_auto_page_break(auto=True, margin=15)
        
    def header(self):
        """Page header."""
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, self.title, align="C")
        self.ln(10)
        
    def footer(self):
        """Page footer with page number."""
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")


def markdown_to_pdf(markdown_content: str, output_path: str, title: str = "Research Report") -> str:
    """
    Convert markdown content to PDF.
    
    Args:
        markdown_content: Markdown text to convert
        output_path: Path to save the PDF
        title: Report title for header
        
    Returns:
        Path to the generated PDF file
    """
    pdf = ReportPDF(title=title)
    pdf.add_page()
    
    # Process markdown line by line
    lines = markdown_content.split("\n")
    
    for line in lines:
        line = line.strip()
        
        if not line:
            pdf.ln(5)
            continue
            
        try:
            # Handle headers
            if line.startswith("# "):
                pdf.set_font("Helvetica", "B", 18)
                pdf.set_text_color(0, 0, 0)
                pdf.multi_cell(0, 10, line[2:])
                pdf.ln(3)
                
            elif line.startswith("## "):
                pdf.set_font("Helvetica", "B", 14)
                pdf.set_text_color(50, 50, 50)
                pdf.multi_cell(0, 8, line[3:])
                pdf.ln(2)
                
            elif line.startswith("### "):
                pdf.set_font("Helvetica", "B", 12)
                pdf.set_text_color(80, 80, 80)
                pdf.multi_cell(0, 7, line[4:])
                pdf.ln(2)
                
            # Handle bullet points
            elif line.startswith("- ") or line.startswith("* "):
                pdf.set_font("Helvetica", "", 11)
                pdf.set_text_color(0, 0, 0)
                text = "- " + line[2:]
                text = _strip_markdown(text)
                text = _wrap_long_words(text)
                pdf.set_x(15)
                pdf.multi_cell(0, 6, text)
                
            # Handle numbered lists
            elif re.match(r"^\d+\.", line):
                pdf.set_font("Helvetica", "", 11)
                pdf.set_text_color(0, 0, 0)
                text = _strip_markdown(line)
                text = _wrap_long_words(text)
                pdf.set_x(15)
                pdf.multi_cell(0, 6, text)
                
            # Regular paragraph
            else:
                pdf.set_font("Helvetica", "", 11)
                pdf.set_text_color(0, 0, 0)
                text = _strip_markdown(line)
                text = _wrap_long_words(text)
                pdf.multi_cell(0, 6, text)
                
        except Exception as e:
            logger.warning(f"Failed to render line: {line[:30]}...", error=str(e))
            # Fallback: simple text
            pdf.set_font("Helvetica", "", 10)
            pdf.set_text_color(255, 0, 0)  # Red to indicate error in PDF
            try:
                clean_line = _strip_markdown(line)
                # Force very aggressive wrapping or just truncation
                clean_line = clean_line[:80] + " [Line truncated due to render error]"
                pdf.set_x(10) # Reset margin
                pdf.multi_cell(0, 6, clean_line)
            except:
                pass # Give up on this line
    
    # Ensure output directory exists
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Save PDF
    pdf.output(str(output_file))
    logger.info("PDF generated", path=str(output_file))
    
    return str(output_file)


def _strip_markdown(text: str) -> str:
    """Remove markdown formatting from text."""
    # Remove bold
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"__(.*?)__", r"\1", text)
    # Remove italic
    text = re.sub(r"\*(.*?)\*", r"\1", text)
    text = re.sub(r"_(.*?)_", r"\1", text)
    # Remove inline code
    text = re.sub(r"`(.*?)`", r"\1", text)
    # Remove links, keep text
    text = re.sub(r"\[(.*?)\]\(.*?\)", r"\1", text)
    return text


def _wrap_long_words(text: str, limit: int = 40) -> str:
    """
    Insert spaces in very long words (like URLs) to allow wrapping.
    Fixes 'Not enough horizontal space' error in fpdf2.
    """
    words = text.split(' ')
    wrapped_words = []
    
    for word in words:
        if len(word) > limit:
            # Split into chunks of 'limit' size
            chunks = [word[i:i+limit] for i in range(0, len(word), limit)]
            wrapped_words.append(" ".join(chunks))
        else:
            wrapped_words.append(word)
            
    return " ".join(wrapped_words)

