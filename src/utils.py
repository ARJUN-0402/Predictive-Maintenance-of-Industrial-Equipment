import logging
import sys
from pathlib import Path

import pandas as pd


def setup_logging(
    name: str = "predictive_maintenance",
    level: int = logging.INFO,
    log_file: Path | None = None,
) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        logger.handlers.clear()
    logger.setLevel(level)
    formatter = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode="a")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


def card_html(title: str, value: str, color: str) -> str:
    return (
        f'<div style="background:#1c1e26;border-radius:12px;padding:16px;'
        f"margin:8px;text-align:center;border-left:4px solid {color};'>"
        f'<div style="font-size:12px;color:#8b8b9b;margin-bottom:4px;">'
        f"{title}</div>"
        f'<div style="font-size:28px;font-weight:bold;color:{color};">'
        f"{value}</div></div>"
    )


def generate_report(
    title: str,
    sections: list[dict[str, str]],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    from fpdf import FPDF

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 12, title, new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(8)
    for section in sections:
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, section.get("heading", ""), new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 10)
        for line in section.get("lines", []):
            pdf.multi_cell(0, 6, str(line))
        pdf.ln(4)
    pdf.output(str(output_path))


def validate_columns(df: pd.DataFrame, expected: list[str]) -> None:
    missing = set(expected) - set(df.columns)
    extra = set(df.columns) - set(expected)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if extra:
        raise ValueError(f"Unexpected columns: {sorted(extra)}")


__all__ = [
    "setup_logging",
    "card_html",
    "generate_report",
    "validate_columns",
]
