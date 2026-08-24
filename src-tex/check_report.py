"""Run local, reproducible checks on the report.

This is a citation, consistency, and exact-phrase similarity audit. It cannot
predict the result of a university or third-party authorship checker.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = Path(__file__).resolve().parent
TEX_PATH = REPORT_DIR / "report.tex"
BIB_PATH = REPORT_DIR / "references.bib"
RESULTS_PATH = REPORT_DIR / "generated" / "results.json"
PDF_CANDIDATES = [REPORT_DIR / "report.pdf", REPORT_DIR / "build" / "report.pdf"]
SAMPLE_PATH = ROOT / "instructions" / "sample.pdf"
INSTRUCTIONS_PATH = ROOT / "instructions" / "instructions.md"

REQUIRED_FIGURES = {
    "workflow.pdf",
    "dataset_audit.pdf",
    "correlation_heatmap.pdf",
    "model_comparison.pdf",
    "confusion_matrices.pdf",
    "roc_curves.pdf",
    "permutation_importance.pdf",
}

EXPECTED_AUTHOR_IDS = {
    "PUR079BCT074",
    "PUR079BCT078",
    "PUR079BCT094",
}
EXPECTED_AUTHOR_NAMES = {
    "Sandesh Poudel",
    "Sankalpa Gautam",
    "Tilak Thapa",
}
FORBIDDEN_MANUSCRIPT_TEXT = {
    "Utsab Pandey",
    "PUR079BCT095",
    "Acknowledgement",
    "Declarations",
    "Future Work",
    "exact website from which this copy was downloaded",
    "exact download source",
    "precise download location",
    "full provenance should be confirmed",
    "70:30",
    "87.01\\%",
    "grouped median imputation",
    "probability=True",
    "max_iter=500",
    "min_samples_leaf=5",
}
DATASET_SOURCE_URL = "https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database"
REQUIRED_FRONT_MATTER = {
    r"\begin{titlepage}": "custom cover page",
    "assets/tu-logo.png": "TU logo",
    r"\thispdfpagelabel{}": "blank PDF label for the cover",
    r"\usepackage{newtxtext,newtxmath}": "Times-style text and math fonts",
    r"\pagenumbering{roman}": "Roman front-matter numbering",
    r"\tableofcontents": "table of contents",
    "LIST OF ABBREVIATIONS": "list of abbreviations",
    r"\pagenumbering{arabic}": "Arabic main-matter numbering",
}
FORBIDDEN_FRONT_MATTER = {
    r"\listoffigures": "list of figures",
    r"\listoftables": "list of tables",
}


def fail(message: str, errors: list[str]) -> None:
    errors.append(message)
    print(f"FAIL: {message}")


def pass_check(message: str) -> None:
    print(f"PASS: {message}")


def extract_pdf_text(pdf_path: Path, *, preserve_layout: bool = False) -> str:
    if shutil.which("pdftotext") is None:
        raise RuntimeError("pdftotext is required for the local similarity audit")
    with tempfile.NamedTemporaryFile(suffix=".txt") as output:
        command = ["pdftotext"]
        if preserve_layout:
            command.append("-layout")
        command.extend([str(pdf_path), output.name])
        subprocess.run(
            command,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        return Path(output.name).read_text(encoding="utf-8", errors="ignore")


def body_words(text: str) -> list[str]:
    """Normalize body prose and ignore one-character equation symbols."""

    text = re.split(r"\nReferences\s*\n", text, maxsplit=1, flags=re.IGNORECASE)[0]
    words = re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)?", text.lower())
    return [word for word in words if len(word) > 1]


def exact_ngram_overlap(left_text: str, right_text: str, size: int = 10) -> set[tuple[str, ...]]:
    left = body_words(left_text)
    right = body_words(right_text)
    left_ngrams = {tuple(left[index : index + size]) for index in range(len(left) - size + 1)}
    right_ngrams = {tuple(right[index : index + size]) for index in range(len(right) - size + 1)}
    return left_ngrams & right_ngrams


def check_citations(tex: str, bib: str, errors: list[str]) -> None:
    cited: set[str] = set()
    for citation_group in re.findall(r"\\cite[a-zA-Z]*\{([^}]+)\}", tex):
        cited.update(key.strip() for key in citation_group.split(",") if key.strip())
    entries = set(re.findall(r"@\w+\s*\{\s*([^,\s]+)\s*,", bib))

    missing = sorted(cited - entries)
    unused = sorted(entries - cited)
    if missing:
        fail(f"citation keys missing from references.bib: {', '.join(missing)}", errors)
    else:
        pass_check(f"all {len(cited)} citation keys exist in references.bib")
    if unused:
        fail(f"uncited bibliography entries: {', '.join(unused)}", errors)
    else:
        pass_check("every bibliography entry is cited")

    if DATASET_SOURCE_URL not in bib or r"\cite{uciKagglePima}" not in tex:
        fail("the verified Kaggle dataset source is missing or uncited", errors)
    else:
        pass_check("the verified Kaggle dataset source is cited")


def check_results(tex: str, result_data: dict, errors: list[str]) -> None:
    dataset = result_data["dataset"]
    split = result_data["split"]
    expected_counts = {
        "records": 768,
        "training_records": 614,
        "test_records": 154,
        "negative_records": 500,
        "positive_records": 268,
    }
    actual_counts = {
        "records": dataset["records"],
        "training_records": split["training_records"],
        "test_records": split["test_records"],
        "negative_records": dataset["class_counts"]["0"],
        "positive_records": dataset["class_counts"]["1"],
    }
    if actual_counts != expected_counts:
        fail(f"unexpected dataset or split counts: {actual_counts}", errors)
    else:
        pass_check("dataset and split counts match the rerun")

    expected_medians = {
        "Glucose": 117.0,
        "BloodPressure": 72.0,
        "SkinThickness": 29.0,
        "Insulin": 125.0,
        "BMI": 32.4,
    }
    actual_medians = result_data["preprocessing"]["training_medians"]
    if actual_medians != expected_medians:
        fail(f"unexpected training medians: {actual_medians}", errors)
    else:
        pass_check("training-only imputation medians match the manuscript")

    generated_table = (REPORT_DIR / "generated" / "model_metrics.tex").read_text(encoding="utf-8")
    table_error_count = len(errors)
    for model_name, values in result_data["models"].items():
        for metric in ["accuracy", "precision", "sensitivity", "specificity", "f1", "roc_auc"]:
            token = f"{100 * float(values[metric]):.2f}\\%"
            if token not in generated_table:
                fail(f"{model_name} {metric} value {token} is absent from the generated table", errors)
    if len(errors) == table_error_count:
        pass_check("the generated metrics table matches results.json")

    required_statements = {
        "768 total records": r"(?:\b768\b.{0,40}\b(?:records|rows)\b|\b(?:records|rows)\b.{0,40}\b768\b)",
        "614 training records": r"(?:\b614\b.{0,40}\btrain\w*\b|\btrain\w*\b.{0,40}\b614\b)",
        "154 held-out test records": r"(?:\b154\b.{0,40}\b(?:test\w*|held-out)\b|\b(?:test\w*|held-out)\b.{0,40}\b154\b)",
        "best accuracy": r"78\.57\\%",
        "best ROC--AUC": r"81\.04\\%",
        "single-split limitation": r"\b(?:one|single)\b.{0,30}\b(?:train--test\s+)?split\b",
        "educational scope": r"\beducational demonstration\b",
    }
    missing_statements = [
        label
        for label, pattern in required_statements.items()
        if re.search(pattern, tex, flags=re.IGNORECASE) is None
    ]
    if missing_statements:
        fail(f"required result or limitation statement is missing: {missing_statements}", errors)
    else:
        pass_check("key findings and limitations are stated in the manuscript")


def check_report_structure(tex: str, errors: list[str]) -> None:
    tex_casefold = tex.casefold()
    missing_names = sorted(name for name in EXPECTED_AUTHOR_NAMES if name.casefold() not in tex_casefold)
    if missing_names:
        fail(f"required cover authors are missing: {', '.join(missing_names)}", errors)
    else:
        pass_check("the cover contains all three approved authors")

    found_ids = set(re.findall(r"PUR079BCT\d{3}", tex, flags=re.IGNORECASE))
    found_ids = {author_id.upper() for author_id in found_ids}
    if found_ids != EXPECTED_AUTHOR_IDS:
        fail(f"cover author IDs do not match the approved set: {sorted(found_ids)}", errors)
    else:
        pass_check("the cover contains exactly the three approved author IDs")

    forbidden = sorted(text for text in FORBIDDEN_MANUSCRIPT_TEXT if text.casefold() in tex_casefold)
    if forbidden:
        fail(f"excluded report text remains: {', '.join(forbidden)}", errors)
    else:
        pass_check("the excluded author and acknowledgement section are absent")

    missing_front_matter = [label for token, label in REQUIRED_FRONT_MATTER.items() if token not in tex]
    if missing_front_matter:
        fail(f"required report structure is missing: {', '.join(missing_front_matter)}", errors)
    else:
        pass_check("cover, contents, abbreviations, and Roman/Arabic numbering are configured")

    unwanted_front_matter = [label for token, label in FORBIDDEN_FRONT_MATTER.items() if token in tex]
    if unwanted_front_matter:
        fail(f"unwanted front matter remains: {', '.join(unwanted_front_matter)}", errors)
    else:
        pass_check("lists of figures and tables are omitted")


def check_render_log(errors: list[str]) -> None:
    log_path = REPORT_DIR / "build" / "report.log"
    if not log_path.exists():
        fail("build/report.log does not exist; compile the report first", errors)
        return
    log = log_path.read_text(encoding="utf-8", errors="ignore")
    bad_patterns = [
        "LaTeX Error",
        "undefined citations",
        "There were undefined references",
        "Overfull \\hbox",
        "Token not allowed in a PDF string",
        "between bookmark levels is greater",
        "destination with the same identifier",
    ]
    found = [pattern for pattern in bad_patterns if pattern.lower() in log.lower()]
    if found:
        fail(f"render log contains: {', '.join(found)}", errors)
    else:
        pass_check("render log has no errors, unresolved references, or overfull boxes")


def to_roman(number: int) -> str:
    values = [
        (1000, "m"),
        (900, "cm"),
        (500, "d"),
        (400, "cd"),
        (100, "c"),
        (90, "xc"),
        (50, "l"),
        (40, "xl"),
        (10, "x"),
        (9, "ix"),
        (5, "v"),
        (4, "iv"),
        (1, "i"),
    ]
    result: list[str] = []
    for value, numeral in values:
        while number >= value:
            result.append(numeral)
            number -= value
    return "".join(result)


def check_rendered_structure(pdf_path: Path, report_text: str, errors: list[str]) -> None:
    layout_text = extract_pdf_text(pdf_path, preserve_layout=True)
    pages = layout_text.split("\f")
    while pages and not pages[-1].strip():
        pages.pop()
    normalized_pages = [" ".join(page.split()) for page in pages]

    intro_pages = [
        index for index, page in enumerate(normalized_pages) if page.upper().startswith("1 INTRODUCTION")
    ]
    if len(intro_pages) != 1:
        fail(f"could not identify exactly one Introduction opening page: {intro_pages}", errors)
    else:
        intro_index = intro_pages[0]
        pagination_errors: list[str] = []
        for index in range(1, intro_index):
            lines = [line.strip() for line in pages[index].splitlines() if line.strip()]
            expected = to_roman(index)
            # PDF text extraction can place a displayed equation number after the
            # footer even though the footer is visibly at the bottom of the page.
            if not lines or expected not in lines[-5:]:
                pagination_errors.append(f"physical page {index + 1}: expected {expected}")
        for index in range(intro_index, len(pages)):
            lines = [line.strip() for line in pages[index].splitlines() if line.strip()]
            expected = str(index - intro_index + 1)
            if not lines or expected not in lines[-5:]:
                pagination_errors.append(f"physical page {index + 1}: expected {expected}")
        if pagination_errors:
            fail(f"rendered page-number sequence is incorrect: {pagination_errors[:4]}", errors)
        else:
            pass_check("rendered front matter uses Roman numerals and main matter restarts at Arabic 1")

    front_titles = [page.upper() for page in normalized_pages[1:4]]
    expected_front_titles = ["ABSTRACT", "TABLE OF CONTENTS", "LIST OF ABBREVIATIONS"]
    incorrect_front_titles = [
        expected
        for expected, page in zip(expected_front_titles, front_titles, strict=False)
        if not page.startswith(expected)
    ]
    if len(front_titles) != len(expected_front_titles) or incorrect_front_titles:
        fail(f"unexpected preliminary-page order: {incorrect_front_titles}", errors)
    else:
        pass_check("preliminary pages contain only the abstract, contents, and abbreviations")

    unwanted_lists = [
        heading for heading in ["LIST OF FIGURES", "LIST OF TABLES"] if heading in report_text.upper()
    ]
    if unwanted_lists:
        fail(f"unwanted rendered front matter remains: {', '.join(unwanted_lists)}", errors)
    else:
        pass_check("rendered PDF omits lists of figures and tables")

    if re.search(r"\bddd(?:[ivxlcdm]+|\d+)\b", report_text, flags=re.IGNORECASE):
        fail("rendered page numbers contain the Springer class's stray 'ddd' prefix", errors)
    else:
        pass_check("rendered page numbers contain no stray footer prefix")

    pdfinfo = subprocess.run(
        ["pdfinfo", str(pdf_path)], check=True, capture_output=True, text=True
    ).stdout
    size_match = re.search(r"Page size:\s+([0-9.]+) x ([0-9.]+) pts", pdfinfo)
    if size_match is None:
        fail("could not read the rendered PDF page size", errors)
    else:
        width, height = (float(value) for value in size_match.groups())
        if abs(width - 595.28) > 0.1 or abs(height - 841.89) > 0.1:
            fail(f"rendered page is not A4: {width} x {height} pt", errors)
        else:
            pass_check("rendered PDF uses A4 page dimensions")

    font_output = subprocess.run(
        ["pdffonts", str(pdf_path)], check=True, capture_output=True, text=True
    ).stdout
    if "TeXGyreTermesX" not in font_output:
        fail("rendered PDF is missing the configured Times-style text font", errors)
    else:
        pass_check("rendered PDF embeds the Times-style text font")


def main() -> int:
    errors: list[str] = []
    tex = TEX_PATH.read_text(encoding="utf-8")
    bib = BIB_PATH.read_text(encoding="utf-8")
    results = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))

    placeholder_patterns = ["TODO", "First Author", "lorem ipsum", "Sample body text"]
    placeholders = [pattern for pattern in placeholder_patterns if pattern.lower() in tex.lower()]
    if placeholders:
        fail(f"placeholder text remains: {', '.join(placeholders)}", errors)
    else:
        pass_check("no template placeholder text remains")

    check_citations(tex, bib, errors)
    check_results(tex, results, errors)
    check_report_structure(tex, errors)

    present_figures = {path.name for path in (REPORT_DIR / "figures").glob("*.pdf")}
    missing_figures = sorted(REQUIRED_FIGURES - present_figures)
    if missing_figures:
        fail(f"missing generated figures: {', '.join(missing_figures)}", errors)
    else:
        pass_check(f"all {len(REQUIRED_FIGURES)} required generated figures exist")

    check_render_log(errors)

    pdf_path = next((path for path in PDF_CANDIDATES if path.exists()), None)
    if pdf_path is None:
        fail("no rendered report PDF exists", errors)
    else:
        report_text = extract_pdf_text(pdf_path)
        report_text_casefold = report_text.casefold()
        missing_pdf_names = sorted(
            name for name in EXPECTED_AUTHOR_NAMES if name.casefold() not in report_text_casefold
        )
        if missing_pdf_names:
            fail(f"required authors are absent from the rendered PDF: {', '.join(missing_pdf_names)}", errors)
        else:
            pass_check("rendered cover contains all three approved authors")

        forbidden_pdf_text = sorted(
            text for text in FORBIDDEN_MANUSCRIPT_TEXT if text.casefold() in report_text_casefold
        )
        if forbidden_pdf_text:
            fail(f"excluded text appears in the rendered PDF: {', '.join(forbidden_pdf_text)}", errors)
        else:
            pass_check("excluded author and acknowledgement text are absent from the rendered PDF")

        check_rendered_structure(pdf_path, report_text, errors)

        sample_text = extract_pdf_text(SAMPLE_PATH)
        instruction_text = INSTRUCTIONS_PATH.read_text(encoding="utf-8", errors="ignore")
        for label, source_text in [("sample report", sample_text), ("assignment instructions", instruction_text)]:
            overlap = exact_ngram_overlap(report_text, source_text, size=10)
            if overlap:
                examples = [" ".join(words) for words in sorted(overlap)[:3]]
                fail(f"exact 10-word phrase overlap with {label}: {examples}", errors)
            else:
                pass_check(f"no exact 10-word prose overlap with the {label}")

    print("NOTE: This local audit cannot guarantee any third-party checker result.")
    if errors:
        print(f"Report audit failed with {len(errors)} issue(s).")
        return 1
    print("Report audit completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
