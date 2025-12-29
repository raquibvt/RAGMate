import os
import re
import json
from bs4 import BeautifulSoup
import html2text

USEFUL_ITEMS = {"1", "1A", "1B", "2", "3", "4", "5", "6", "7", "7A", "8", "9", "9A", "9B"}
UNWANTED_SECTION_PATTERNS = [
    r'^Signatures$',
    r'^Exhibit Index$',
    r'^Exhibits$',
    r'Certifications.*302|906',
    r'Proxy Statement',
    r'Schedule II.*Valuation',
    r'Table of Contents',
]

def clean_10k_text_blacklist(text):
    blacklist_patterns = [
        r"(?i)http[s]?://",
        r"(?i)xbrl",
        r"(?i)presentation",
        r"(?i)namespace",
        r"(?i)definition",
        r"(?i)member",
        r"(?i)domainItemType",
        r"(?i)monetaryItemType",
        r"(?i)documentType",
        r"(?i)roleRef",
        r"(?i)auth_ref",
        r"(?i)localname",
        r"(?i)nsuri",
        r"(?i)lang",
        r"[^a-zA-Z0-9\s\.\,\$\%\|\:\-]",
        r"^[0-9\s\.\,\$\-]{20,}$",
    ]

    important_keywords = [
        "Item 1", "Item 1A", "Item 2", "Item 3", "Item 7", "Item 7A",
        "Item 8", "Item 9", "Financial Statements", "Management’s Discussion",
        "Risk Factors", "Controls and Procedures"
    ]

    lines = text.splitlines()
    cleaned_lines = []

    for line in lines:
        stripped = line.strip()
        if any(kw.lower() in stripped.lower() for kw in important_keywords):
            cleaned_lines.append(stripped)
            continue
        if stripped.startswith("|") and stripped.count("|") >= 2:
            cleaned_lines.append(stripped)
            continue
        if any(re.search(pattern, stripped) for pattern in blacklist_patterns):
            continue
        if re.search(r"[a-zA-Z]", stripped) and len(stripped) > 3:
            cleaned_lines.append(stripped)

    return "\n".join(cleaned_lines)

def remove_table_of_contents(text):
    toc_end_patterns = [
        r'\n\s*Item\s+1\s*[\.\-:]\s+Business',
        r'\n\s*Item\s+1A?\s*[\.\-:]\s+',
    ]
    for pattern in toc_end_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return text[match.start():]
    return text

def extract_structured_items(text):
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    item_pattern = re.compile(
        r'\n(Item\s+(1A?|1B|2|3|4|5|6|7A?|8|9A?|9B|10|11|12|13|14|15)[\.\:\-]?\s+([^\n]+))\n',
        re.IGNORECASE
    )
    matches = list(item_pattern.finditer(text))
    sections = []
    for i, match in enumerate(matches):
        full_header = match.group(1).strip()
        item_num = match.group(2).strip().upper()
        item_title = match.group(3).strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section_text = text[start:end].strip()
        sections.append({
            "item_num": f"Item {item_num}",
            "item_title": item_title,
            "section_text": section_text
        })
    return sections

def is_section_useful(item_num, item_title):
    if item_num.split()[-1] not in USEFUL_ITEMS:
        return False
    for pat in UNWANTED_SECTION_PATTERNS:
        if re.search(pat, item_title, re.IGNORECASE):
            return False
    return True

def extract_text_from_html(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    soup = BeautifulSoup(content, "lxml")
    for tag in soup(['script', 'style']):
        tag.decompose()
    return soup.get_text(separator="\n")

def extract_metadata_from_text(text):
    metadata = {}
    text = text.replace('\n', ' ').replace('\r', ' ')
    text = re.sub(r'\s+', ' ', text)
    patterns = {
        'form_type': r'CONFORMED SUBMISSION TYPE:\s*(\S+)',
        'company_name': r'COMPANY CONFORMED NAME:\s*(.+?)\s+CENTRAL INDEX KEY:',
        'fiscal_year_end': r'FISCAL YEAR END:\s*(\d+)',
        'period_of_report': r'CONFORMED PERIOD OF REPORT:\s*(\d+)',
        'filed_date': r'FILED AS OF DATE:\s*(\d+)'
    }
    for field, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            value = match.group(1).strip()
            if field in ['period_of_report', 'filed_date'] and len(value) == 8:
                value = f"{value[:4]}-{value[4:6]}-{value[6:]}"
            if field == 'fiscal_year_end' and len(value) == 4:
                value = f"{value[:2]}-{value[2:]}"
            metadata[field] = value
    exchange_match = re.findall(r'The Nasdaq Stock Market LLC|NYSE|CBOE|BATS|AMEX', text)
    if exchange_match:
        metadata['exchange_listings'] = list(set(exchange_match))
    sic_match = re.search(r'STANDARD INDUSTRIAL CLASSIFICATION:\s*(.+?)\s*\[', text)
    if sic_match:
        metadata['standard_industrial_classification'] = sic_match.group(1).strip()
    former_names = re.findall(r'FORMER CONFORMED NAME:\s*(.+?)\s+DATE OF NAME CHANGE:', text)
    if former_names:
        metadata['former_names'] = list(set([name.strip() for name in former_names]))
    return metadata

def process_10k_to_chunked_json(file_path, output_path):
    with open(file_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    plain_text = extract_text_from_html(file_path)
    plain_text = remove_table_of_contents(plain_text)
    sections = extract_structured_items(plain_text)
    filing_metadata = extract_metadata_from_text(raw_text)

    output_chunks = []
    source_doc = os.path.basename(output_path)

    for sec in sections:
        item_num = sec["item_num"]
        item_title = sec["item_title"]
        body = sec["section_text"].strip()
        full_section_name = f"{item_num} - {item_title}"

        if not is_section_useful(item_num, item_title):
            continue

        if len(body.split()) < 50:
            print(f"Skipping short/low-substance section: {full_section_name}")
            continue

        output_chunks.append({
            "text": body,
            "metadata": {
                "chunk_id": f"{source_doc}_chunk_{len(output_chunks)}",
                "source_doc": source_doc,
                "section": full_section_name,
                "doc_metadata": filing_metadata
            }
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_chunks, f, indent=2)

    print(f"Saved chunked 10-K to {output_path}")

def construct_filename(file_path, base_dir, filter_file_list=["10-K"]):
    rel_path = os.path.relpath(file_path, base_dir)
    parts = rel_path.split(os.sep)
    if len(parts) < 4:
        print(f"Skipping unexpected path structure: {file_path}")
        return None, None
    ticker = parts[0]
    form_type = parts[1]
    accession = parts[2]
    if form_type not in filter_file_list:
        return None, None
    try:
        year_suffix = accession.split('-')[1]
        year = int(f"20{year_suffix}") if int(year_suffix) < 50 else int(f"19{year_suffix}")
    except:
        year = None
    filename = f"{ticker}_{year}_{form_type}_chunks.json" if year else None
    return filename, year

def process_directory(base_dir, output_dir, start_year=None, end_year=None, allowed_tickers=None):
    os.makedirs(output_dir, exist_ok=True)
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if not file.lower().endswith(".txt"):
                continue
            file_path = os.path.join(root, file)
            output_filename, year = construct_filename(file_path, base_dir)
            if not output_filename or year is None:
                continue
            ticker = output_filename.split("_")[0]
            if allowed_tickers and ticker not in allowed_tickers:
                print(f"Skipping {file_path} (ticker {ticker} not in allowed list)")
                continue
            if start_year and year < start_year:
                print(f"Skipping {file_path} (year {year} before {start_year})")
                continue
            if end_year and year > end_year:
                print(f"Skipping {file_path} (year {year} after {end_year})")
                continue
            output_path = os.path.join(output_dir, output_filename)
            if os.path.exists(output_path):
                print(f"Skipping {file_path} (already processed)")
                continue
            print(f"Processing {file_path} -> {output_filename}")
            try:
                process_10k_to_chunked_json(file_path, output_path)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

# === CONFIGURATION ===
# tickers = ["AAPL", "GOOGL", "NVDA", "ADBE", "ORCL"]
tickers = ["MSFT"]
base_directory = "sec"
output_directory = "sec_data"
start_year = 2020
end_year = 2024

# === EXECUTE ===
process_directory(base_directory, output_directory, start_year, end_year, allowed_tickers=tickers)
