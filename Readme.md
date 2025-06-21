# 🕵️‍♂️ STATIC_WEBSITE_AUDIT

A Python-based tool to audit static websites by crawling live sites or downloading from the Web Archive and comparing content between two versions. Supports semantic text, literal HTML, and visual (image) comparisons.

---

## 📌 Features

- Crawl and extract HTML from live websites or from the Web Archive (Wayback Machine)
- Compare two versions of the same page or site:
  - **Literal differences** (HTML structure)
  - **Semantic differences** (textual meaning)
  - **Visual differences** (screenshots/images)

- Use cases include:
  - Verifying staging vs production deployments
  - Validating archival snapshots vs current state
  - Tracking website evolution over time

---

## 🛠️ Quickstart

### Prerequisites

- Python 3.8+
- Chrome or Chromium browser (for screenshot comparison)
- `pip` for installing dependencies
- Bedrock / Claude, Google Vertex / Genai / OpenAI (keys value from environment)

### Installation

```bash
git clone https://github.com/enterpriseqa/static_website_audit.git
cd static_website_audit
pip install -r requirements.txt
```

## Usage

1. Compare two versions of the webpage,

```
import asyncio
from audits.audit_library import download_and_compare
from web_archive.webpage_download.webpage_download import get_webpage_data_async


if __name__ == "__main__":
    asyncio.run(download_and_compare("https://example.com/v1", "https://example.com/v2"))
```

2. Audit via Web Archive

    Look at : examples/website_archive_audit.py


### Results & Reporting

1. Full Results
```
{
  "summary": {
    "common_pages_found": 3,
    "pages_only_in_v1": [
      "bexley-business-employment/invoices/pay-invoice",
      "consultations",
      "discover-bexley",
    ],
    "pages_only_in_v2": [
      "about-council/democracy-and-elections",
      "about-council/feedback",
    ]
  },
  "page_reports": {
    "about-council": "MINOR_DIFFERENCE",
    "index": "VAST_DIFFERENCE",
    "services/benefits-and-financial-help/benefits/help-rent-or-council-tax": "MINOR_DIFFERENCE",
    "services/children-young-people-and-families": "MINOR_DIFFERENCE",
  }
}
```

### Acknowledgments

Web Archive (Wayback Machine)

BeautifulSoup, difflib

PIL, OpenCV for visual diffing

Langchain, AWS Bedrock, Google GenAI, Google Vertex, OpenAI