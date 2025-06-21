import asyncio
import os
from pydantic import BaseModel, Field, HttpUrl
import requests
from urllib.parse import urlparse
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError
from bs4 import BeautifulSoup


class ArchiveDownloaderConfig(BaseModel):
    """
    Configuration model for the live site mapping crawler.
    """
    sitemap_file: str = Field(
        ...,
        description="The starting URL for the crawl. Must be a valid HTTP/HTTPS URL.",
        examples=["https://www.example.com"]
    )
    date_to_download: str = Field(
        ...,
        description="date to download (yyyy-MM, yyyy-MM-dd)"
    )
    output_dir: str = Field(
        ...,
        min_length=1,
        description="The path to the file where the sitemap will be saved.",
        examples=["example_sitemap.txt"]
    )
    
def create_path_from_url(base_output_dir: str, url: str) -> str:
    parsed_url = urlparse(url)
    path = parsed_url.path.lstrip('/')
    if not path:
        path = 'index'
    if '.' in os.path.basename(path) and len(os.path.basename(path).split('.')[-1]) < 5:
        path = os.path.dirname(path)
    full_path = os.path.join(base_output_dir, path)
    os.makedirs(full_path, exist_ok=True)
    return full_path

async def handle_cookie_consent(page):
    print("    -> Waiting for cookie consent banners...")
    strategies = [
        ("Bexley-style banner", "page", None, "div#ccc-notify button#ccc-notify-accept"),
        ("Iframe-based banner", "iframe", "iframe#cc_iframe", "button:has-text('Accept all cookies')"),
        ("Generic 'Accept all'", "page", None, "button:has-text('Accept all')"),
    ]
    for description, loc_type, frame_selector, button_selector in strategies:
        try:
            wait_timeout = 7000
            if loc_type == "page":
                button = page.locator(button_selector)
            else:
                frame_locator = page.frame_locator(frame_selector)
                button = frame_locator.locator(button_selector)
            await button.click(timeout=wait_timeout)
            print(f"    -> ✅ Success! Clicked button for '{description}'.")
            await page.wait_for_timeout(2000)
            return
        except (PlaywrightTimeoutError, Exception):
            pass
    print("    -> All cookie strategies attempted.")

def find_closest_snapshot_url(target_url: str, timestamp: str) -> str | None:
    api_url = (
        f"http://web.archive.org/cdx/search/cdx?url={target_url}&from={timestamp}&to={timestamp}"
        f"&output=json&filter=statuscode:200&limit=1"
    )
    try:
        # Add a retry mechanism for robustness
        for attempt in range(3):
            try:
                response = requests.get(api_url, timeout=30)
                response.raise_for_status()
                data = response.json()
                if len(data) > 1:
                    snapshot = data[1]
                    return f"https://web.archive.org/web/{snapshot[1]}/{snapshot[2]}"
                return None # No snapshot found
            except (requests.exceptions.ConnectionError, requests.exceptions.ReadTimeout) as e:
                print(f"    -> Network error on attempt {attempt + 1}: {e}. Retrying...")
                asyncio.sleep(5)
        print(f"    -> ❌ CDX API Error for {target_url} after multiple retries.")
        return None
    except Exception as e:
        print(f"    -> ❌ Unhandled CDX API Error for {target_url}: {e}")
    return None

async def download_page_artifacts(snapshot_url: str, save_path: str):
    """Fetches a single archived page and saves its artifacts."""
    print(f"  -> Downloading: {snapshot_url}")
    screenshot_path = os.path.join(save_path, "full_screenshot.png")
    text_path = os.path.join(save_path, "full_page.txt")

    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        try:
            await page.goto(snapshot_url, wait_until="domcontentloaded", timeout=90000)
            await page.evaluate('() => { const e = document.getElementById("wm-ipp-base"); if (e) e.style.display = "none"; }')
            await handle_cookie_consent(page)
            await page.wait_for_load_state("networkidle", timeout=60000)
            await page.screenshot(path=screenshot_path, full_page=True)
            html_content = await page.content()
            soup = BeautifulSoup(html_content, 'html.parser')
            with open(text_path, "w", encoding="utf-8") as f:
                f.write(soup.get_text(separator='\n', strip=True))
            print(f"    -> ✅ Artifacts saved to: {save_path}")
        except Exception as e:
            print(f"  -> ❌ Error processing {snapshot_url}: {e}")
        finally:
            await browser.close()

async def download_archive_from_sitemap(archiver_download_config: ArchiveDownloaderConfig):
    """
    Reads a sitemap file and downloads the archived version of each URL for a specific date.
    """
    sitemap_file = archiver_download_config.sitemap_file
    output_dir = archiver_download_config.output_dir
    date_to_download = archiver_download_config.date_to_download
    print(f"--- Starting Archive Downloader from Sitemap ---")
    print(f"  Sitemap: {sitemap_file}")
    print(f"  Date: {date_to_download}")
    print(f"  Output Dir: {output_dir}")

    try:
        with open(sitemap_file, 'r', encoding='utf-8') as f:
            urls_to_download = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"❌ Error: Sitemap file not found at '{sitemap_file}'.")
        print("Please run 'map_live_site.py' first to generate it.")
        return

    timestamp = date_to_download.replace('-', '')
    os.makedirs(output_dir, exist_ok=True)

    for url in urls_to_download:
        print(f"\nProcessing URL: {url}")
        snapshot_url = find_closest_snapshot_url(url, timestamp)
        if snapshot_url:
            save_path = create_path_from_url(output_dir, url)
            await download_page_artifacts(snapshot_url, save_path)
        else:
            print(f"  -> No snapshot found for this URL at {date_to_download}. Skipping.")

    print("\n--- Download Process Complete ---")

def main():
    config = {
        "sitemap_file": "bexley_sitemap.txt",
        "date": "2025-01",  # The single YYYY-MM timestamp you want to download
        "output_dir": "bexley_archive_2025-01"
    }
    archiver_download_config = ArchiveDownloaderConfig(**config)
    
    asyncio.run(download_archive_from_sitemap(archiver_download_config))

if __name__ == "__main__":
    main()