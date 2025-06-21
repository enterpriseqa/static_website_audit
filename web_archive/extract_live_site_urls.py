import asyncio
import os
from urllib.parse import urlparse, urljoin
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field


class SiteMapConfig(BaseModel):
    """
    Configuration model for the live site mapping crawler.
    """
    start_url: str = Field(
        ...,
        description="The starting URL for the crawl. Must be a valid HTTP/HTTPS URL.",
        examples=["https://www.example.com"]
    )
    max_depth: int = Field(
        default=1,
        ge=0,  # ge=0 means the value must be greater than or equal to 0
        description="Crawl depth. 0=homepage only, 1=homepage + its links, etc."
    )
    output_file: str = Field(
        ...,
        min_length=1,
        description="The path to the file where the sitemap will be saved.",
        examples=["example_sitemap.txt"]
    )
    
def get_base_domain(url: str) -> str:
    return urlparse(url).netloc

async def crawl_and_extract_links(config: SiteMapConfig):
    start_url = config.start_url
    max_depth = config.max_depth
    output_file = config.output_file
    
    """
    Internal crawling logic. Takes the validated config model as input.
    """
    print(f"--- Starting Live Site Mapper ---")
    
    print(f"  Start URL: {start_url}")
    print(f"  Max Depth: {max_depth}")

    base_domain = get_base_domain(start_url)
    queue = asyncio.Queue()
    visited_urls = set()

    await queue.put((start_url, 0))
    visited_urls.add(start_url)

    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()

        while not queue.empty():
            current_url, current_depth = await queue.get()

            if current_depth > max_depth:
                continue

            print(f"Crawling [Depth {current_depth}]: {current_url}")

            try:
                await page.goto(current_url, wait_until="domcontentloaded", timeout=60000)
                html_content = await page.content()
                soup = BeautifulSoup(html_content, 'html.parser')

                if current_depth < max_depth:
                    for a_tag in soup.find_all('a', href=True):
                        href = a_tag['href']
                        if not href or any(href.startswith(p) for p in ['mailto:', 'tel:', 'javascript:', '#']):
                            continue
                        
                        absolute_url = urljoin(current_url, href)
                        parsed_url = urlparse(absolute_url)
                        
                        # Normalize URL (remove query params and fragments)
                        normalized_url = parsed_url._replace(query="", fragment="").geturl()

                        if get_base_domain(normalized_url) == base_domain and normalized_url not in visited_urls:
                            visited_urls.add(normalized_url)
                            await queue.put((normalized_url, current_depth + 1))
                            print(f"  -> Found: {normalized_url}")

            except Exception as e:
                print(f"  -> ❌ Error crawling {current_url}: {e}")
        
        await browser.close()

    print(f"\n--- Crawl Finished ---")
    print(f"Found {len(visited_urls)} unique URLs.")

    # Save the list of URLs to the output file
    with open(output_file, 'w', encoding='utf-8') as f:
        for url in sorted(list(visited_urls)):
            f.write(url + '\n')
    
    print(f"✅ Sitemap saved to: {output_file}")

def main():
    config = {
        "start_url": "https://www.bexley.gov.uk",
        "max_depth": 1,
        "output_file": "bexley_sitemap.txt"
    }
    sitemap_config = SiteMapConfig(**config)
    
    asyncio.run(crawl_and_extract_links(sitemap_config))

if __name__ == "__main__":
    main()