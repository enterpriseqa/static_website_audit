
import base64
from collections import deque
import io
import os
from typing import List
from urllib.parse import urljoin, urlparse
from playwright.async_api import async_playwright, Error, Locator
from playwright_stealth import stealth_async
from PIL import Image

async def find_dynamic_elements_async(page) -> List[Locator]:
    """
    Analyzes a webpage to find likely dynamic elements using heuristics.

    Args:
        page: An active Playwright Page object.

    Returns:
        A list of Playwright Locator objects to be used for masking.
    """
    print("--- Starting dynamic element detection ---")
    dynamic_locators = []
    
    # --- Heuristic 1: Two-Load Comparison ---
    print("Heuristic 1: Comparing image sources across two page loads...")
    try:
        # First load
        await page.goto(page.url, wait_until="networkidle")
        initial_image_srcs = await page.evaluate("Array.from(document.querySelectorAll('img')).map(img => img.src)")
        
        # Second load
        await page.reload(wait_until="networkidle")
        reloaded_image_srcs = await page.evaluate("Array.from(document.querySelectorAll('img')).map(img => img.src)")

        # Find sources that are different
        initial_set = set(initial_image_srcs)
        reloaded_set = set(reloaded_image_srcs)
        
        changed_srcs = (initial_set - reloaded_set).union(reloaded_set - initial_set)

        if changed_srcs:
            print(f"Found {len(changed_srcs)} image(s) with changed sources.")
            for src in changed_srcs:
                # Find the locator for the image with the changed src
                # We check both initial and reloaded states to find the element
                locator = page.locator(f'img[src="{src}"]')
                if await locator.count() > 0:
                    dynamic_locators.append(locator)
        else:
            print("No images changed source on reload.")

    except Error as e:
        print(f"Could not perform two-load comparison: {e}")

    # --- Heuristic 2: Structural Carousel/Slider Detection ---
    print("Heuristic 2: Searching for common slider/carousel class names...")
    try:
        # This is a simple version. It can be expanded with more keywords.
        carousel_selector = '[class*="carousel"], [class*="slider"], [class*="swiper"]'
        carousel_locators = page.locator(carousel_selector)
        count = await carousel_locators.count()
        if count > 0:
            print(f"Found {count} potential carousel/slider container(s).")
            for i in range(count):
                dynamic_locators.append(carousel_locators.nth(i))
        else:
            print("No common carousel/slider structures found.")
    except Error as e:
        print(f"Could not perform structural search: {e}")
        
    print(f"--- Dynamic element detection finished. Found {len(dynamic_locators)} elements to mask. ---")
    return dynamic_locators

user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36"


MODEL_IMAGE_DIMENSION_LIMIT = 8000

def resize_image_if_needed(image_path: str) -> str:
    """
    Checks if an image exceeds Bedrock's dimension limits and resizes it if necessary.

    Args:
        image_path: The path to the original image.

    Returns:
        The path to the compliant image (either the original or a new temporary one).
    """
    with Image.open(image_path) as img:
        width, height = img.size
        
        if width > MODEL_IMAGE_DIMENSION_LIMIT or height > MODEL_IMAGE_DIMENSION_LIMIT:
            print(f"Image {os.path.basename(image_path)} ({width}x{height}) exceeds {MODEL_IMAGE_DIMENSION_LIMIT}px limit. Resizing...")
            
            # Calculate new dimensions while preserving aspect ratio
            if width > height:
                new_width = MODEL_IMAGE_DIMENSION_LIMIT
                new_height = int(height * (new_width / width))
            else:
                new_height = MODEL_IMAGE_DIMENSION_LIMIT
                new_width = int(width * (new_height / height))
            
            resized_img = img.resize((new_width, new_height), Image.LANCZOS)
            
            # Save to a temporary path
            original_dir = os.path.dirname(image_path)
            original_name, original_ext = os.path.splitext(os.path.basename(image_path))
            temp_path = os.path.join(original_dir, f"{original_name}_resized{original_ext}")
            
            resized_img.save(temp_path)
            
            # Save the resized image to an in-memory buffer
            buffer = io.BytesIO()
            # The format needs to be specified for the buffer, e.g., 'PNG'
            resized_img.save(buffer, format=img.format or 'PNG')
            print("Image resized in memory.")
            
            # Return the bytes from the buffer           
            print(f"Resized image saved to temporary path: {temp_path}")
            return (temp_path, buffer.getvalue())
            
    # If no resizing was needed, return the original path
    return (image_path, None)

async def get_webpage_data_async(url: str, storage_path) -> tuple[str, str] | tuple[None, None]:
    """
    Asynchronously navigates to a URL, takes a full-page screenshot, 
    and extracts all visible text.

    Args:
        url: The URL of the webpage to process.

    Returns:
        A tuple containing:
        - The base64 encoded string of the full-page screenshot (PNG).
        - The extracted text content of the page as a single string.
        Returns (None, None) if navigation or processing fails.
    """
    image_path = f"{storage_path}/full_screenshot.png"
    text_path = f"{storage_path}/full_page.txt"
    try:
        async with async_playwright() as p:
            # Launch a browser. Chromium is a good default.
            browser = await p.chromium.launch(headless=True)
            
            context = await browser.new_context(
                user_agent=user_agent,
                viewport={'width': 1920, 'height': 1080}
            )
            page = await context.new_page()

            await stealth_async(page)
            
            print(f"Navigating to {url}...")
            # Navigate to the page. 'networkidle' waits for network activity to cease.
            # We use a generous timeout of 60 seconds.
            await page.goto(url, wait_until="networkidle", timeout=60000)
            print("Page loaded successfully.")

            await page.wait_for_selector('body', state='attached', timeout=15000)
            
            print("Body element is ready.")
            
            elements_to_mask = await find_dynamic_elements_async(page)
            
            # --- STABILITY WORKFLOW ---
            # 1. Scroll to the bottom to trigger all lazy-loaded content
            print("Scrolling to bottom to trigger lazy loading...")
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await page.wait_for_timeout(2000) # Wait for lazy-loaded images to render

            # 2. Scroll back to the top to reset dynamic UI elements (like sticky headers)
            print("Scrolling back to top to ensure a stable state...")
            await page.evaluate("window.scrollTo(0, 0)")
            
            # 3. Wait for the page to be visually "at rest" after scrolling up
            await page.wait_for_timeout(2000)
            # --- END STABILITY WORKFLOW ---

            # 1. Take a full-page screenshot and get the raw bytes
            print("Taking full-page screenshot...")
            screenshot_bytes = await page.screenshot(
                path=image_path,
                full_page=True,
                mask=elements_to_mask # Use the detected elements for masking
            )
            
            (resized_image_path, resized_image_bytes) = resize_image_if_needed(image_path)
            print(f"resized image path, original image path: {resized_image_path}, {image_path}")
            
            if (resized_image_bytes is not None):
                print("Using resized image")
                screenshot_bytes = resized_image_bytes


            # Encode the bytes into a base64 string for easy transport
            screenshot_base64 = base64.b64encode(screenshot_bytes).decode('utf-8')
            print(f"Screenshot captured and encoded to base64 (length: {len(screenshot_base64)}).")

            # 2. Extract the visible text content from the page
            print("Extracting text content...")
            # page.evaluate() is an awaitable coroutine in the async API
            text_content = await page.evaluate("document.body.innerText")
            print(f"Text extracted (length: {len(text_content)} characters).")

        print(f"Saving text content to {text_path}...")
        with open(text_path, "w", encoding="utf-8") as f:
            f.write(text_content)
            
            # Clean up and close the browser
            await browser.close()

            return screenshot_base64, text_content

    except Error as e:
        print(f"An error occurred with Playwright: {e}")
        return None, None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None, None
    
async def crawl_site_async(base_url: str, max_depth: int) -> Set[str]:
    """
    Crawls a website starting from a base URL to a maximum depth.

    Args:
        base_url: The starting URL to crawl (e.g., "https://www.example.com").
        max_depth: The maximum number of levels to crawl (0 is just the base page).

    Returns:
        A set of unique relative paths found on the site.
    """
    print(f"--- Starting crawl of {base_url} up to depth {max_depth} ---")
    urls_to_visit = deque([(base_url, 0)])
    visited_urls = set()
    found_paths = set()
    
    # Normalize base_url to ensure correct 'startswith' check
    parsed_base_url = urlparse(base_url)
    base_domain = f"{parsed_base_url.scheme}://{parsed_base_url.netloc}"

    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()

        while urls_to_visit:
            current_url, current_depth = urls_to_visit.popleft()

            if current_url in visited_urls or current_depth > max_depth:
                continue

            try:
                print(f"Crawling (depth {current_depth}): {current_url}")
                await page.goto(current_url, wait_until="domcontentloaded", timeout=30000)
                visited_urls.add(current_url)
                
                # Add the found path to our set
                relative_path = urlparse(current_url).path
                if not relative_path: relative_path = "/"
                found_paths.add(relative_path)

                # Find all links on the page
                links = await page.eval_on_selector_all('a', 'elements => elements.map(el => el.href)')
                
                for link in links:
                    # Resolve relative links (e.g., "/about") to absolute URLs
                    absolute_link = urljoin(base_url, link)
                    
                    # Normalize the link
                    parsed_link = urlparse(absolute_link)
                    clean_link = f"{parsed_link.scheme}://{parsed_link.netloc}{parsed_link.path}"

                    if clean_link.startswith(base_domain) and clean_link not in visited_urls:
                        urls_to_visit.append((clean_link, current_depth + 1))

            except Error as e:
                print(f"Could not process URL {current_url}: {e}")
        
        await browser.close()

    print(f"--- Crawl finished. Found {len(found_paths)} unique pages. ---")
    return found_paths    