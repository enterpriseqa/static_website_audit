import asyncio
import os
import random
import string

from audits.audit_library import compare_folders_recursively
from web_archive.webpage_download.webpage_download import get_webpage_data_async


def generate_random_prefix(length=8):
    """Generates a random string of characters and digits for a filename prefix."""
    characters = string.ascii_letters + string.digits
    return ''.join(random.choice(characters) for _ in range(length))


async def download_contents(old_url, new_url, old_log_path, new_log_path):
    os.makedirs(old_log_path, exist_ok=True)
    os.makedirs(new_log_path, exist_ok=True)
    await get_webpage_data_async(old_url,f"{old_log_path}/" )   
    await get_webpage_data_async(new_url,f"{new_log_path}/" )
    return (old_log_path, new_log_path)

if __name__ == "__main__":
    random_prefix = generate_random_prefix(5)
    log_base_apth = f"logs/{random_prefix}"
    old_log_path = f"{log_base_apth}/old/"
    new_log_path = f"{log_base_apth}/new/"
    result_path = f"{log_base_apth}/result"
    (expected_log_path, actual_log_path) = asyncio.run(download_contents("https://valliapanr.wordpress.com/main-page-v2/",
                                                                         "https://valliapanr.wordpress.com/main-page-v3/",
                                                                         old_log_path, new_log_path))
    asyncio.run(compare_folders_recursively(expected_log_path, actual_log_path, result_path))
