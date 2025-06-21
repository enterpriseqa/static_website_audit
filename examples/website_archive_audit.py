import asyncio
from audits.audit_library import compare_folders_recursively
from web_archive.extract_live_site_urls import SiteMapConfig, crawl_and_extract_links
from web_archive.webarchive_downloader import ArchiveDownloaderConfig, download_archive_from_sitemap


async def download_and_audit_archived_websites(sitemap_config: SiteMapConfig, start_date_download_config: ArchiveDownloaderConfig,
                                               end_date_download_config: ArchiveDownloaderConfig, result_path: str):
        await crawl_and_extract_links(sitemap_config)
        await download_archive_from_sitemap(start_date_download_config)
        await download_archive_from_sitemap(end_date_download_config)
        await compare_folders_recursively(start_date_download_config.output_dir, end_date_download_config.output_dir, result_path)


if __name__ == "__main__":
    archive_process_name = "bexley_archive"
    base_url = "https://www.bexley.gov.uk"
    url_file_list = "bexley_sitemap.txt"
    start_date_to_download = "2025-03"
    v1_config = {
        "start_url": base_url,
        "max_depth": 1,
        "output_file": url_file_list
    }
    log_base_apth = f"logs/"
    
    result_path = f"logs/{archive_process_name}"
    sitemap_config = SiteMapConfig(**v1_config)
    start_date = "2025-03"
    end_date = "2025-05"
    start_date_config = {
        "sitemap_file": "bexley_sitemap.txt",
        "date_to_download":start_date,
        "output_dir": f"{result_path}/{start_date}"
    }
    archiver_download_start_date_config = ArchiveDownloaderConfig(**start_date_config)
    end_date_config = {
        "sitemap_file": "bexley_sitemap.txt",
        "date_to_download": end_date, 
        "output_dir": f"{result_path}/{end_date}"
    }
    archiver_download_end_date_config = ArchiveDownloaderConfig(**end_date_config)
    asyncio.run(download_and_audit_archived_websites(sitemap_config, archiver_download_start_date_config, archiver_download_end_date_config,
                                                     result_path))
