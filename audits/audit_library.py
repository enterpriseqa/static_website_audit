import asyncio
import base64
import difflib
import io
import json
import os
from pathlib import Path
from typing import Dict, Optional

from agent.agent_lib import invoke_with_retry_async, parse_llm_json_output,semantic_text_analyzer_chain,literal_text_analyzer_chain, visual_analyzer_chain
from agent.types import TextComparisonResult, TextDiffResult, VisualComparisonResult
from PIL import Image

import cv2
import numpy as np


def normalize_text(text: str) -> str:
    normalized_text = text.replace('\r\n', '\n')
    lines = normalized_text.split('\n')
    stripped_lines = [line.strip() for line in lines]
    non_empty_lines = [line for line in stripped_lines if line]
    return '\n'.join(non_empty_lines)

text_schema_json = TextComparisonResult.schema_json(indent=2)
visual_schema_json = VisualComparisonResult.schema_json(indent=2)
diff_schema_json = TextDiffResult.schema_json(indent=2)

MODEL_IMAGE_DIMENSION_LIMIT = 8000

async def analyze_semantic_text(prod_text: str, staging_text: str) -> Optional[TextComparisonResult]:
    """
    Analyzes the semantic difference between two blocks of text.
    Generated code

        
    Args:
        prod_text: The original (production) text.
        staging_text: The new (staging) text.
        
    Returns:
        A TextComparisonResult Pydantic model, or None on failure.
    """
    print("  [Task] Running Semantic Text Analysis...")
    normalized_prod = normalize_text(prod_text)
    normalized_staging = normalize_text(staging_text)

    if normalized_prod == normalized_staging:
        print("  -> Texts are identical after normalization. Skipping semantic LLM call.")
        return TextComparisonResult(SEMANTIC_DIFF_STATUS="IDENTICAL", SEMANTIC_DIFF_SUMMARY=[])


    
    raw_result = await invoke_with_retry_async(semantic_text_analyzer_chain, {
        "text_schema": text_schema_json,
        "text_prod": normalized_prod,
        "text_staging": normalized_staging
    }, retries = 15)
    return parse_llm_json_output(raw_result, TextComparisonResult)


async def analyze_literal_text_diff(prod_text: str, staging_text: str) -> Optional[TextDiffResult]:
    print("  [Task] Running Literal Text Analysis...")
    normalized_prod = normalize_text(prod_text)
    normalized_staging = normalize_text(staging_text)

    diff_output = generate_text_diff(normalized_prod, normalized_staging)

    if not diff_output:
        print("  -> `difflib` found no literal text differences.")
        return TextDiffResult(TEXT_DIFF_STATUS="IDENTICAL", TEXT_DIFF_SUMMARY=[])

    print("  -> `difflib` found literal text differences. Asking AI to summarize.")
    raw_result = await invoke_with_retry_async(literal_text_analyzer_chain, {
        "diff_schema": diff_schema_json,
        "diff_output": diff_output
    }, retries=15)
    return parse_llm_json_output(raw_result, TextDiffResult)

async def analyze_visual_difference(prod_image_path: str, staging_image_path: str, diff_image_path: str) -> Optional[VisualComparisonResult]:

    print("  [Task] Running Visual Analysis...")

    #pixel_differences_found = create_diff_image(prod_image_path, staging_image_path, diff_image_path)
    pixel_differences_found = create_structural_diff_image_v2(prod_image_path, staging_image_path, diff_image_path)

    if not pixel_differences_found:
        print("  -> Skipping visual AI analysis as images are pixel-identical.")
        return VisualComparisonResult(VISUAL_DIFF_STATUS="IDENTICAL", VISUAL_DIFF_SUMMARY=[])

    print("  -> Pixel differences detected. Preparing visual analysis with diff image...")
    prod_image_b64 = base64.b64encode(get_compliant_image_bytes(prod_image_path)).decode('utf-8')
    staging_image_b64 = base64.b64encode(get_compliant_image_bytes(staging_image_path)).decode('utf-8')
    diff_image_b64 = base64.b64encode(get_compliant_image_bytes(diff_image_path)).decode('utf-8')


    raw_result = await invoke_with_retry_async(visual_analyzer_chain, {
        "visual_schema": visual_schema_json,
        "diff_image_b64": diff_image_b64
    },retries = 15)
    return parse_llm_json_output(raw_result, VisualComparisonResult)

def generate_text_diff(text1: str, text2: str) -> str:
    diff = difflib.unified_diff(
        text1.splitlines(keepends=True),
        text2.splitlines(keepends=True),
        fromfile='expected',
        tofile='actual',
    )
    return ''.join(diff)

def get_compliant_image_bytes(image_path: str) -> bytes:
    with Image.open(image_path) as img:
        width, height = img.size
        if width > MODEL_IMAGE_DIMENSION_LIMIT or height > MODEL_IMAGE_DIMENSION_LIMIT:
            print(f"  -> Image {os.path.basename(image_path)} ({width}x{height}) exceeds limit. Resizing...")
            ratio = min(MODEL_IMAGE_DIMENSION_LIMIT / width, MODEL_IMAGE_DIMENSION_LIMIT / height)
            new_width, new_height = int(width * ratio), int(height * ratio)
            resized_img = img.resize((new_width, new_height), Image.LANCZOS)
            buffer = io.BytesIO()
            resized_img.save(buffer, format=img.format or 'PNG')
            return buffer.getvalue()
    with open(image_path, "rb") as f:
        return f.read()
    

def create_structural_diff_image_v2(image_path1: str, image_path2: str, diff_path: str, confidence_threshold=0.95) -> bool:
    """
    Compares two images using a structural, block-based approach (V2).
    This version uses a rectangular kernel to better isolate lines of text and
    prevent incorrect merging of vertical content blocks. It is more robust
    against layout shifts and minor content changes.

    Returns:
        True if significant differences were found, False otherwise.
    """
    print("  -> Performing structural image comparison V2 (line-based)...")
    try:
        img1_bgr = cv2.imread(image_path1)
        img2_bgr = cv2.imread(image_path2)

        if img1_bgr is None or img2_bgr is None:
            print("❌ Error: One or both images could not be read.")
            return True

        # --- 1. Pre-process and find content blocks (contours) in both images ---
        def get_content_blocks(image):
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
            
            # --- THE KEY IMPROVEMENT IS HERE ---
            # Use a rectangular kernel that is wide but not tall.
            # This connects words on a line, but not the lines themselves.
            rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 5))
            dilated = cv2.dilate(thresh, rect_kernel, iterations=2)
            
            contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Return bounding boxes of significant contours
            return [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > 200]

        blocks1 = get_content_blocks(img1_bgr)
        blocks2 = get_content_blocks(img2_bgr)

        # --- 2. Find all matching blocks using template matching ---
        # This is a more robust way to find pairs than the previous version.
        matched_pairs = []
        # Keep track of blocks in image 2 that have already been matched
        # to prevent one block from being matched multiple times.
        matched_in_img2 = [False] * len(blocks2)

        for i, (x1, y1, w1, h1) in enumerate(blocks1):
            template = img1_bgr[y1:y1+h1, x1:x1+w1]
            best_match_val = -1
            best_match_idx = -1

            for j, (x2, y2, w2, h2) in enumerate(blocks2):
                # Skip if this block in img2 is already part of a better match
                if matched_in_img2[j]:
                    continue
                
                # Search for the template in the second image
                target = img2_bgr[y2:y2+h2, x2:x2+w2]
                # Resize template to target to handle minor font rendering differences
                if template.shape != target.shape:
                    template_resized = cv2.resize(template, (w2, h2))
                else:
                    template_resized = template

                res = cv2.matchTemplate(target, template_resized, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, _ = cv2.minMaxLoc(res)

                if max_val > best_match_val:
                    best_match_val = max_val
                    best_match_idx = j

            if best_match_val > confidence_threshold:
                matched_pairs.append((i, best_match_idx))
                matched_in_img2[best_match_idx] = True

        # --- 3. Identify added and deleted blocks based on matches ---
        matched_indices_in_img1 = {pair[0] for pair in matched_pairs}
        matched_indices_in_img2 = {pair[1] for pair in matched_pairs}

        deleted_blocks = [blocks1[i] for i in range(len(blocks1)) if i not in matched_indices_in_img1]
        added_blocks = [blocks2[j] for j in range(len(blocks2)) if j not in matched_indices_in_img2]

        # --- 4. Generate the final diff image ---
        if not deleted_blocks and not added_blocks:
            print("  -> No significant structural differences found.")
            return False

        print(f"  -> Found {len(deleted_blocks)} changed/deleted and {len(added_blocks)} added blocks.")
        
        # Create a grayscale version of the new image to use as the background
        diff_visual = cv2.cvtColor(cv2.cvtColor(img2_bgr, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)

        # Highlight deleted blocks in RED on the grayscale background
        for x, y, w, h in deleted_blocks:
            cv2.rectangle(diff_visual, (x, y), (x + w, y + h), (0, 0, 255), 3) # Red for deleted/changed

        # Highlight added blocks in GREEN with their full color content
        for x, y, w, h in added_blocks:
            color_block = img2_bgr[y:y+h, x:x+w]
            diff_visual[y:y+h, x:x+w] = color_block
            cv2.rectangle(diff_visual, (x, y), (x + w, y + h), (0, 255, 0), 3) # Green for added

        cv2.imwrite(diff_path, diff_visual)
        return True

    except Exception as e:
        print(f"❌ Error creating structural diff image V2 with OpenCV: {e}")
        return True

def create_structural_diff_image(image_path1: str, image_path2: str, diff_path: str, confidence_threshold=0.95) -> bool:
    """
    Compares two images using a structural, block-based approach that is robust
    to layout shifts. It identifies content blocks in each image and highlights
    blocks that are added, deleted, or significantly changed.

    Returns:
        True if significant differences were found, False otherwise.
    """
    print("  -> Performing structural image comparison (robust to layout shifts)...")
    try:
        img1_bgr = cv2.imread(image_path1)
        img2_bgr = cv2.imread(image_path2)

        if img1_bgr is None or img2_bgr is None:
            print("❌ Error: One or both images could not be read.")
            return True # Fail safe

        # --- 1. Pre-process and find content blocks (contours) in both images ---
        def get_content_blocks(image):
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # Invert the image because findContours looks for white objects on a black background
            _, thresh = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
            # Dilate to connect text characters into larger blocks
            kernel = np.ones((15,15), np.uint8)
            dilated = cv2.dilate(thresh, kernel, iterations=2)
            contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # Filter out very small noise contours
            return [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > 500]

        blocks1 = get_content_blocks(img1_bgr)
        blocks2 = get_content_blocks(img2_bgr)
        
        img2_gray = cv2.cvtColor(img2_bgr, cv2.COLOR_BGR2GRAY) # For template matching

        # --- 2. Find blocks from image 1 that are missing or changed in image 2 ---
        missing_or_changed_blocks = []
        for x1, y1, w1, h1 in blocks1:
            template = img1_bgr[y1:y1+h1, x1:x1+w1]
            
            # Search for this block in the second image
            res = cv2.matchTemplate(img2_bgr, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(res)
            
            if max_val < confidence_threshold:
                missing_or_changed_blocks.append((x1, y1, w1, h1))

        # --- 3. Find blocks from image 2 that are new (added) ---
        added_blocks = []
        for x2, y2, w2, h2 in blocks2:
            template = img2_bgr[y2:y2+h2, x2:x2+w2]
            
            # Search for this new block in the first image
            res = cv2.matchTemplate(img1_bgr, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(res)
            
            if max_val < confidence_threshold:
                added_blocks.append((x2, y2, w2, h2))

        # --- 4. Generate the final diff image ---
        if not missing_or_changed_blocks and not added_blocks:
            print("  -> No significant structural differences found.")
            return False

        print(f"  -> Found {len(missing_or_changed_blocks)} changed/deleted and {len(added_blocks)} added blocks.")
        
        # Create a grayscale version of the new image to use as the background
        diff_visual = cv2.cvtColor(img2_gray, cv2.COLOR_GRAY2BGR)

        # Highlight changed/deleted blocks in RED on the original positions
        for x, y, w, h in missing_or_changed_blocks:
            # We can't show the original content, so we just draw a red box
            # on the grayscale background where it *used* to be.
            cv2.rectangle(diff_visual, (x, y), (x + w, y + h), (0, 0, 255), 3) # Red for deleted/changed

        # Highlight added blocks in GREEN with their full color content
        for x, y, w, h in added_blocks:
            color_block = img2_bgr[y:y+h, x:x+w]
            diff_visual[y:y+h, x:x+w] = color_block
            cv2.rectangle(diff_visual, (x, y), (x + w, y + h), (0, 255, 0), 3) # Green for added

        cv2.imwrite(diff_path, diff_visual)
        return True

    except Exception as e:
        print(f"❌ Error creating structural diff image with OpenCV: {e}")
        return True # Fail safe
    

async def run_full_comparison(prod_text_path: str, staging_text_path: str, prod_image_path: str, staging_image_path: str,
                              result_path) -> Dict:
    """
    Runs the full comparison pipeline by orchestrating the individual analyzers.
    """
    print("--- Starting Full Comparison Pipeline ---")
    with open(prod_text_path, "r", encoding="utf-8") as f: prod_text = f.read()
    with open(staging_text_path, "r", encoding="utf-8") as f: staging_text = f.read()

    diff_image_path = os.path.join(result_path, "diff.png")

    # --- Step 2: Concurrently run all analyzers ---
    print("  -> Launching all analysis tasks concurrently...")
    tasks = [
        analyze_semantic_text(prod_text, staging_text),
        analyze_literal_text_diff(prod_text, staging_text),
        analyze_visual_difference(prod_image_path, staging_image_path, diff_image_path)
    ]

    semantic_result, literal_result, visual_result = await asyncio.gather(*tasks)

    # --- Step 3: Validate and Aggregate Results ---
    if not all([semantic_result, literal_result, visual_result]):
        return {"status": "ERROR", "details": "One or more analysis components failed."}

    status_hierarchy = {"IDENTICAL": 0, "MINOR_DIFFERENCE": 1, "VAST_DIFFERENCE": 2}
    final_status_num = max(
        status_hierarchy[semantic_result.SEMANTIC_DIFF_STATUS],
        status_hierarchy[literal_result.TEXT_DIFF_STATUS],
        status_hierarchy[visual_result.VISUAL_DIFF_STATUS]
    )
    final_status = [k for k, v in status_hierarchy.items() if v == final_status_num][0]

    final_summary = []
    if semantic_result.SEMANTIC_DIFF_SUMMARY: final_summary.extend([f"[Semantic] {s}" for s in semantic_result.SEMANTIC_DIFF_SUMMARY])
    if literal_result.TEXT_DIFF_SUMMARY: final_summary.extend([f"[Literal Text] {s}" for s in literal_result.TEXT_DIFF_SUMMARY])
    if visual_result.VISUAL_DIFF_SUMMARY: final_summary.extend([f"[Visual] {s}" for s in visual_result.VISUAL_DIFF_SUMMARY])

    final_report = {
        "semantic_text_analysis": semantic_result.dict(),
        "literal_text_analysis": literal_result.dict(),
        "visual_analysis": visual_result.dict(),
        "final_aggregated_result": {
            "status": final_status,
            "DIFFERENCE_SUMMARY": final_summary or ["No differences detected across all analyses."]
        }
    }
    
    # Optional: Cleanup diff image
    # if os.path.exists(diff_image_path):
    #     os.remove(diff_image_path)
        
    return final_report

async def compare_downloaded_contents(base_path: str, result_path: str) -> dict:
    """Compares downloaded artifacts located in 'expected' and 'actual' subfolders."""
    print(f"--- Starting comparison of artifacts in: {base_path} ---")
    expected_path = os.path.join(base_path, "expected")
    actual_path = os.path.join(base_path, "actual")
    if not os.path.isdir(expected_path) or not os.path.isdir(actual_path):
        return {"error": "'expected' and/or 'actual' folder not found."}

    pages_to_compare = sorted(list(set(os.listdir(expected_path)) & set(os.listdir(actual_path))))
    full_report = {}

    for page_id in pages_to_compare:
        print(f"\n--- Comparing page: {page_id} ---")
        prod_text_path = os.path.join(expected_path, page_id, "full_page.txt")
        prod_image_path = os.path.join(expected_path, page_id, "full_screenshot.png")
        actual_text_path = os.path.join(actual_path, page_id, "full_page.txt")
        actual_image_path = os.path.join(actual_path, page_id, "full_screenshot.png")

        if not all(os.path.exists(p) for p in [prod_text_path, prod_image_path, actual_text_path, actual_image_path]):
            full_report[page_id] = {"status": "ERROR", "details": "Missing one or more artifact files."}
            continue
        
        full_report[page_id] = await run_full_comparison(
            prod_text_path,
            actual_text_path,
            prod_image_path,
            actual_image_path,
            result_path
        )

    return full_report




async def compare_folders_recursively(v1_path: str, v2_path: str, result_path: str) -> dict:
    """
    Compares two directory trees recursively, looking for common subdirectories
    containing 'full_page.txt' and 'full_screenshot.png' to run AI analysis on.
    """
    print(f"--- Starting Recursive Comparison ---")
    print(f"  V1 Path: {v1_path}")
    print(f"  V2 Path: {v2_path}")

    base_v1 = Path(v1_path)
    base_v2 = Path(v2_path)

    if not base_v1.is_dir() or not base_v2.is_dir():
        return {"error": "One or both base paths are not valid directories."}

    # 1. Find all subdirectories in v1 that contain the required files
    v1_pages = set()
    for txt_file in base_v1.rglob('full_page.txt'):
        if (txt_file.parent / 'full_screenshot.png').exists():
            # Store the relative path to the subdirectory
            v1_pages.add(txt_file.parent.relative_to(base_v1))

    # 2. Find all subdirectories in v2 that contain the required files
    v2_pages = set()
    for txt_file in base_v2.rglob('full_page.txt'):
        if (txt_file.parent / 'full_screenshot.png').exists():
            v2_pages.add(txt_file.parent.relative_to(base_v2))

    # 3. Determine which pages to compare, which are unique, etc.
    common_pages = sorted(list(v1_pages & v2_pages))
    only_in_v1 = sorted(list(v1_pages - v2_pages))
    only_in_v2 = sorted(list(v2_pages - v1_pages))

    print(f"\nFound {len(common_pages)} common pages to compare.")
    print(f"Found {len(only_in_v1)} pages only in V1.")
    print(f"Found {len(only_in_v2)} pages only in V2.")

    full_report = {
        "summary": {
            "common_pages_found": len(common_pages),
            "pages_only_in_v1": [str(p) for p in only_in_v1],
            "pages_only_in_v2": [str(p) for p in only_in_v2],
        },
        "page_reports": {}
    }

    # 4. Process and compare the common pages
    os.makedirs(result_path, exist_ok=True)
    full_result_path = Path(result_path) / ' full_result.json'

    for rel_path in common_pages:
        page_key = str(rel_path)
        print(f"\n--- Comparing Page: {page_key} ---")

        v1_dir = base_v1 / rel_path
        v2_dir = base_v2 / rel_path

        os.makedirs(result_path / rel_path, exist_ok=True)
        report_path = result_path / rel_path / 'comparison_result.txt'
        if (report_path.exists()):
            continue

        v1_text_path = v1_dir / 'full_page.txt'
        v1_image_path = v1_dir / 'full_screenshot.png'
        v2_text_path = v2_dir / 'full_page.txt'
        v2_image_path = v2_dir / 'full_screenshot.png'

        # This check is slightly redundant due to how we built the sets, but it's good practice
        if not all(p.exists() for p in [v1_text_path, v1_image_path, v2_text_path, v2_image_path]):
             full_report["page_reports"][page_key] = {
                "comparison_status": "SKIPPED",
                "details": "One or more artifact files were missing, despite the directory being common."
            }
             continue

        # Run the AI comparison
        comparison_result = await run_full_comparison(
            prod_text_path=v1_text_path,
            staging_text_path=v2_text_path,
            prod_image_path=str(v1_image_path),
            staging_image_path=str(v2_image_path),
            result_path=result_path
        )
        
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(comparison_result, f, indent=2)
            print(f"report saved to: {report_path}")
        
        
        print(report_path)
        with open(report_path, "r", encoding="utf-8") as f:
            comparison_result = json.load(f)
            full_report["page_reports"][page_key] = comparison_result['final_aggregated_result']['status']
        

    print("\n--- Comparison Finished ---")
    with open(full_result_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)
        print(f"report saved to: {full_report}")
    return full_report