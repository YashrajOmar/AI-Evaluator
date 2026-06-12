"""
image_extractor.py
Extract images from PDF/PPTX files for multimodal RAG.
"""

import os
import fitz  # PyMuPDF
from pathlib import Path
from PIL import Image
import io


def extract_images_from_pdf(pdf_path: str, output_dir: str = "extracted_images") -> list:
    """
    Extract all images from a PDF file.
    
    Parameters:
        pdf_path: Path to PDF file
        output_dir: Directory to save extracted images
    
    Returns:
        List of dicts with image info: {
            'image_path': str,
            'page_num': int,
            'source_file': str,
            'image_index': int
        }
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    pdf_name = Path(pdf_path).stem
    pdf_output_dir = os.path.join(output_dir, pdf_name)
    os.makedirs(pdf_output_dir, exist_ok=True)
    
    # Open PDF
    doc = fitz.open(pdf_path)
    
    extracted_images = []
    image_count = 0
    
    print(f"Processing: {pdf_path}")
    print(f"Total pages: {len(doc)}")
    
    # Iterate through pages
    for page_num in range(len(doc)):
        page = doc[page_num]
        
        # Get images on this page
        image_list = page.get_images(full=True)
        
        if image_list:
            print(f"  Page {page_num + 1}: Found {len(image_list)} images")
        
        for img_index, img in enumerate(image_list):
            try:
                # Get image XREF
                xref = img[0]
                
                # Extract image
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                
                # Save image
                image_filename = f"page{page_num + 1}_img{img_index + 1}.{image_ext}"
                image_path = os.path.join(pdf_output_dir, image_filename)
                
                with open(image_path, "wb") as img_file:
                    img_file.write(image_bytes)
                
                # Store metadata
                extracted_images.append({
                    'image_path': image_path,
                    'page_num': page_num + 1,
                    'source_file': pdf_path,
                    'image_index': image_count
                })
                
                image_count += 1
                
            except Exception as e:
                print(f"    Error extracting image {img_index} from page {page_num + 1}: {e}")
    
    doc.close()
    
    print(f"✓ Extracted {image_count} images from {pdf_path}")
    return extracted_images


def extract_images_from_directory(data_dir: str = "data", output_dir: str = "extracted_images") -> list:
    """
    Extract images from all PDFs in a directory.
    
    Parameters:
        data_dir: Directory containing PDF files
        output_dir: Directory to save extracted images
    
    Returns:
        List of all extracted image metadata
    """
    all_images = []
    
    # Find all PDF files
    pdf_files = list(Path(data_dir).glob("*.pdf"))
    
    if not pdf_files:
        print(f"No PDF files found in {data_dir}")
        return all_images
    
    print(f"\n{'='*60}")
    print(f"Found {len(pdf_files)} PDF files")
    print(f"{'='*60}\n")
    
    # Extract images from each PDF
    for pdf_path in pdf_files:
        images = extract_images_from_pdf(str(pdf_path), output_dir)
        all_images.extend(images)
    
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Total PDFs processed: {len(pdf_files)}")
    print(f"Total images extracted: {len(all_images)}")
    print(f"Images saved to: {output_dir}/")
    print(f"{'='*60}\n")
    
    return all_images


def get_image_info(image_path: str) -> dict:
    """
    Get basic information about an image.
    
    Parameters:
        image_path: Path to image file
    
    Returns:
        Dict with image info (size, format, mode)
    """
    try:
        with Image.open(image_path) as img:
            return {
                'size': img.size,
                'format': img.format,
                'mode': img.mode,
                'width': img.width,
                'height': img.height
            }
    except Exception as e:
        return {'error': str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# CLI Usage
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract images from PDFs")
    parser.add_argument("--data-dir", default="data", help="Directory containing PDFs")
    parser.add_argument("--output-dir", default="extracted_images", help="Output directory for images")
    
    args = parser.parse_args()
    
    # Extract images
    images = extract_images_from_directory(args.data_dir, args.output_dir)
    
    # Show sample image info
    if images:
        print("\nSample image info:")
        sample = images[0]
        print(f"  Path: {sample['image_path']}")
        print(f"  Page: {sample['page_num']}")
        print(f"  Source: {sample['source_file']}")
        
        info = get_image_info(sample['image_path'])
        print(f"  Size: {info.get('width')}x{info.get('height')}")
        print(f"  Format: {info.get('format')}")
