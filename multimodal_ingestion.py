"""
multimodal_ingestion.py
Unified pipeline for ingesting text + images from PDFs into RAG system.
"""

from pathlib import Path
from langchain_core.documents import Document
from image_extractor import extract_images_from_directory
from image_captioner import caption_extracted_images
from caption_store import save_captions_from_extraction, CaptionStore
from ingestion import load_documents


def ingest_multimodal_documents(
    data_dir: str = "data",
    extract_images: bool = True,
    caption_model: str = "Salesforce/blip-image-captioning-large"
) -> list[Document]:
    """
    Ingest both text and images from PDFs.
    
    Parameters:
        data_dir: Directory containing PDF files
        extract_images: Whether to extract and caption images
        caption_model: BLIP-1 model for captioning
    
    Returns:
        List of Document objects (text + image captions)
    """
    print(f"\n{'='*70}")
    print("MULTIMODAL DOCUMENT INGESTION")
    print(f"{'='*70}\n")
    
    # Step 1: Load text documents (existing pipeline)
    print("STEP 1: Loading text documents...")
    text_docs = load_documents(data_dir)
    print(f"✓ Loaded {len(text_docs)} text chunks\n")
    
    # Step 2: Extract and caption images (if enabled)
    image_docs = []
    if extract_images:
        print("STEP 2: Extracting images from PDFs...")
        image_metadata = extract_images_from_directory(data_dir)
        
        if image_metadata:
            print("\nSTEP 3: Generating captions with BLIP-1...")
            captioned_images = caption_extracted_images(
                image_metadata,
                prompt=None,  # BLIP-1 works better without prompts
                model_name=caption_model
            )
            
            print("\nSTEP 4: Storing captions...")
            caption_store = save_captions_from_extraction(captioned_images)
            
            print("\nSTEP 5: Converting captions to documents...")
            # Convert captions to Document objects
            for caption_entry in captioned_images:
                image_name = Path(caption_entry['image_path']).name
                source_file = caption_entry.get('source_file', 'unknown')
                page_num = caption_entry.get('page_num', 'unknown')
                caption = caption_entry.get('caption', '')
                
                # Create document content
                content = f"[IMAGE: {image_name}]\n"
                content += f"Description: {caption}"
                
                # Create Document object
                doc = Document(
                    page_content=content,
                    metadata={
                        'source': source_file,
                        'page': page_num,
                        'type': 'image_caption',
                        'image_path': caption_entry['image_path']
                    }
                )
                image_docs.append(doc)
            
            print(f"✓ Created {len(image_docs)} image documents\n")
        else:
            print("✓ No images found in PDFs\n")
    
    # Step 3: Combine text and image documents
    all_docs = text_docs + image_docs
    
    print(f"{'='*70}")
    print("INGESTION SUMMARY:")
    print(f"{'='*70}")
    print(f"Text documents: {len(text_docs)}")
    print(f"Image documents: {len(image_docs)}")
    print(f"Total documents: {len(all_docs)}")
    print(f"{'='*70}\n")
    
    return all_docs


def build_multimodal_vector_store(
    data_dir: str = "data",
    extract_images: bool = True,
    index_dir: str = "faiss_index"
):
    """
    Build FAISS vector store with text + image captions.
    
    Parameters:
        data_dir: Directory containing PDF files
        extract_images: Whether to include images
        index_dir: Directory to save FAISS index
    """
    from vector_store import get_embeddings
    from langchain_community.vectorstores import FAISS
    import os
    
    print(f"\n{'='*70}")
    print("BUILDING MULTIMODAL VECTOR STORE")
    print(f"{'='*70}\n")
    
    # Ingest documents (using BLIP-1 model - 990MB)
    docs = ingest_multimodal_documents(
        data_dir, 
        extract_images,
        caption_model="Salesforce/blip-image-captioning-large"  # 990MB model
    )
    
    if not docs:
        raise ValueError(f"No documents found in '{data_dir}'")
    
    # Build FAISS index
    print("Building FAISS index with embeddings...")
    embeddings = get_embeddings()
    vector_store = FAISS.from_documents(docs, embeddings)
    
    # Save to disk
    os.makedirs(index_dir, exist_ok=True)
    vector_store.save_local(index_dir)
    
    print(f"\n{'='*70}")
    print("✓ MULTIMODAL VECTOR STORE BUILT SUCCESSFULLY")
    print(f"{'='*70}")
    print(f"Index saved to: {index_dir}/")
    print(f"Total documents indexed: {len(docs)}")
    print(f"  - Text chunks: {sum(1 for d in docs if d.metadata.get('type') != 'image_caption')}")
    print(f"  - Image captions: {sum(1 for d in docs if d.metadata.get('type') == 'image_caption')}")
    print(f"{'='*70}\n")
    
    return vector_store


# ─────────────────────────────────────────────────────────────────────────────
# CLI Usage
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Build multimodal vector store with text + images"
    )
    parser.add_argument("--data-dir", default="data", help="Directory containing PDFs")
    parser.add_argument("--no-images", action="store_true", help="Skip image extraction")
    parser.add_argument("--index-dir", default="faiss_index", help="Output directory for index")
    parser.add_argument("--model", default="Salesforce/blip-image-captioning-large", help="BLIP-1 model (default: 990MB large model)")
    
    args = parser.parse_args()
    
    # Build vector store
    build_multimodal_vector_store(
        data_dir=args.data_dir,
        extract_images=not args.no_images,
        index_dir=args.index_dir
    )
