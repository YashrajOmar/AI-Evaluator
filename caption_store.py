
"""
caption_store.py
Store image captions as structured data for RAG pipeline integration.
"""

import json
import os
from pathlib import Path
from datetime import datetime


class CaptionStore:
    """
    Manage storage and retrieval of image captions.
    """
    
    def __init__(self, store_path: str = "image_captions.json"):
        """
        Initialize caption store.
        
        Parameters:
            store_path: Path to JSON file for storing captions
        """
        self.store_path = store_path
        self.captions = self._load()
    
    def _load(self) -> dict:
        """Load existing captions from disk."""
        if os.path.exists(self.store_path):
            with open(self.store_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {
            'metadata': {
                'created': datetime.now().isoformat(),
                'last_updated': datetime.now().isoformat(),
                'total_images': 0
            },
            'captions': []
        }
    
    def save(self):
        """Save captions to disk."""
        self.captions['metadata']['last_updated'] = datetime.now().isoformat()
        self.captions['metadata']['total_images'] = len(self.captions['captions'])
        
        with open(self.store_path, 'w', encoding='utf-8') as f:
            json.dump(self.captions, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Saved {len(self.captions['captions'])} captions to {self.store_path}")
    
    def add_caption(self, image_path: str, caption: str, metadata: dict = None):
        """
        Add a single caption.
        
        Parameters:
            image_path: Path to image file
            caption: Generated caption text
            metadata: Optional dict with page_num, source_file, etc.
        """
        entry = {
            'image_path': image_path,
            'caption': caption,
            'metadata': metadata or {}
        }
        self.captions['captions'].append(entry)
    
    def add_captions_batch(self, caption_list: list):
        """
        Add multiple captions at once.
        
        Parameters:
            caption_list: List of dicts with image_path, caption, and optional metadata
        """
        for item in caption_list:
            self.add_caption(
                image_path=item.get('image_path'),
                caption=item.get('caption'),
                metadata={
                    'page_num': item.get('page_num'),
                    'source_file': item.get('source_file'),
                    'image_index': item.get('image_index')
                }
            )
        
        print(f"✓ Added {len(caption_list)} captions to store")
    
    def get_all_captions(self) -> list:
        """Get all stored captions."""
        return self.captions['captions']
    
    def get_captions_by_source(self, source_file: str) -> list:
        """Get captions from a specific source file."""
        return [
            c for c in self.captions['captions']
            if c.get('metadata', {}).get('source_file') == source_file
        ]
    
    def export_for_rag(self) -> list:
        """
        Export captions in format suitable for RAG pipeline.
        
        Returns:
            List of dicts with 'content' and 'metadata' for each caption
        """
        rag_docs = []
        
        for caption_entry in self.captions['captions']:
            # Format caption as document
            image_name = Path(caption_entry['image_path']).name
            source_file = caption_entry.get('metadata', {}).get('source_file', 'unknown')
            page_num = caption_entry.get('metadata', {}).get('page_num', 'unknown')
            
            # Create rich text content
            content = f"[IMAGE: {image_name}]\n"
            content += f"Source: {source_file}, Page: {page_num}\n"
            content += f"Description: {caption_entry['caption']}"
            
            rag_docs.append({
                'content': content,
                'metadata': {
                    'source': source_file,
                    'page': page_num,
                    'type': 'image_caption',
                    'image_path': caption_entry['image_path']
                }
            })
        
        return rag_docs
    
    def get_stats(self) -> dict:
        """Get statistics about stored captions."""
        captions = self.captions['captions']
        
        # Count by source
        sources = {}
        for c in captions:
            src = c.get('metadata', {}).get('source_file', 'unknown')
            sources[src] = sources.get(src, 0) + 1
        
        return {
            'total_captions': len(captions),
            'sources': sources,
            'created': self.captions['metadata'].get('created'),
            'last_updated': self.captions['metadata'].get('last_updated')
        }


def save_captions_from_extraction(
    image_metadata_with_captions: list,
    store_path: str = "image_captions.json"
) -> CaptionStore:
    """
    Save captions from image extraction + captioning pipeline.
    
    Parameters:
        image_metadata_with_captions: Output from caption_extracted_images()
        store_path: Path to save JSON file
    
    Returns:
        CaptionStore instance
    """
    store = CaptionStore(store_path)
    store.add_captions_batch(image_metadata_with_captions)
    store.save()
    
    # Print stats
    stats = store.get_stats()
    print(f"\n{'='*60}")
    print("CAPTION STORE STATISTICS:")
    print(f"{'='*60}")
    print(f"Total captions: {stats['total_captions']}")
    print(f"Sources: {len(stats['sources'])}")
    for src, count in stats['sources'].items():
        print(f"  - {Path(src).name}: {count} images")
    print(f"{'='*60}\n")
    
    return store


# ─────────────────────────────────────────────────────────────────────────────
# CLI Usage
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Manage image caption storage")
    parser.add_argument("--store", default="image_captions.json", help="Caption store file")
    parser.add_argument("--show-stats", action="store_true", help="Show statistics")
    parser.add_argument("--export-rag", action="store_true", help="Export for RAG pipeline")
    
    args = parser.parse_args()
    
    store = CaptionStore(args.store)
    
    if args.show_stats:
        stats = store.get_stats()
        print(f"\n{'='*60}")
        print("CAPTION STORE STATISTICS:")
        print(f"{'='*60}")
        print(f"Total captions: {stats['total_captions']}")
        print(f"Created: {stats['created']}")
        print(f"Last updated: {stats['last_updated']}")
        print(f"\nSources:")
        for src, count in stats['sources'].items():
            print(f"  - {Path(src).name}: {count} images")
        print(f"{'='*60}\n")
    
    if args.export_rag:
        rag_docs = store.export_for_rag()
        print(f"\n{'='*60}")
        print("RAG EXPORT:")
        print(f"{'='*60}")
        print(f"Exported {len(rag_docs)} documents")
        
        if rag_docs:
            print("\nSample document:")
            print("-" * 60)
            print(rag_docs[0]['content'])
            
            print("-" * 60)
            print(f"Metadata: {rag_docs[0]['metadata']}")
        
        print(f"{'='*60}\n")
    
    if not args.show_stats and not args.export_rag:
        print("Caption store initialized")
        print(f"Store file: {args.store}")
        print("\nUse --show-stats to view statistics")
        print("Use --export-rag to export for RAG pipeline")
