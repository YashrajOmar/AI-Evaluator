"""
image_captioner.py
Generate captions for images using BLIP-2 vision-language model.
"""

import os
from pathlib import Path
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration


class ImageCaptioner:
    """
    Generate semantic descriptions of images using BLIP-1.
    """
    
    def __init__(self, model_name: str = "Salesforce/blip-image-captioning-large", device: str = None):
        """
        Initialize BLIP-1 model.
        
        Parameters:
            model_name: Hugging Face model identifier
            device: 'cuda', 'cpu', or None (auto-detect)
        """
        print(f"Loading BLIP-1 model: {model_name}")
        
        # Auto-detect device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = device
        print(f"Using device: {device}")
        
        # Load processor and model
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        )
        self.model.to(device)
        
        print("✓ Model loaded successfully")
    
    def generate_caption(self, image_path: str, prompt: str = None) -> str:
        """
        Generate caption for a single image.
        
        Parameters:
            image_path: Path to image file
            prompt: Optional text prompt to guide caption generation (not used for BLIP-1)
        
        Returns:
            Generated caption text
        """
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')
            
            # BLIP-1 works best without text prompts - just image captioning
            inputs = self.processor(image, return_tensors="pt").to(self.device, torch.float16 if self.device == "cuda" else torch.float32)
            
            # Generate caption with better parameters
            generated_ids = self.model.generate(
                **inputs, 
                max_new_tokens=30,  # Shorter for better quality
                num_beams=5,  # Beam search for better captions
                early_stopping=True
            )
            caption = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            
            return caption
            
        except Exception as e:
            print(f"Error generating caption for {image_path}: {e}")
            return f"[Error: Could not generate caption]"
    
    def generate_captions_batch(self, image_paths: list, prompt: str = None) -> list:
        """
        Generate captions for multiple images.
        
        Parameters:
            image_paths: List of image file paths
            prompt: Optional text prompt for all images
        
        Returns:
            List of dicts with image_path and caption
        """
        results = []
        
        print(f"\nGenerating captions for {len(image_paths)} images...")
        
        for i, image_path in enumerate(image_paths, 1):
            print(f"  [{i}/{len(image_paths)}] Processing: {Path(image_path).name}")
            
            caption = self.generate_caption(image_path, prompt)
            
            results.append({
                'image_path': image_path,
                'caption': caption
            })
            
            print(f"    Caption: {caption}")
        
        print(f"\n✓ Generated {len(results)} captions")
        return results


def caption_extracted_images(
    image_metadata: list,
    prompt: str = None,  # Prompt not used for BLIP-1 (works better without it)
    model_name: str = "Salesforce/blip-image-captioning-large"
) -> list:
    """
    Generate captions for all extracted images.
    
    Parameters:
        image_metadata: List of dicts from image_extractor
        prompt: Not used for BLIP-1 (kept for API compatibility)
        model_name: BLIP-1 model to use
    
    Returns:
        List of dicts with image metadata + captions
    """
    if not image_metadata:
        print("No images to caption")
        return []
    
    # Initialize captioner
    captioner = ImageCaptioner(model_name)
    
    # Extract image paths
    image_paths = [img['image_path'] for img in image_metadata]
    
    # Generate captions
    captions = captioner.generate_captions_batch(image_paths, prompt=None)  # No prompt for better BLIP-1 results
    
    # Merge with metadata
    for img_meta, caption_data in zip(image_metadata, captions):
        img_meta['caption'] = caption_data['caption']
    
    return image_metadata


# ─────────────────────────────────────────────────────────────────────────────
# CLI Usage
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate captions for images using BLIP-1")
    parser.add_argument("--image-dir", default="extracted_images", help="Directory containing extracted images")
    parser.add_argument("--model", default="Salesforce/blip-image-captioning-large", help="BLIP-1 model name")
    parser.add_argument("--prompt", default="Describe this educational diagram or figure:", help="Caption prompt")
    
    args = parser.parse_args()
    
    # Find all images
    image_dir = Path(args.image_dir)
    if not image_dir.exists():
        print(f"Error: Directory {args.image_dir} not found")
        print("Run image_extractor.py first to extract images from PDFs")
        exit(1)
    
    # Get all image files
    image_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        image_files.extend(image_dir.rglob(ext))
    
    if not image_files:
        print(f"No images found in {args.image_dir}")
        print("Make sure to run image_extractor.py first")
        exit(0)
    
    print(f"\n{'='*60}")
    print(f"Found {len(image_files)} images")
    print(f"{'='*60}\n")
    
    # Create metadata
    image_metadata = [{'image_path': str(img)} for img in image_files]
    
    # Generate captions
    results = caption_extracted_images(image_metadata, args.prompt, args.model)
    
    # Show summary
    print(f"\n{'='*60}")
    print("SAMPLE CAPTIONS:")
    print(f"{'='*60}")
    for i, result in enumerate(results[:3], 1):
        print(f"\n{i}. {Path(result['image_path']).name}")
        print(f"   Caption: {result['caption']}")
    
    if len(results) > 3:
        print(f"\n... and {len(results) - 3} more")
