#!/usr/bin/env python3
"""
Test script to check what rembg.remove() actually returns
Run this locally to see the actual output format
"""
import sys
from PIL import Image
import numpy as np
import io

try:
    from rembg import remove, new_session
    print("✓ rembg imported successfully")
except ImportError as e:
    print(f"✗ Failed to import rembg: {e}")
    print("Install with: pip install rembg")
    sys.exit(1)

def test_rembg_output(image_path):
    """Test what rembg.remove() returns"""
    print(f"\n=== Testing rembg with image: {image_path} ===")
    
    # Load image
    try:
        image = Image.open(image_path)
        print(f"✓ Loaded image: {image.mode}, size: {image.size}")
    except Exception as e:
        print(f"✗ Failed to load image: {e}")
        return
    
    # Convert to RGB if needed (rembg expects RGB input)
    if image.mode != "RGB":
        if image.mode == "RGBA":
            print(f"  Converting RGBA → RGB (rembg will add alpha back)")
        image = image.convert("RGB")
    
    # Process with rembg
    print("\nProcessing with rembg.remove()...")
    try:
        session = new_session("u2netp")  # Use same model as production
        result = remove(
            image,
            session=session,
            alpha_matting=True,
            alpha_matting_foreground_threshold=240,
            alpha_matting_background_threshold=10,
            alpha_matting_erode_size=10,
            post_process_mask=True,
        )
        print(f"✓ rembg processing complete")
    except Exception as e:
        print(f"✗ rembg processing failed: {e}")
        return
    
    # Check what rembg returned
    print(f"\n=== rembg Result Analysis ===")
    print(f"Mode: {result.mode}")
    print(f"Size: {result.size}")
    print(f"Format: {result.format}")
    
    # Check alpha channel if RGBA
    if result.mode == "RGBA":
        alpha = np.array(result.split()[-1])
        min_alpha = int(alpha.min())
        max_alpha = int(alpha.max())
        mean_alpha = float(alpha.mean())
        transparent_pixels = int(np.sum(alpha < 10))  # Nearly transparent
        opaque_pixels = int(np.sum(alpha > 250))  # Nearly opaque
        
        print(f"\nAlpha Channel Analysis:")
        print(f"  Min alpha: {min_alpha}")
        print(f"  Max alpha: {max_alpha}")
        print(f"  Mean alpha: {mean_alpha:.2f}")
        print(f"  Transparent pixels (alpha < 10): {transparent_pixels} ({transparent_pixels/alpha.size*100:.1f}%)")
        print(f"  Opaque pixels (alpha > 250): {opaque_pixels} ({opaque_pixels/alpha.size*100:.1f}%)")
        
        # Check RGB values for background areas
        rgb_array = np.array(result)[:, :, :3]
        alpha_array = np.array(result)[:, :, 3]
        
        # Find transparent areas
        transparent_mask = alpha_array < 10
        if np.any(transparent_mask):
            transparent_rgb = rgb_array[transparent_mask]
            if len(transparent_rgb) > 0:
                mean_rgb_transparent = transparent_rgb.mean(axis=0)
                print(f"\nTransparent areas RGB values:")
                print(f"  Mean RGB: ({mean_rgb_transparent[0]:.1f}, {mean_rgb_transparent[1]:.1f}, {mean_rgb_transparent[2]:.1f})")
        
        # Find opaque areas
        opaque_mask = alpha_array > 250
        if np.any(opaque_mask):
            opaque_rgb = rgb_array[opaque_mask]
            if len(opaque_rgb) > 0:
                mean_rgb_opaque = opaque_rgb.mean(axis=0)
                print(f"\nOpaque areas RGB values:")
                print(f"  Mean RGB: ({mean_rgb_opaque[0]:.1f}, {mean_rgb_opaque[1]:.1f}, {mean_rgb_opaque[2]:.1f})")
        
        # Check if opaque areas are white
        white_threshold = 240
        opaque_white = opaque_mask & np.all(rgb_array > white_threshold, axis=2)
        white_pixel_count = int(np.sum(opaque_white))
        print(f"\nOpaque white pixels (RGB > 240): {white_pixel_count} ({white_pixel_count/alpha.size*100:.1f}%)")
        
        if white_pixel_count > 0:
            print(f"⚠️  WARNING: rembg returned {white_pixel_count} opaque white pixels!")
            print(f"   This means rembg is NOT creating transparency for these areas")
        else:
            print(f"✓ No opaque white pixels found - transparency looks good")
    
    elif result.mode == "RGB":
        print(f"\n⚠️  WARNING: rembg returned RGB instead of RGBA!")
        print(f"   This means no transparency - background will be opaque")
        rgb_array = np.array(result)
        white_pixels = np.all(rgb_array > 240, axis=2)
        white_count = int(np.sum(white_pixels))
        print(f"   White pixels (RGB > 240): {white_count} ({white_count/rgb_array.shape[0]/rgb_array.shape[1]*100:.1f}%)")
    else:
        print(f"\n⚠️  WARNING: rembg returned unexpected mode: {result.mode}")
    
    # Save test output
    output_path = "test_rembg_output.png"
    result.save(output_path, format="PNG")
    print(f"\n✓ Saved test output to: {output_path}")
    print(f"   Open this file to see what rembg actually produced")
    
    return result

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_rembg.py <image_path>")
        print("\nExample:")
        print("  python test_rembg.py test_image.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    test_rembg_output(image_path)
