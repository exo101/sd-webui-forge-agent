#!/usr/bin/env python3
"""
Batch Generate Text2Music Examples using LM
Generates 50 examples and saves them to examples/text2music/
"""
import os
import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from acestep.llm_inference import LLMHandler
from loguru import logger
from tqdm import tqdm


def generate_examples(num_examples=50, output_dir="examples/text2music", start_index=1):
    """
    Generate examples using LM and save to JSON files
    
    Args:
        num_examples: Number of examples to generate
        output_dir: Output directory for JSON files
        start_index: Starting index for example files
    """
    # Initialize LLM Handler
    logger.info("Initializing LLM Handler...")
    llm_handler = LLMHandler()
    
    # Initialize LM
    checkpoint_dir = os.path.join(project_root, "checkpoints")
    
    # Use default LM model
    available_models = llm_handler.get_available_5hz_lm_models()
    if not available_models:
        logger.error("No 5Hz LM models found in checkpoints directory")
        return
    
    # Prefer acestep-5Hz-lm-0.6B if available
    lm_model = "acestep-5Hz-lm-0.6B" if "acestep-5Hz-lm-0.6B" in available_models else available_models[0]
    logger.info(f"Using LM model: {lm_model}")
    
    # Initialize LM
    status_msg, success = llm_handler.initialize(
        checkpoint_dir=checkpoint_dir,
        lm_model_path=lm_model,
        backend="vllm",  # Use vllm for faster generation
        device="auto",
        offload_to_cpu=False,
        dtype=None,
    )
    
    if not success:
        logger.error(f"Failed to initialize LM: {status_msg}")
        return
    
    logger.info(f"LM initialized successfully: {status_msg}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate examples
    successful_count = 0
    failed_count = 0
    
    for i in tqdm(range(num_examples), desc="Generating examples"):
        example_num = start_index + i
        output_file = os.path.join(output_dir, f"example_{example_num:02d}.json")
        
        logger.info(f"Generating example {example_num}/{start_index + num_examples - 1}...")
        
        try:
            # Generate example using LM
            metadata, status = llm_handler.understand_audio_from_codes(
                audio_codes="NO USER INPUT",  # Empty input triggers example generation
                use_constrained_decoding=True,
                temperature=0.85,
                cfg_scale=1.0,
                top_k=None,
                top_p=0.9,
            )
            
            if not metadata:
                logger.warning(f"Failed to generate example {example_num}: {status}")
                failed_count += 1
                continue
            
            # Build JSON data with all available fields
            example_data = {
                "think": True,  # Always true for LM-generated examples
                "caption": metadata.get("caption", ""),
                "lyrics": metadata.get("lyrics", ""),
            }
            
            # Add optional metadata fields if they exist and are not "N/A"
            if "bpm" in metadata and metadata["bpm"] not in [None, "N/A", ""]:
                try:
                    # Convert to int if it's a valid number
                    example_data["bpm"] = int(metadata["bpm"]) if isinstance(metadata["bpm"], (int, str)) else metadata["bpm"]
                except (ValueError, TypeError):
                    example_data["bpm"] = metadata["bpm"]
            
            if "duration" in metadata and metadata["duration"] not in [None, "N/A", ""]:
                try:
                    # Convert to int if it's a valid number
                    example_data["duration"] = int(metadata["duration"]) if isinstance(metadata["duration"], (int, str)) else metadata["duration"]
                except (ValueError, TypeError):
                    example_data["duration"] = metadata["duration"]
            
            if "keyscale" in metadata and metadata["keyscale"] not in [None, "N/A", ""]:
                example_data["keyscale"] = metadata["keyscale"]
            
            if "language" in metadata and metadata["language"] not in [None, "N/A", ""]:
                example_data["language"] = metadata["language"]
            
            if "timesignature" in metadata and metadata["timesignature"] not in [None, "N/A", ""]:
                example_data["timesignature"] = metadata["timesignature"]
            
            # Save to JSON file
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(example_data, f, ensure_ascii=False, indent=4)
            
            logger.info(f"✅ Saved example {example_num} to {output_file}")
            logger.info(f"   Caption preview: {example_data['caption'][:100]}...")
            successful_count += 1
            
        except Exception as e:
            logger.error(f"❌ Error generating example {example_num}: {str(e)}")
            failed_count += 1
            continue
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info(f"Generation complete!")
    logger.info(f"Successful: {successful_count}/{num_examples}")
    logger.info(f"Failed: {failed_count}/{num_examples}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"{'='*60}\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate text2music examples using LM")
    parser.add_argument("--num", type=int, default=100, help="Number of examples to generate (default: 100)")
    parser.add_argument("--output-dir", type=str, default="examples/text2music", help="Output directory (default: examples/text2music)")
    parser.add_argument("--start-index", type=int, default=1, help="Starting index for example files (default: 1)")
    
    args = parser.parse_args()
    
    generate_examples(
        num_examples=args.num,
        output_dir=args.output_dir,
        start_index=args.start_index
    )
