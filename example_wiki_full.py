from vessel import Vessel
import os
import time
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from datetime import timedelta

# Try to import datasets for Wikipedia data
try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False
    print("Warning: datasets library not found. Install with: pip install datasets")
    print("Run: pip install datasets")
    exit(1)

load_dotenv()
VESSEL_BASE_URL = os.getenv("VESSEL_BASE_URL")
VESSEL_API_KEY = os.getenv("VESSEL_API_KEY")

# Configuration
EMBEDDING_MODEL_NAME = "vessel-embedding-nano"
MAX_ARTICLES = 15_000  # Set to an integer to limit articles, or None for all of Wikipedia
TARGET_TOKENS_PER_BATCH = 22_500_000  # 22.5M tokens per batch (reduced to avoid 413 errors)
OUTPUT_DIR = "wiki_embeddings"

# Simple tokenizer approximation (4 chars ≈ 1 token)
def estimate_tokens(text):
    return len(text) // 4

def chunk_text(text, max_tokens=8_000):
    """
    Chunk text into smaller pieces suitable for embeddings.
    Using ~7,000 tokens per chunk to be safe with most embedding models.
    """
    target_chunk_size = 7_000  # tokens per chunk
    target_char_size = target_chunk_size * 4  # approximate chars
    
    chunks = []
    sentences = text.replace('\n', ' ').split('. ')
    
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) < target_char_size:
            current_chunk += sentence + ". "
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

def download_wikipedia_articles(max_articles=None):
    """
    Download Wikipedia articles.
    If max_articles is None, download all available articles.
    Returns a generator that yields articles.
    """
    if not HAS_DATASETS:
        print("ERROR: datasets library is required.")
        print("Install with: pip install datasets")
        return
    
    print(f"\nLoading Wikipedia dataset from Hugging Face (streaming mode)...")
    
    # Load the Wikipedia dataset
    try:
        dataset = load_dataset(
            "wikimedia/wikipedia",
            "20231101.en",  # English Wikipedia from November 1, 2023
            split="train",
            streaming=True,
            trust_remote_code=True
        )
    except Exception as e:
        print(f"Error loading dataset with 20231101.en, trying alternative...")
        try:
            dataset = load_dataset(
                "wikimedia/wikipedia",
                "20231101.en",
                split="train",
                streaming=True
            )
        except Exception as e2:
            print(f"Error: {e2}")
            print("Trying to use the dataset without trust_remote_code...")
            dataset = load_dataset(
                "wikipedia",
                "20220301.en",
                split="train",
                streaming=True
            )
    
    article_count = 0
    for article in dataset:
        # Extract text from article
        text = article.get('text', '')
        
        if not text or estimate_tokens(text) < 100:
            continue
        
        yield text
        article_count += 1
        
        if max_articles is not None and article_count >= max_articles:
            break

def prepare_batches(articles_generator, target_tokens_per_batch):
    """
    Process articles and organize them into batches of approximately target_tokens_per_batch.
    Returns a list of batches, where each batch is a list of text chunks.
    """
    batches = []
    current_batch = []
    current_batch_tokens = 0
    total_articles = 0
    total_chunks = 0
    
    print(f"\nPreparing batches of ~{target_tokens_per_batch:,} tokens each...")
    
    for article in articles_generator:
        chunks = chunk_text(article)
        
        for chunk in chunks:
            chunk_tokens = estimate_tokens(chunk)
            
            # If adding this chunk would exceed the target, start a new batch
            if current_batch_tokens + chunk_tokens > target_tokens_per_batch and current_batch:
                batches.append(current_batch)
                print(f"  Batch {len(batches)} prepared: {len(current_batch):,} chunks, ~{current_batch_tokens:,} tokens")
                current_batch = []
                current_batch_tokens = 0
            
            current_batch.append(chunk)
            current_batch_tokens += chunk_tokens
            total_chunks += 1
        
        total_articles += 1
        if total_articles % 100 == 0:
            print(f"  Processed {total_articles:,} articles, {total_chunks:,} chunks, {len(batches)} batches prepared...")
    
    # Add the last batch if it has any chunks
    if current_batch:
        batches.append(current_batch)
        print(f"  Batch {len(batches)} prepared: {len(current_batch):,} chunks, ~{current_batch_tokens:,} tokens")
    
    print(f"\n✅ Prepared {len(batches)} batches from {total_articles:,} articles ({total_chunks:,} total chunks)")
    return batches, total_articles, total_chunks

def save_embeddings_batch(batch_idx, embeddings_data, output_dir):
    """
    Save embeddings to disk efficiently using numpy's compressed format.
    Each batch is saved as a separate .npz file with metadata.
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Convert embeddings to numpy array for efficient storage
    embeddings_array = np.array([emb.embedding for emb in embeddings_data])
    
    # Save with compression
    filename = output_path / f"batch_{batch_idx:04d}.npz"
    np.savez_compressed(
        filename,
        embeddings=embeddings_array,
        batch_idx=batch_idx,
        count=len(embeddings_data)
    )
    
    return filename

def format_time_remaining(seconds):
    """Format seconds into human-readable time."""
    if seconds < 0:
        return "calculating..."
    
    td = timedelta(seconds=int(seconds))
    days = td.days
    hours, remainder = divmod(td.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    parts = []
    if days > 0:
        parts.append(f"{days}d")
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    if seconds > 0 or not parts:
        parts.append(f"{seconds}s")
    
    return " ".join(parts)

def main():
    client = Vessel(
        base_url=VESSEL_BASE_URL,
        api_key=VESSEL_API_KEY
    )
    
    print("="*80)
    print("FULL WIKIPEDIA EMBEDDING PROCESSOR")
    print("="*80)
    print(f"Model: {EMBEDDING_MODEL_NAME}")
    print(f"Max articles: {MAX_ARTICLES if MAX_ARTICLES else 'ALL (entire English Wikipedia)'}")
    print(f"Target tokens per batch: {TARGET_TOKENS_PER_BATCH:,}")
    print(f"Output directory: {OUTPUT_DIR}")
    
    # Download and prepare batches
    articles_gen = download_wikipedia_articles(max_articles=MAX_ARTICLES)
    batches, total_articles, total_chunks = prepare_batches(articles_gen, TARGET_TOKENS_PER_BATCH)
    
    if not batches:
        print("ERROR: No batches prepared")
        return
    
    total_batches = len(batches)
    print(f"\n{'='*80}")
    print(f"CREATING {total_batches} ASYNC BATCH JOBS")
    print(f"{'='*80}")
    
    # Create all async batch jobs
    batch_jobs = []
    for idx, batch_chunks in enumerate(batches):
        print(f"\nSubmitting batch {idx + 1}/{total_batches}...")
        print(f"  Chunks in batch: {len(batch_chunks):,}")
        
        batch_job = client.embeddings.create_async(
            model=EMBEDDING_MODEL_NAME,
            input=batch_chunks
        )
        
        batch_jobs.append({
            'job': batch_job,
            'idx': idx,
            'chunks': len(batch_chunks),
            'submitted_at': time.time()
        })
        
        print(f"  ✅ Batch {idx + 1} submitted (Job ID: {batch_job.id})")
    
    print(f"\n{'='*80}")
    print(f"ALL {total_batches} BATCHES SUBMITTED - MONITORING PROGRESS")
    print(f"{'='*80}")
    
    # Track completion
    completed_batches = []
    total_tokens_processed = 0
    total_cost = 0.0
    start_time = time.time()
    
    # Monitor all jobs until completion
    while len(completed_batches) < total_batches:
        for batch_info in batch_jobs:
            # Skip if already completed
            if batch_info['idx'] in [b['idx'] for b in completed_batches]:
                continue
            
            # Check if this batch is finished
            status = batch_info['job'].poll()
            if status.finished:
                # Get the results
                response = batch_info['job'].get()
                
                if response.success:
                    # Save embeddings to disk
                    saved_file = save_embeddings_batch(
                        batch_info['idx'],
                        response.data,
                        OUTPUT_DIR
                    )
                    
                    # Update tracking
                    completed_batches.append(batch_info)
                    total_tokens_processed += response.input_tokens
                    total_cost += response.exact_cost
                    
                    # Calculate statistics
                    batches_completed = len(completed_batches)
                    batches_remaining = total_batches - batches_completed
                    elapsed_time = time.time() - start_time
                    
                    # Estimate time remaining based on average throughput
                    if batches_completed > 0:
                        avg_time_per_batch = elapsed_time / batches_completed
                        estimated_time_remaining = avg_time_per_batch * batches_remaining
                    else:
                        estimated_time_remaining = -1
                    
                    # Estimate total cost
                    if total_tokens_processed > 0:
                        cost_per_token = total_cost / total_tokens_processed
                        # Estimate remaining tokens (rough approximation)
                        avg_tokens_per_batch = total_tokens_processed / batches_completed
                        estimated_remaining_tokens = avg_tokens_per_batch * batches_remaining
                        estimated_remaining_cost = estimated_remaining_tokens * cost_per_token
                        estimated_total_cost = total_cost + estimated_remaining_cost
                    else:
                        estimated_total_cost = 0.0
                    
                    # Print progress
                    print(f"\n{'='*80}")
                    print(f"Batch {batch_info['job'].id} completed ({batches_completed} of {total_batches}):")
                    print(f"  Number of tokens: {response.input_tokens:,}")
                    print(f"  Cost: ${response.exact_cost:.6f} USD")
                    print(f"  Throughput: {response.throughput:,.2f} tokens/second")
                    print(f"  Embeddings saved to: {saved_file}")
                    print(f"  Estimated time to total completion: {format_time_remaining(estimated_time_remaining)}")
                    print(f"  Estimated total cost: ${estimated_total_cost:.2f} USD")
                    print(f"{'='*80}")
                else:
                    print(f"\n❌ Batch {batch_info['idx'] + 1} failed: {response.error}")
                    completed_batches.append(batch_info)  # Mark as "completed" to move on
        
        # Sleep briefly before checking again
        if len(completed_batches) < total_batches:
            time.sleep(5)
    
    # Final summary
    total_time = time.time() - start_time
    
    print(f"\n{'='*80}")
    print("✅ ALL BATCHES COMPLETED!")
    print(f"{'='*80}")
    print(f"\n📊 Final Statistics:")
    print(f"  Total articles processed: {total_articles:,}")
    print(f"  Total chunks processed: {total_chunks:,}")
    print(f"  Total batches: {total_batches}")
    print(f"  Total tokens: {total_tokens_processed:,}")
    print(f"  Total cost: ${total_cost:.6f} USD")
    print(f"  Total time: {format_time_remaining(total_time)}")
    print(f"  Average throughput: {total_tokens_processed/total_time:,.2f} tokens/second")
    print(f"  Embeddings saved to: {OUTPUT_DIR}/")
    print(f"\n{'='*80}")

if __name__ == "__main__":
    main()

