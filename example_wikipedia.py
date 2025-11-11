from vessel import Vessel
import os
from dotenv import load_dotenv

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

EMBEDDING_MODEL_NAME = "vessel-embedding-mini"

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

def download_wikipedia_subset(target_tokens=5_000_000):
    """
    Download a subset of Wikipedia articles until we reach target token count.
    Uses the Hugging Face datasets library with the wikimedia/wikipedia dataset.
    """
    if not HAS_DATASETS:
        print("ERROR: datasets library is required.")
        print("Install with: pip install datasets")
        return []
    
    print(f"\nDownloading Wikipedia data (target: {target_tokens:,} tokens)...")
    print("Loading Wikipedia dataset from Hugging Face (streaming mode)...")
    
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
    
    all_text = []
    total_tokens = 0
    article_count = 0
    
    print("Fetching articles...")
    for article in dataset:
        # Extract text from article
        text = article.get('text', '')
        
        if not text:
            continue
            
        tokens = estimate_tokens(text)
        
        if tokens > 100:  # Skip very short articles
            all_text.append(text)
            total_tokens += tokens
            article_count += 1
            
            if article_count % 100 == 0:
                print(f"  Downloaded {article_count} articles, ~{total_tokens:,} tokens...")
            
            if total_tokens >= target_tokens:
                break
    
    print(f"✅ Downloaded {article_count} articles with ~{total_tokens:,} tokens")
    return all_text, article_count

def main():
    client = Vessel(
        base_url=VESSEL_BASE_URL,
        api_key=VESSEL_API_KEY
    )
    
    print("="*80)
    print("WIKIPEDIA EMBEDDING SPEED TEST (Vessel SDK)")
    print("="*80)
    print(f"Model: {EMBEDDING_MODEL_NAME}")
    
    # Download Wikipedia data
    articles, article_count = download_wikipedia_subset(target_tokens=25_000_000)
    
    if not articles:
        print("ERROR: Failed to download Wikipedia data")
        return
    
    # Chunk all articles
    print("\nChunking articles for embeddings...")
    all_chunks = []
    for article in articles:
        chunks = chunk_text(article)
        all_chunks.extend(chunks)
    
    total_estimated_tokens = sum(estimate_tokens(chunk) for chunk in all_chunks)
    print(f"✅ Created {len(all_chunks)} chunks")
    print(f"   Estimated total tokens: {total_estimated_tokens:,}")
    
    # Use the Vessel SDK to create embeddings (automatically handles batching)
    print("\n" + "="*80)
    print("CREATING EMBEDDINGS...")
    print("="*80)
    print(f"Processing {len(all_chunks)} chunks using Vessel SDK...")
    print("(The SDK automatically handles batch creation, upload, and processing)")
    
    response = client.embeddings.create(
        model=EMBEDDING_MODEL_NAME,
        input=all_chunks,
        verbose=True
    )
    
    if not response.success:
        print(f"\n❌ Error: {response.error}")
        return
    
    # Display results
    print("\n" + "="*80)
    print("📊 SPEED TEST RESULTS")
    print("="*80)
    
    print(f"\n📦 Batch Information:")
    print(f"   Total embeddings generated: {len(response.data):,}")
    print(f"   Total chunks processed: {len(all_chunks):,}")
    
    print(f"\n🔢 Token Statistics:")
    print(f"   Estimated tokens (pre-processing): {total_estimated_tokens:,}")
    print(f"   Actual tokens (from API): {response.input_tokens:,}")
    print(f"   Accuracy of estimation: {(total_estimated_tokens/response.input_tokens*100):.1f}%")
    
    print(f"\n⏱️  Performance:")
    print(f"   Processing time: {response.processing_time:.2f} seconds ({response.processing_time/60:.2f} minutes)")
    print(f"   Throughput: {response.throughput:,.2f} tokens/second")
    print(f"   Chunks per second: {len(response.data)/response.processing_time:.2f}")
    
    print(f"\n💰 Cost Analysis:")
    print(f"   Total tokens: {response.input_tokens:,}")
    print(f"   Total cost (formatted): {response.total_cost} USD")
    print(f"   Total cost (exact): {response.exact_cost:.10f} USD")
    print(f"   Cost per embedding: ${response.exact_cost/len(response.data):.10f} USD")
    
    # Sample embeddings
    print("\n" + "="*80)
    print("SAMPLE RESULTS (first 3):")
    print("="*80)
    for i in range(min(3, len(response.data))):
        embedding = response.data[i]
        print(f"\nEmbedding {i+1}:")
        print(f"  Dimension: {len(embedding.embedding)}")
        print(f"  First 5 values: {[f'{v:.4f}' for v in embedding.embedding[:5]]}")
    
    # Estimate full Wikipedia processing
    print("\n" + "="*80)
    print("🌍 FULL WIKIPEDIA ESTIMATION")
    print("="*80)
    
    # Calculate average tokens per article from our test data
    avg_tokens_per_article = response.input_tokens / article_count if article_count > 0 else 2000
    
    # English Wikipedia has approximately 7.08 million articles (as of October 2025)
    estimated_total_articles = 7_082_001
    estimated_total_tokens = int(estimated_total_articles * avg_tokens_per_article)
    
    print(f"\n📊 Full Wikipedia Estimates:")
    print(f"   Estimated total articles: {estimated_total_articles:,}")
    print(f"   Average tokens per article (from test): {avg_tokens_per_article:,.0f}")
    print(f"   Estimated total tokens: {estimated_total_tokens:,}")
    
    # Calculate estimated cost and time based on our test results
    cost_per_token = response.exact_cost / response.input_tokens if response.input_tokens > 0 else 0
    estimated_full_cost = estimated_total_tokens * cost_per_token
    estimated_processing_time_seconds = estimated_total_tokens / response.throughput if response.throughput > 0 else 0
    estimated_processing_time_hours = estimated_processing_time_seconds / 3600
    
    print(f"\n💰 Cost Projection:")
    print(f"   Estimated total cost: ${estimated_full_cost:,.2f} USD")
    print(f"   Cost per article: ${estimated_full_cost/estimated_total_articles:.6f} USD")
    
    print(f"\n⏱️  Time Projection:")
    print(f"   Based on throughput: {response.throughput:,.2f} tokens/second")
    print(f"   Estimated processing time: {estimated_processing_time_hours:,.1f} hours ({estimated_processing_time_hours/24:.1f} days)")
    
    # Calculate what we processed as a percentage
    percentage_processed = (response.input_tokens / estimated_total_tokens) * 100
    print(f"\n📈 Test Coverage:")
    print(f"   Processed in this test: {response.input_tokens:,} tokens ({percentage_processed:.3f}% of full Wikipedia)")
    print(f"   Articles in this test: {article_count:,} ({(article_count/estimated_total_articles)*100:.2f}% of full Wikipedia)")
    
    print("\n" + "="*80)
    print("✅ Speed test completed successfully!")
    print("="*80)

if __name__ == "__main__":
    main()



""" nano model output:
python3 example_wikipedia.py 
================================================================================
WIKIPEDIA EMBEDDING SPEED TEST (Vessel SDK)
================================================================================
Model: vessel-embedding-nano

Downloading Wikipedia data (target: 25,000,000 tokens)...
Loading Wikipedia dataset from Hugging Face (streaming mode)...
`trust_remote_code` is not supported anymore.
Please check that the Hugging Face dataset 'wikimedia/wikipedia' isn't based on a loading script and remove `trust_remote_code`.
If the dataset is based on a loading script, please ask the dataset author to remove it and convert it to a standard format like Parquet.
Resolving data files: 100%|█████████████████████████████████████████████████████████████████████████████████| 41/41 [00:00<00:00, 86589.36it/s]
Fetching articles...
  Downloaded 100 articles, ~824,339 tokens...
  Downloaded 200 articles, ~1,542,393 tokens...
  Downloaded 300 articles, ~2,136,855 tokens...
...
  Downloaded 11400 articles, ~24,965,265 tokens...
✅ Downloaded 11448 articles with ~25,002,797 tokens

Chunking articles for embeddings...
✅ Created 12685 chunks
   Estimated total tokens: 25,004,756

================================================================================
CREATING EMBEDDINGS...
================================================================================
Processing 12685 chunks using Vessel SDK...
(The SDK automatically handles batch creation, upload, and processing)
Creating batch of 12685 embeddings...
Uploading batch file...
Creating batch job (ID: file-fde86d4144a249d28e70143c33e897db)...
Waiting for batch to complete...
  Status: validating
    Completed: 0/0
  Status: queued
    Completed: 0/12685
  Status: in_progress
    Completed: 0/12685
  Status: completed
    Completed: 12685/12685
✅ Batch completed successfully
   Processing time: 85.14 seconds
   Throughput: 269471.67 tokens/sec

================================================================================
📊 SPEED TEST RESULTS
================================================================================

📦 Batch Information:
   Total embeddings generated: 12,685
   Total chunks processed: 12,685

🔢 Token Statistics:
   Estimated tokens (pre-processing): 25,004,756
   Actual tokens (from API): 22,943,395
   Accuracy of estimation: 109.0%

⏱️  Performance:
   Processing time: 85.14 seconds (1.42 minutes)
   Throughput: 269,471.67 tokens/second
   Chunks per second: 148.99

💰 Cost Analysis:
   Total tokens: 22,943,395
   Total cost (formatted): $0.06 USD
   Total cost (exact): 0.0573584875 USD
   Cost per embedding: $0.0000045218 USD

================================================================================
SAMPLE RESULTS (first 3):
================================================================================

Embedding 1:
  Dimension: 768
  First 5 values: ['-3.7188', '0.3086', '-4.7500', '8.0625', '-10.3750']

Embedding 2:
  Dimension: 768
  First 5 values: ['-3.5469', '-1.0859', '-5.4688', '2.8125', '2.5781']

Embedding 3:
  Dimension: 768
  First 5 values: ['-7.2188', '2.5000', '0.4316', '5.1250', '-10.6250']

================================================================================
🌍 FULL WIKIPEDIA ESTIMATION
================================================================================

📊 Full Wikipedia Estimates:
   Estimated total articles: 7,082,001
   Average tokens per article (from test): 2,004
   Estimated total tokens: 14,193,321,657

💰 Cost Projection:
   Estimated total cost: $35.48 USD
   Cost per article: $0.000005 USD

⏱️  Time Projection:
   Based on throughput: 269,471.67 tokens/second
   Estimated processing time: 14.6 hours (0.6 days)

📈 Test Coverage:
   Processed in this test: 22,943,395 tokens (0.162% of full Wikipedia)
   Articles in this test: 11,448 (0.16% of full Wikipedia)

================================================================================
✅ Speed test completed successfully!
================================================================================
"""






""" mini model output:
  Downloaded 11400 articles, ~24,965,265 tokens...
✅ Downloaded 11448 articles with ~25,002,797 tokens

Chunking articles for embeddings...
✅ Created 12685 chunks
   Estimated total tokens: 25,004,756

================================================================================
CREATING EMBEDDINGS...
================================================================================
Processing 12685 chunks using Vessel SDK...
(The SDK automatically handles batch creation, upload, and processing)
Creating batch of 12685 embeddings...
Uploading batch file...
Creating batch job (ID: file-256ba5a7a7484dee970b5fc47ea9b141)...
Waiting for batch to complete...
  Status: validating
    Completed: 0/0
  Status: queued
    Completed: 0/12685
  Status: in_progress
    Completed: 0/12685
  Status: completed
    Completed: 12685/12685
✅ Batch completed successfully
   Processing time: 173.74 seconds
   Throughput: 135611.79 tokens/sec

================================================================================
📊 SPEED TEST RESULTS
================================================================================

📦 Batch Information:
   Total embeddings generated: 12,685
   Total chunks processed: 12,685

🔢 Token Statistics:
   Estimated tokens (pre-processing): 25,004,756
   Actual tokens (from API): 23,561,623
   Accuracy of estimation: 106.1%

⏱️  Performance:
   Processing time: 173.74 seconds (2.90 minutes)
   Throughput: 135,611.79 tokens/second
   Chunks per second: 73.01

💰 Cost Analysis:
   Total tokens: 23,561,623
   Total cost (formatted): $0.21 USD
   Total cost (exact): 0.2120546070 USD
   Cost per embedding: $0.0000167170 USD

================================================================================
SAMPLE RESULTS (first 3):
================================================================================

Embedding 1:
  Dimension: 1024
  First 5 values: ['2.4929', '-6.6120', '-0.7007', '7.3830', '2.1054']

Embedding 2:
  Dimension: 1024
  First 5 values: ['2.1420', '-9.5359', '-0.9175', '6.2952', '2.3901']

Embedding 3:
  Dimension: 1024
  First 5 values: ['-5.1750', '-5.1252', '-0.7141', '7.3394', '2.2105']

================================================================================
🌍 FULL WIKIPEDIA ESTIMATION
================================================================================

📊 Full Wikipedia Estimates:
   Estimated total articles: 7,082,001
   Average tokens per article (from test): 2,058
   Estimated total tokens: 14,575,771,981

💰 Cost Projection:
   Estimated total cost: $131.18 USD
   Cost per article: $0.000019 USD

⏱️  Time Projection:
   Based on throughput: 135,611.79 tokens/second
   Estimated processing time: 29.9 hours (1.2 days)

📈 Test Coverage:
   Processed in this test: 23,561,623 tokens (0.162% of full Wikipedia)
   Articles in this test: 11,448 (0.16% of full Wikipedia)

================================================================================
✅ Speed test completed successfully!
================================================================================
"""