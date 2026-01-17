#!/usr/bin/env python3
"""
Example: AI Detection with Vessel SDK

This example demonstrates how to use the vessel-detect-large model
to classify texts as AI-generated or human-written.
"""

from vessel import Vessel

# Initialize the Vessel client
client = Vessel(
    base_url="https://vessel.acampi.dev/v1",
    api_key="your-api-key-here"
)

# Example texts to classify
texts = [
    "The quick brown fox jumps over the lazy dog.",
    "Artificial intelligence is transforming the way we work and live.",
    "In a world where technology advances at an unprecedented pace, "
    "it is imperative that we consider the ethical implications.",
    "lol that movie was so good!! cant wait to see it again 😄"
]

print("=" * 80)
print("AI Detection Example - vessel-detect-large")
print("=" * 80)
print()

# Single request with multiple texts
print(f"Classifying {len(texts)} texts...")
print()

response = client.classify.create(
    model="vessel-detect-large",
    texts=texts,
    verbose=True
)

if response.success:
    print()
    print("Results:")
    print("-" * 80)
    
    for i, (text, classification) in enumerate(zip(texts, response.classifications)):
        is_ai = classification['is_ai']
        confidence = classification['confidence']
        
        # Determine label
        if is_ai > 0.5:
            label = "AI-Generated"
            percentage = is_ai * 100
        else:
            label = "Human-Written"
            percentage = (1 - is_ai) * 100
        
        print(f"\nText {i+1}: \"{text[:60]}{'...' if len(text) > 60 else ''}\"")
        print(f"  Classification: {label} ({percentage:.1f}% probability)")
        print(f"  Confidence: {confidence:.2f}")
    
    print()
    print("-" * 80)
    print(f"Processing time: {response.processing_time:.2f}s")
    print(f"Throughput: {response.throughput:.0f} tokens/sec")
    print(f"Cost: {response.total_cost}")
    print()
else:
    print(f"Error: {response.error}")


# Example: Async batch processing
print()
print("=" * 80)
print("Async Batch Classification Example")
print("=" * 80)
print()

more_texts = [
    "This product exceeded my expectations in every way!",
    "The integration of machine learning algorithms has significantly "
    "enhanced the system's predictive capabilities.",
    "hey can u send me that link again? i lost it lol",
]

print(f"Starting async batch job for {len(more_texts)} texts...")

# Create async batch job
batch_job = client.classify.create_async(
    model="vessel-detect-large",
    texts=more_texts
)

print(f"Batch job created: {batch_job.id}")
print("Polling for completion...")

# Poll until complete
while True:
    status = batch_job.poll()
    print(f"  Status: {status.status}")
    
    if status.finished:
        break
    
    import time
    time.sleep(2)

# Get results
if status.status == "completed":
    result = batch_job.get()
    
    if result.success:
        print()
        print("Async Batch Results:")
        print("-" * 80)
        
        for i, (text, classification) in enumerate(zip(more_texts, result.classifications)):
            is_ai = classification['is_ai']
            confidence = classification['confidence']
            
            if is_ai > 0.5:
                label = "AI-Generated"
                percentage = is_ai * 100
            else:
                label = "Human-Written"
                percentage = (1 - is_ai) * 100
            
            print(f"\nText {i+1}: \"{text[:60]}{'...' if len(text) > 60 else ''}\"")
            print(f"  Classification: {label} ({percentage:.1f}% probability)")
            print(f"  Confidence: {confidence:.2f}")
        
        print()
        print("-" * 80)
        print(f"Total cost: {result.total_cost}")
        print()
else:
    print(f"Batch job failed: {status.status}")

print()
print("✅ Examples completed!")
print()
