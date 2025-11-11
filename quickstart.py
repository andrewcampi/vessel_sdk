from vessel import Vessel
import os
import json
from dotenv import load_dotenv
import time

load_dotenv()
VESSEL_BASE_URL = os.getenv("VESSEL_BASE_URL")
VESSEL_API_KEY = os.getenv("VESSEL_API_KEY")

client = Vessel(
    base_url=VESSEL_BASE_URL,
    api_key=VESSEL_API_KEY
)

response = client.models.list()
for model in response.data:
    if (model.type == "chat") or (model.type == "embedding"):
        print("Name:", model.name)
        print("Type:", model.type)
        print(f"Input cost: ${model.input_cost} USD per 1M tokens")
        print(f"Output cost: ${model.output_cost} USD per 1M tokens")
        print("--------------------------------")


response = client.accounts.retrieve()
print("Email:", response.data.email)
print("Credits Remaining (USD):", response.data.credits)


# The clint automatically handles building the input file, uploading it, creating the batch job, polling and retrieving the results, and calculating the response metrics.
print("Single embedding...")
response = client.embeddings.create(
    model="vessel-embedding-nano",
    input="Hello, how are you?", # List of length 1, or just a string. The client automatically handles that.
    verbose=False # This will print statuses and progress if True, default is False
)
print("Success:", response.success) # True if the request was successful, False otherwise
if response.success:
    print("Input tokens:", response.input_tokens) # The number of tokens in the input
    print("Embedding dimension:", len(response.data[0].embedding)) # The dimension of the embedding
    print("Embedding:", response.data[0].embedding) # The embedding
    print(f"Reported throughput: {response.throughput} tokens/sec") # The total throughput of the request
    print(f"Processing time: {response.processing_time} seconds") # The time it took to process the request
    print(f"Cost: {response.total_cost} USD") # The total cost of the request (rounded for human readability)
    print(f"Exact Cost: {response.exact_cost} USD") # The exact cost of the request (scientific notation)
else:
    print("Error:", response.error) # The error message


print("Batch embeddings...")
response = client.embeddings.create( # The rate limit is 10 seconds between requests. The client automatically handles that.
    model="vessel-embedding-nano",
    input=["Hello, how are you?", "Hello, how are you?", "Hello, how are you?", "Hello, how are you?", "Hello, how are you?"],
    verbose=False # This will print statuses and progress if True, default is False
)
print("Success:", response.success) # True if the request was successful, False otherwise
if response.success:
    print("Input tokens:", response.input_tokens) # The number of tokens in the input
    print("Embedding dimension:", len(response.data[0].embedding)) # The dimension of the embedding
    print("Embeddings Generated:", len(response.data)) # The number of embeddings generated
    print(f"Reported throughput: {response.throughput} tokens/sec") # The total throughput of the request
    print(f"Processing time: {response.processing_time} seconds") # The time it took to process the request
    print(f"Cost: {response.total_cost} USD") # The total cost of the request (rounded for human readability)
    print(f"Exact Cost: {response.exact_cost} USD") # The exact cost of the request (scientific notation)
else:
    print("Error:", response.error) # The error message


print("Single chat completion...")
response = client.chat.completions.create(
    model="vessel-llm-nano",
    messages=[
        {"role": "user", "content": "Hello, how are you?"}
    ], # If list of dicts, then the client handles it. Batches are lists of lists of dicts, and handled automatically.
    verbose=False # This will print statuses and progress if True, default is False
)
print("Success:", response.success) # True if the request was successful, False otherwise
if response.success:
    print("Input tokens:", response.input_tokens) # The number of tokens in the input
    print("Output tokens:", response.output_tokens) # The number of tokens in the output
    print("Response:", response.data[0].message.content) # The response from the model
    print(f"Reported throughput: {response.throughput} tokens/sec") # The total throughput of the request
    print(f"Processing time: {response.processing_time} seconds") # The time it took to process the request
    print(f"Input cost: {response.input_cost} USD") # The cost of the input (rounded for human readability)
    print(f"Output cost: {response.output_cost} USD") # The cost of the output (rounded for human readability)
    print(f"Total cost: {response.total_cost} USD") # The total cost of the request (rounded for human readability)
    print(f"Exact Cost: {response.exact_cost} USD") # The exact cost of the request (scientific notation)
else:
    print("Error:", response.error) # The error message


print("Batch chat completions")
response = client.chat.completions.create(
    model="vessel-llm-nano",
    messages=[
        [
            {"role": "user", "content": "Hello, how are you?"}
        ],
        [
            {"role": "user", "content": "What is the capital of France?"}
        ],
        [
            {"role": "user", "content": "What is the capital of Germany?"}
        ]
    ],
    verbose=False # This will print statuses and progress if True, default is False
)
print("Success:", response.success) # True if the request was successful, False otherwise
if response.success:
    print("Input tokens:", response.input_tokens) # The number of tokens in the input
    print("Output tokens:", response.output_tokens) # The number of tokens in the output
    print("Responses:", json.dumps(response.data, indent=4)) # The responses from the model
    print(f"Reported throughput: {response.throughput} tokens/sec") # The total throughput of the request
    print(f"Processing time: {response.processing_time} seconds") # The time it took to process the request
    print(f"Input cost: {response.input_cost} USD") # The cost of the input (rounded for human readability)
    print(f"Output cost: {response.output_cost} USD") # The cost of the output (rounded for human readability)
    print(f"Total cost: {response.total_cost} USD") # The total cost of the request (rounded for human readability)
    print(f"Exact Cost: {response.exact_cost} USD") # The exact cost of the request (scientific notation)
else:
    print("Error:", response.error) # The error message


# Batch chat completions (async)
print("Batch chat completions (async)...")
batch_job = client.chat.completions.create_async(
    model="vessel-llm-mini-instruct",
    messages=[
        [
            {"role": "user", "content": "What is the capital of France?"}
        ],
        [
            {"role": "user", "content": "What is the capital of Germany?"}
        ],
        [
            {"role": "user", "content": "What is the capital of Italy?"}
        ],
        [
            {"role": "user", "content": "What is the capital of Spain?"}
        ],
        [
            {"role": "user", "content": "What is the capital of Portugal?"}
        ],
        [
            {"role": "user", "content": "What is the capital of Greece?"}
        ]
    ],
    #verbose=False # Verbose not supported for create_async requests
) # batch_job is simply a response object. No need for `await` to call it. It instantly returns an object with .id, and a function, poll() (that can use used to check status and determine if finished), and a function get() (that can be used to get the results once finished). 

# You can do other things while waiting for the batch to complete. Here, we will just wait in a simple loop.
while not batch_job.poll().finished:
    print(f"Waiting for batch to complete... (current status: {batch_job.poll().status})")
    time.sleep(2)

print(f"Batch completed! (current status: {batch_job.poll().status})")
response = batch_job.get() # This is the same response object as the one returned by the synchronous create request.
print("Success:", response.success) # True if the request was successful, False otherwise
if response.success:
    print("Input tokens:", response.input_tokens) # The number of tokens in the input
    print("Output tokens:", response.output_tokens) # The number of tokens in the output
    print("Responses:", json.dumps(response.data, indent=4)) # The responses from the model
    print(f"Reported throughput: {response.throughput} tokens/sec") # The total throughput of the request
    print(f"Processing time: {response.processing_time} seconds") # The time it took to process the request
    print(f"Input cost: {response.input_cost} USD") # The cost of the input (rounded for human readability)
    print(f"Output cost: {response.output_cost} USD") # The cost of the output (rounded for human readability)
    print(f"Total cost: {response.total_cost} USD") # The total cost of the request (rounded for human readability)
    print(f"Exact Cost: {response.exact_cost} USD") # The exact cost of the request (scientific notation)
else:
    print("Error:", response.error) # The error message


# Batch embeddings (async)
print("Batch embeddings (async)...")
batch_job = client.embeddings.create_async(
    model="vessel-embedding-nano",
    input=[
        "Hello, how are you?",
        "What is the meaning of life?",
        "Tell me about artificial intelligence.",
        "What is the weather like today?",
        "How do I learn to code?"
    ]
) # batch_job is simply a response object. No need for `await` to call it. It instantly returns an object with .id, and a function, poll() (that can use used to check status and determine if finished), and a function get() (that can be used to get the results once finished). 

# You can do other things while waiting for the batch to complete. Here, we will just wait in a simple loop.
while not batch_job.poll().finished:
    print(f"Waiting for batch to complete... (current status: {batch_job.poll().status})")
    time.sleep(2)

print(f"Batch completed! (current status: {batch_job.poll().status})")
response = batch_job.get() # This is the same response object as the one returned by the synchronous create request.
print("Success:", response.success) # True if the request was successful, False otherwise
if response.success:
    print("Input tokens:", response.input_tokens) # The number of tokens in the input
    print("Embedding dimension:", len(response.data[0].embedding)) # The dimension of the embedding
    print("Embeddings Generated:", len(response.data)) # The number of embeddings generated
    print(f"Reported throughput: {response.throughput} tokens/sec") # The total throughput of the request
    print(f"Processing time: {response.processing_time} seconds") # The time it took to process the request
    print(f"Cost: {response.total_cost} USD") # The total cost of the request (rounded for human readability)
    print(f"Exact Cost: {response.exact_cost} USD") # The exact cost of the request (scientific notation)
else:
    print("Error:", response.error) # The error message


""" Note:
The below speeds are slower than advertised because we are sending tiny batches. Larger batches will result in faster speeds!
"""

""" Output:
python3 quickstart.py 
Name: vessel-llm-nano
Type: chat
Input cost: $0.0065 USD per 1M tokens
Output cost: $0.05 USD per 1M tokens
--------------------------------
Name: vessel-llm-nano-instruct
Type: chat
Input cost: $0.0075 USD per 1M tokens
Output cost: $0.065 USD per 1M tokens
--------------------------------
Name: vessel-llm-mini
Type: chat
Input cost: $0.027 USD per 1M tokens
Output cost: $0.2 USD per 1M tokens
--------------------------------
Name: vessel-llm-mini-instruct
Type: chat
Input cost: $0.012 USD per 1M tokens
Output cost: $0.125 USD per 1M tokens
--------------------------------
Name: vessel-llm-small
Type: chat
Input cost: $0.03 USD per 1M tokens
Output cost: $0.2 USD per 1M tokens
--------------------------------
Name: vessel-llm-medium-reasoning
Type: chat
Input cost: $0.08 USD per 1M tokens
Output cost: $0.85 USD per 1M tokens
--------------------------------
Name: vessel-embedding-nano
Type: embedding
Input cost: $0.0025 USD per 1M tokens
Output cost: $0 USD per 1M tokens
--------------------------------
Name: vessel-embedding-mini
Type: embedding
Input cost: $0.009 USD per 1M tokens
Output cost: $0 USD per 1M tokens
--------------------------------
Name: vessel-embedding-small
Type: embedding
Input cost: $0.015 USD per 1M tokens
Output cost: $0 USD per 1M tokens
--------------------------------
Name: vessel-embedding-medium
Type: embedding
Input cost: $0.021 USD per 1M tokens
Output cost: $0 USD per 1M tokens
--------------------------------
Email: [manually redacted]
Credits Remaining (USD): [manually redacted]
Single embedding...
Success: True
Input tokens: 8
Embedding dimension: 768
Embedding: [2.8125, 0.3203125, -3.46875, 5.40625, 0.87890625, 1.78125, 1.671875, 0.0810546875, 0.26171875, 3.46875, 1.109375, -2.640625, -0.001068115234375, 2.171875, -0.95703125, 0.040771484375, 1.0625, -0.365234375, -1.84375, -2.25, 2.046875, 2.046875, 0.47265625, 3.046875, 0.9609375, -0.90234375, 3.046875, 2.671875, -0.6015625, -1.6171875, -0.419921875, -0.875, -1.8203125, -7.1875, 1.921875, -1.3984375, 2.328125, 3.203125, -0.310546875, -26.75, 0.49609375, 1.1796875, -0.71875, -0.55859375, -2.625, -0.6015625, -0.0986328125, 0.0771484375, -3.96875, 4.3125, 0.609375, 0.19921875, 0.9375, -3.65625, -1.8046875, -1.8359375, -3.03125, -0.2314453125, -4.6875, 0.69140625, 1.921875, -0.189453125, -0.4765625, -2.5625, -2.21875, -0.84765625, 1.5625, -17.75, 0.60546875, 1.875, 1.6796875, -0.58203125, 0.77734375, 0.224609375, -0.7578125, -0.390625, 2.65625, -7.0625, 0.267578125, 5.4375, 0.6640625, -2.375, -1.0078125, 4.21875, -0.37890625, -2.296875, 2.4375, -0.2001953125, -1.7109375, 0.31640625, -0.11962890625, -0.32421875, -0.87890625, -1.3359375, 2.78125, 0.47265625, -0.5546875, -0.1337890625, 1.2734375, 1.5, 1.1171875, -0.080078125, -0.55859375, 0.68359375, -0.625, 1.9296875, -3.484375, -0.10791015625, -0.98046875, -2.59375, 0.55859375, 3.0, -0.9765625, 2.0625, 0.1962890625, -0.51171875, 1.90625, 0.63671875, -2.609375, 0.546875, 0.234375, 1.796875, -1.703125, 3.734375, -2.34375, 5.4375, -3.9375, 1.0546875, -1.0859375, 0.0064697265625, 2.78125, 0.244140625, -0.5234375, -0.7578125, -2.5625, -0.390625, -1.71875, 0.298828125, -0.65234375, -4.25, -1.0625, 2.0625, 2.28125, -0.31640625, -1.6796875, 1.859375, -0.62890625, -0.50390625, 1.7109375, -2.390625, 0.63671875, 0.294921875, 2.84375, 1.1953125, 1.65625, 0.87109375, -0.90234375, -0.4296875, 1.140625, -1.4609375, -0.5546875, -2.1875, -1.265625, -3.359375, -0.5859375, -1.109375, -3.671875, 0.099609375, 1.5234375, -1.0234375, 3.40625, -1.3203125, 1.5234375, -0.283203125, -0.76171875, -0.0908203125, -0.380859375, -3.125, 2.03125, -0.051025390625, 0.388671875, -3.65625, -0.87890625, -0.373046875, 0.43359375, -2.578125, -3.28125, -0.9140625, 0.88671875, -0.81640625, -2.4375, 0.345703125, 4.53125, 0.77734375, 1.34375, 1.7109375, -2.046875, 0.1962890625, -0.76953125, -2.484375, 1.0234375, -0.2099609375, -6.6875, -1.3203125, -2.703125, -1.625, 1.296875, -1.796875, -0.34765625, 1.546875, 6.65625, -3.796875, 0.1748046875, 5.46875, 1.71875, -2.15625, 0.875, -0.41015625, -2.8125, 5.25, 1.921875, -2.015625, -1.6875, -1.7421875, 2.3125, 1.4609375, -0.81640625, 0.265625, 2.90625, 1.515625, 0.98046875, 1.75, -0.89453125, -0.0157470703125, 1.046875, 1.8671875, -2.21875, 0.24609375, -1.6640625, 0.609375, 2.96875, -1.6015625, 0.94140625, 1.1015625, -1.375, 1.2734375, -2.296875, -0.57421875, -0.017578125, -3.171875, -7.59375, 1.625, 0.057373046875, -2.71875, 0.10107421875, -0.1953125, 2.296875, -5.875, 2.71875, -0.2236328125, -1.0390625, 1.265625, 0.271484375, -3.328125, 0.94921875, -1.2734375, 1.6171875, 0.21484375, 0.90625, -1.546875, 1.296875, 0.0225830078125, -1.6484375, 1.15625, 1.625, 0.09423828125, -0.267578125, 2.484375, 3.765625, -0.78515625, 1.15625, 6.96875, -1.1328125, 3.6875, -1.4453125, -2.390625, -3.109375, 2.1875, -0.46875, 1.9453125, 3.0625, 0.1484375, -6.0625, 2.34375, 3.671875, -1.8203125, 2.203125, -5.6875, -2.28125, -1.6328125, -0.25390625, 2.03125, -2.046875, 0.236328125, -1.5703125, -1.59375, -0.44140625, -4.0, -1.8671875, 0.671875, 1.359375, 0.2041015625, -1.8359375, -2.75, -1.3515625, -3.71875, -0.125, 3.203125, 1.1171875, -0.76171875, -1.609375, 0.2421875, 0.6875, -3.578125, -0.0118408203125, 0.337890625, -2.609375, -0.01318359375, 1.8046875, 2.34375, 1.8984375, -2.125, 1.8828125, -5.90625, 1.2890625, -1.359375, -3.671875, -2.4375, 2.671875, -2.203125, -0.43359375, 1.6875, -3.375, 6.3125, -1.421875, 2.203125, -0.1357421875, -1.2109375, 0.04150390625, -1.8203125, -1.890625, 0.65234375, 2.078125, 8.5, -2.796875, 2.0, 1.96875, 0.53515625, -2.75, -2.34375, 1.1015625, -0.84765625, -0.91796875, 0.2353515625, 0.6875, 2.03125, -0.90234375, 0.2060546875, -4.53125, 0.04736328125, -1.2734375, 2.90625, -0.06884765625, -0.041015625, 5.40625, 1.6171875, -0.12060546875, 1.0390625, 0.5234375, -0.37890625, -4.59375, -0.28515625, -0.67578125, 0.8828125, 1.21875, -0.703125, 3.328125, 2.1875, -0.78515625, 0.890625, -0.90234375, -0.50390625, -3.453125, -0.423828125, 1.6953125, -1.0234375, 2.125, 1.2890625, -2.21875, 3.5625, 3.125, -0.189453125, 3.546875, 1.5703125, 4.75, -0.035888671875, 0.25390625, 51.5, 2.25, 2.359375, 1.953125, 3.578125, -0.0458984375, -3.078125, -1.8515625, 3.234375, -0.193359375, 4.9375, 0.8125, -4.84375, 0.8515625, -3.46875, 0.12060546875, 1.4921875, -0.224609375, 0.56640625, 1.6875, 4.09375, 0.234375, -0.423828125, -1.609375, 4.25, -1.3125, 3.09375, -0.42578125, -2.890625, 0.8046875, 0.5703125, -0.9453125, -0.1318359375, 0.76171875, 2.28125, -1.0546875, 0.92578125, 2.203125, -0.69921875, 0.12353515625, -5.0, 0.3984375, 0.58984375, -2.15625, 1.09375, -0.65234375, -2.375, -1.1171875, 1.1484375, 1.5859375, -2.921875, -2.34375, -2.765625, 1.921875, -1.8359375, 2.3125, -1.84375, -2.453125, -1.046875, 0.72265625, -3.109375, 1.296875, 2.078125, -2.828125, 1.875, -0.984375, -0.498046875, -1.640625, 1.921875, -3.09375, -4.78125, -0.361328125, 0.07080078125, 0.0174560546875, 0.828125, 0.2060546875, 1.421875, -0.306640625, -2.71875, 1.75, -0.2158203125, 2.890625, 3.46875, -0.435546875, 2.421875, 1.4453125, -0.59765625, 2.3125, -2.484375, 0.66796875, 0.33984375, -2.609375, -1.3359375, 4.28125, -0.578125, -0.76953125, 1.0546875, 0.0250244140625, 1.2890625, -2.109375, -0.95703125, -0.609375, -2.703125, 0.484375, 2.734375, 5.6875, 0.341796875, -4.5625, 1.09375, 1.796875, -2.609375, -0.30859375, 5.28125, 1.453125, 2.921875, -0.0341796875, 2.984375, 3.03125, 0.84765625, 0.045166015625, -1.8515625, -1.140625, -6.9375, -0.88671875, 1.0234375, -1.8046875, 0.99609375, 2.46875, -0.322265625, -1.3828125, -0.0218505859375, 0.81640625, 0.34375, 1.375, 2.828125, -2.578125, -3.1875, 2.921875, -1.1640625, 1.109375, 2.078125, 0.59375, 1.7578125, 0.134765625, -2.734375, 1.234375, -0.30859375, -1.375, -5.9375, -1.0390625, 2.5625, -2.359375, 1.390625, 2.25, 0.427734375, -2.359375, 0.193359375, 1.8984375, -0.125, 1.1796875, -2.171875, -0.70703125, 1.2109375, 1.265625, -0.50390625, 0.5625, 0.30078125, 1.9921875, -1.5859375, -1.2109375, -0.38671875, 4.40625, -2.21875, 1.265625, -1.171875, 4.28125, -0.1396484375, -2.921875, 4.9375, 0.057373046875, 0.486328125, 1.765625, -0.94921875, 0.9296875, -2.03125, 0.98046875, -4.28125, -0.51171875, 0.6484375, -4.03125, -1.5546875, -2.28125, -1.6171875, -1.3515625, 2.921875, 0.2412109375, 3.0625, 2.03125, 2.71875, -3.171875, -2.546875, -0.173828125, -3.359375, -0.84375, 0.390625, 0.53125, 2.03125, -1.4296875, -1.9609375, -1.2578125, -2.265625, -1.4765625, -2.28125, 2.1875, 0.8359375, 1.390625, 0.9140625, 1.03125, 1.3125, -0.41796875, -3.65625, 0.578125, 1.65625, -0.53515625, 0.79296875, 1.8046875, 0.412109375, -2.09375, 1.6015625, 3.328125, 0.498046875, 1.3203125, -0.404296875, 0.2119140625, -1.0234375, 7.59375, 0.45703125, -2.6875, -0.59375, -1.0859375, 2.921875, -3.8125, -0.8671875, 0.416015625, -2.375, 0.047119140625, -2.140625, 0.6484375, -0.875, 1.6328125, -0.828125, -6.6875, -2.453125, 1.515625, -1.765625, 1.4375, 1.6015625, -2.671875, 0.6875, 0.55078125, 4.46875, 0.92578125, 0.59375, -1.359375, 0.7890625, 1.390625, -2.171875, -0.84375, 4.28125, -1.59375, -2.640625, 0.94140625, -2.453125, -1.875, 5.90625, -0.90625, -1.8203125, -2.953125, -2.859375, -3.4375, -2.53125, 1.6953125, -4.40625, 1.2734375, 1.15625, -4.78125, -0.89453125, 2.625, 0.263671875, 0.21484375, 2.84375, -2.96875, 2.546875, 0.546875, -0.6015625, -1.578125, 3.234375, 3.28125, 1.0625, -0.86328125, 3.34375, -0.2373046875, -0.9296875, -0.39453125, -0.12109375, -0.2333984375, 2.9375, 1.9609375, -1.6015625, -0.64453125, -2.21875, 2.21875, 1.3359375, -0.66015625, -1.484375, 0.032470703125, -0.294921875, 0.0146484375, 1.4609375, -0.9609375, -1.0546875, -2.546875, 1.6875, 0.51171875, 5.96875, 0.57421875, -0.154296875, -3.484375, -2.28125, 3.96875, -0.380859375, 0.008056640625, 1.3046875, 1.828125, 1.0625, 0.1357421875, -1.6328125, 0.73046875, -1.3125, 3.359375, 0.1708984375, -2.828125, -0.28515625, 3.5625, 3.0, -5.03125, -3.90625, 2.921875, -1.9765625, -9.0, 0.3984375, -1.21875, 0.234375, 0.83203125, 0.3671875, 2.40625, 1.4609375, -0.09521484375, -2.28125, 1.140625, 1.796875]
Reported throughput: 0.92709 tokens/sec
Processing time: 8.62914 seconds
Cost: $0.000000020 USD
Exact Cost: 2e-08 USD
Batch embeddings...
Success: True
Input tokens: 40
Embedding dimension: 768
Embeddings Generated: 5
Reported throughput: 2479.85121 tokens/sec
Processing time: 0.01613 seconds
Cost: $0.00000010 USD
Exact Cost: 1.0000000000000001e-07 USD
Single chat completion...
Success: True
Input tokens: 21
Output tokens: 27
Response: I'm doing well, thank you! Just running smooth through my systems today. How about you? What brings you here today?
Reported throughput: 5.58023 tokens/sec
Processing time: 8.6018 seconds
Input cost: $0.00000014 USD
Output cost: $0.00000130 USD
Total cost: $0.00000144 USD
Exact Cost: 1.4865e-06 USD
Batch chat completions
Success: True
Input tokens: 260
Output tokens: 182
Responses: [
    {
        "message": {
            "role": "assistant",
            "content": "I'm doing well, thank you for asking! As a friendly and helpful assistant created by the Vessel Platform team, I'm here to assist you with any questions or information you need. How can I help you today?"
        },
        "index": 0
    },
    {
        "message": {
            "role": "assistant",
            "content": "France's capital is Paris, a beautiful city known for its art, fashion, gastronomy, and historical landmarks like the Eiffel Tower and Notre-Dame Cathedral. It's located in the northern central part of the country. As this information is widely known, I can provide it now! If you have any more questions about France or anything else, feel free to ask."
        },
        "index": 1
    },
    {
        "message": {
            "role": "assistant",
            "content": "The capital of Germany is Berlin. It's a vibrant city known for its rich history, culture, and modern attractions like museums, art galleries, and its famous Brandenburg Gate. If you're looking to visit in November, October often offers pleasant weather despite it being autumn. Enjoy your trip!"
        },
        "index": 2
    }
]
Reported throughput: 561.07747 tokens/sec
Processing time: 0.78777 seconds
Input cost: $0.0000017 USD
Output cost: $0.0000091 USD
Total cost: $0.0000108 USD
Exact Cost: 1.079e-05 USD
Batch chat completions (async)...
Waiting for batch to complete... (current status: queued)
Waiting for batch to complete... (current status: queued)
Waiting for batch to complete... (current status: queued)
Waiting for batch to complete... (current status: in_progress)
Waiting for batch to complete... (current status: in_progress)
Batch completed! (current status: completed)
Success: True
Input tokens: 618
Output tokens: 21
Responses: [
    {
        "message": {
            "role": "assistant",
            "content": "The capital of France is Paris."
        },
        "index": 0
    },
    {
        "message": {
            "role": "assistant",
            "content": "Berlin"
        },
        "index": 1
    },
    {
        "message": {
            "role": "assistant",
            "content": "Rome"
        },
        "index": 2
    },
    {
        "message": {
            "role": "assistant",
            "content": "Madrid"
        },
        "index": 3
    },
    {
        "message": {
            "role": "assistant",
            "content": "The capital of Portugal is Lisbon."
        },
        "index": 4
    },
    {
        "message": {
            "role": "assistant",
            "content": "Athens"
        },
        "index": 5
    }
]
Reported throughput: 771.11516 tokens/sec
Processing time: 0.82867 seconds
Input cost: $0.0000074 USD
Output cost: $0.0000026 USD
Total cost: $0.0000100 USD
Exact Cost: 1.0041000000000001e-05 USD
Batch embeddings (async)...
Waiting for batch to complete... (current status: queued)
Waiting for batch to complete... (current status: queued)
Waiting for batch to complete... (current status: in_progress)
Waiting for batch to complete... (current status: in_progress)
Batch completed! (current status: completed)
Success: True
Input tokens: 43
Embedding dimension: 768
Embeddings Generated: 5
Reported throughput: 2501.45433 tokens/sec
Processing time: 0.01719 seconds
Cost: $0.00000011 USD
Exact Cost: 1.075e-07 USD
"""