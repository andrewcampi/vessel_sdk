"""Embeddings resource for the Vessel SDK."""

from typing import TYPE_CHECKING, Union, List
import json
import time
import re
from ..types import EmbeddingsResponse, DictLike, format_costs_together, EmbeddingsBatchJob

if TYPE_CHECKING:
    from openai import OpenAI
    from ..client import RateLimiter


class Embeddings:
    """Resource for creating embeddings."""
    
    def __init__(self, client: "OpenAI", rate_limiter: "RateLimiter"):
        self._client = client
        self._rate_limiter = rate_limiter
    
    def _extract_wait_time_from_error(self, error_message: str) -> float:
        """Extract wait time in seconds from rate limit error message.
        
        Args:
            error_message: The error message containing wait time.
        
        Returns:
            Wait time in seconds, or 5.0 as default.
        """
        # Try to find patterns like "wait X seconds" or "wait X second"
        match = re.search(r'wait (\d+(?:\.\d+)?)\s*seconds?', error_message, re.IGNORECASE)
        if match:
            return float(match.group(1))
        return 5.0  # Default fallback
    
    def _create_batch_with_retry(self, input_file_id: str, endpoint: str, metadata: dict, max_retries: int = 3) -> object:
        """Create a batch job with automatic retry on rate limits.
        
        Args:
            input_file_id: The input file ID for the batch.
            endpoint: The API endpoint to use.
            metadata: Metadata for the batch job.
            max_retries: Maximum number of retry attempts.
        
        Returns:
            The created batch job object.
        
        Raises:
            Exception: If all retries are exhausted.
        """
        from openai import PermissionDeniedError
        
        for attempt in range(max_retries):
            try:
                batch_job = self._client.batches.create(
                    input_file_id=input_file_id,
                    endpoint=endpoint,
                    completion_window="24h",
                    metadata=metadata
                )
                return batch_job
            
            except PermissionDeniedError as e:
                error_str = str(e)
                
                # Check if this is a rate limit error
                if 'rate limit' in error_str.lower() or 'rate_limit' in error_str.lower():
                    # Extract wait time from error message
                    wait_time = self._extract_wait_time_from_error(error_str)
                    
                    # Add a small buffer (0.5 seconds)
                    wait_time_with_buffer = wait_time + 0.5
                    
                    if attempt < max_retries - 1:
                        print(f"⏳ Rate limit hit. Waiting {wait_time_with_buffer:.1f} seconds before retry (attempt {attempt + 1}/{max_retries})...")
                        time.sleep(wait_time_with_buffer)
                    else:
                        # Last attempt failed
                        print(f"❌ Rate limit error persists after {max_retries} attempts")
                        raise
                else:
                    # Not a rate limit error, re-raise immediately
                    raise
            
            except Exception as e:
                # For other exceptions, re-raise immediately
                raise
        
        # Should never reach here, but just in case
        raise Exception("Failed to create batch after all retries")
    
    def _get_model_pricing(self, model: str) -> tuple[float, float]:
        """Get pricing for a model.
        
        Args:
            model: The model name.
        
        Returns:
            Tuple of (input_cost, output_cost) per 1M tokens.
        """
        try:
            models = self._client.models.list()
            for m in models.data:
                if m.id == model:
                    pricing = getattr(m, 'pricing', {})
                    if pricing:
                        return pricing.get('input', 0.0), pricing.get('output', 0.0)
            return 0.0, 0.0
        except:
            return 0.0, 0.0
    
    def create(
        self,
        model: str,
        input: Union[str, List[str]],
        verbose: bool = False
    ) -> EmbeddingsResponse:
        """Create embeddings for the given input.
        
        Args:
            model: The model to use for embeddings.
            input: A string or list of strings to embed.
            verbose: Whether to print progress information.
        
        Returns:
            EmbeddingsResponse: Response containing embedding data and metrics.
        """
        try:
            # Wait for rate limit
            self._rate_limiter.wait()
            
            # Normalize input to list
            if isinstance(input, str):
                input_list = [input]
            else:
                input_list = input
            
            # Get model pricing
            input_cost_per_1m, output_cost_per_1m = self._get_model_pricing(model)
            
            # If single input or small batch, use direct API
            if len(input_list) == 1:
                if verbose:
                    print("Creating single embedding...")
                
                start_time = time.time()
                response = self._client.embeddings.create(
                    model=model,
                    input=input_list[0]
                )
                end_time = time.time()
                
                processing_time = round(end_time - start_time, 5)
                input_tokens = response.usage.prompt_tokens
                throughput = round(input_tokens / processing_time, 5) if processing_time > 0 else 0
                
                # Calculate costs (pricing is per 1M tokens)
                input_cost_value = input_tokens * input_cost_per_1m / 1_000_000
                output_cost_value = 0.0  # Embeddings don't have output tokens
                exact_cost_value = input_cost_value + output_cost_value
                
                # Format costs (total is calculated as rounded input + rounded output)
                input_cost, output_cost, total_cost = format_costs_together(
                    input_cost_value, output_cost_value
                )
                
                # Use DictLike for JSON serialization support
                data = [DictLike(embedding=response.data[0].embedding, index=0)]
                
                if verbose:
                    print(f"✅ Completed in {processing_time:.2f} seconds")
                
                return EmbeddingsResponse(
                    success=True,
                    data=data,
                    input_tokens=input_tokens,
                    throughput=throughput,
                    processing_time=processing_time,
                    input_cost=input_cost,
                    output_cost=output_cost,
                    total_cost=total_cost,
                    exact_cost=exact_cost_value,
                    error=None
                )
            
            # For batch processing
            if verbose:
                print(f"Creating batch of {len(input_list)} embeddings...")
            
            # Create batch tasks
            batch_tasks = []
            for i, text in enumerate(input_list):
                task = {
                    "custom_id": f"embed_task_{i}",
                    "method": "POST",
                    "url": "/v1/embeddings",
                    "body": {
                        "model": model,
                        "input": text
                    }
                }
                batch_tasks.append(task)
            
            # Write to temporary file
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
                batch_filename = f.name
                for task in batch_tasks:
                    f.write(json.dumps(task) + "\n")
            
            if verbose:
                print("Uploading batch file...")
            
            # Upload batch file
            with open(batch_filename, "rb") as f:
                batch_input_file = self._client.files.create(
                    file=f,
                    purpose="batch"
                )
            
            if verbose:
                print(f"Creating batch job (ID: {batch_input_file.id})...")
            
            # Create batch job with retry logic
            start_time = time.time()
            batch_job = self._create_batch_with_retry(
                input_file_id=batch_input_file.id,
                endpoint="/v1/embeddings",
                metadata={"description": "Vessel SDK batch embeddings"}
            )
            
            batch_id = batch_job.id
            
            if verbose:
                print("Waiting for batch to complete...")
            
            # Wait for completion
            last_status = None
            while True:
                batch = self._client.batches.retrieve(batch_id)
                
                if verbose and batch.status != last_status:
                    print(f"  Status: {batch.status}")
                    if hasattr(batch, 'request_counts') and batch.request_counts:
                        counts = batch.request_counts
                        print(f"    Completed: {counts.completed}/{counts.total}")
                    last_status = batch.status
                
                if batch.status == "completed":
                    end_time = time.time()
                    break
                elif batch.status == "failed":
                    error_msg = "Batch processing failed"
                    if hasattr(batch, 'errors') and batch.errors:
                        error_msg += f": {batch.errors}"
                    return EmbeddingsResponse(
                        success=False,
                        data=[],
                        input_tokens=0,
                        throughput=0,
                        processing_time=0,
                        error=error_msg
                    )
                else:
                    time.sleep(2)
            
            # Download results
            result_file_id = batch.output_file_id
            result_content = self._client.files.content(result_file_id).content
            
            # Parse results
            results = []
            for line in result_content.decode('utf-8').strip().split('\n'):
                if line:
                    results.append(json.loads(line))
            
            # Extract embeddings
            embeddings_data = []
            total_tokens = 0
            for result in results:
                embedding_vector = result['response']['body']['data'][0]['embedding']
                index = int(result['custom_id'].split('_')[-1])
                # Use DictLike for JSON serialization support
                embeddings_data.append(DictLike(embedding=embedding_vector, index=index))
                total_tokens += result['response']['body']['usage']['prompt_tokens']
            
            # Sort by index to maintain order
            embeddings_data.sort(key=lambda x: x.index)
            
            # Calculate metrics
            processing_time = round(getattr(batch, 'processing_time', end_time - start_time), 5)
            throughput = round(total_tokens / processing_time, 5) if processing_time > 0 else 0
            
            # Calculate costs (pricing is per 1M tokens)
            input_cost_value = total_tokens * input_cost_per_1m / 1_000_000
            output_cost_value = 0.0  # Embeddings don't have output tokens
            exact_cost_value = input_cost_value + output_cost_value
            
            # Format costs (total is calculated as rounded input + rounded output)
            input_cost, output_cost, total_cost = format_costs_together(
                input_cost_value, output_cost_value
            )
            
            if verbose:
                print(f"✅ Batch completed successfully")
                print(f"   Processing time: {processing_time:.2f} seconds")
                print(f"   Throughput: {throughput:.2f} tokens/sec")
            
            # Cleanup temp file
            import os
            try:
                os.unlink(batch_filename)
            except:
                pass
            
            return EmbeddingsResponse(
                success=True,
                data=embeddings_data,
                input_tokens=total_tokens,
                throughput=throughput,
                processing_time=processing_time,
                input_cost=input_cost,
                output_cost=output_cost,
                total_cost=total_cost,
                exact_cost=exact_cost_value,
                error=None
            )
            
        except Exception as e:
            return EmbeddingsResponse(
                success=False,
                data=[],
                input_tokens=0,
                throughput=0,
                processing_time=0,
                error=str(e)
            )
    
    def create_async(
        self,
        model: str,
        input: List[str]
    ) -> EmbeddingsBatchJob:
        """Create an asynchronous batch job for embeddings.
        
        This method creates a batch job and returns immediately without waiting.
        The returned EmbeddingsBatchJob object can be used to poll status and retrieve results.
        
        Args:
            model: The model to use for embeddings.
            input: A list of strings to embed.
        
        Returns:
            EmbeddingsBatchJob: A batch job object with .id, .poll(), and .get() methods.
        
        Raises:
            Exception: If the batch job creation fails.
        """
        # Wait for rate limit
        self._rate_limiter.wait()
        
        # Get model pricing
        input_cost_per_1m, output_cost_per_1m = self._get_model_pricing(model)
        
        # Create batch tasks
        batch_tasks = []
        for i, text in enumerate(input):
            task = {
                "custom_id": f"embed_task_{i}",
                "method": "POST",
                "url": "/v1/embeddings",
                "body": {
                    "model": model,
                    "input": text
                }
            }
            batch_tasks.append(task)
        
        # Write to temporary file
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            batch_filename = f.name
            for task in batch_tasks:
                f.write(json.dumps(task) + "\n")
        
        # Upload batch file
        with open(batch_filename, "rb") as f:
            batch_input_file = self._client.files.create(
                file=f,
                purpose="batch"
            )
        
        # Create batch job with retry logic
        batch_job = self._create_batch_with_retry(
            input_file_id=batch_input_file.id,
            endpoint="/v1/embeddings",
            metadata={"description": "Vessel SDK batch embeddings"}
        )
        
        # Cleanup temp file
        import os
        try:
            os.unlink(batch_filename)
        except:
            pass
        
        # Return batch job object
        return EmbeddingsBatchJob(
            batch_id=batch_job.id,
            client=self._client,
            model=model,
            input_cost_per_1m=input_cost_per_1m,
            num_requests=len(input)
        )

