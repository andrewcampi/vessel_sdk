"""Classify/AI Detection resource for the Vessel SDK."""

from typing import TYPE_CHECKING, List
import json
import time
import re
from ..types import ClassifyResponse, DictLike, format_costs_together, ClassifyBatchJob

if TYPE_CHECKING:
    from openai import OpenAI
    from ..client import RateLimiter


class Classify:
    """Resource for AI detection classification."""
    
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
        texts: List[str],
        verbose: bool = False
    ) -> ClassifyResponse:
        """Classify texts for AI detection.
        
        Args:
            model: The model to use for classification (e.g., "vessel-detect-large").
            texts: A list of strings to classify (up to 64 texts).
            verbose: Whether to print progress information.
        
        Returns:
            ClassifyResponse: Response containing classification results and metrics.
        """
        try:
            # Wait for rate limit
            self._rate_limiter.wait()
            
            # Validate input
            if not texts or len(texts) == 0:
                raise ValueError("At least one text is required")
            
            if len(texts) > 64:
                raise ValueError("Maximum 64 texts allowed per request")
            
            # Get model pricing
            input_cost_per_1m, output_cost_per_1m = self._get_model_pricing(model)
            
            # Make direct API call
            if verbose:
                print(f"Classifying {len(texts)} text(s)...")
            
            # Make request using the OpenAI client's base request method
            # Since there's no built-in classify method, we'll use batch processing
            start_time = time.time()
            
            # Create batch tasks
            batch_tasks = []
            for i, text in enumerate(texts):
                task = {
                    "custom_id": f"classify_task_{i}",
                    "method": "POST",
                    "url": "/v1/classify",
                    "body": {
                        "model": model,
                        "texts": [text]
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
            batch_job = self._create_batch_with_retry(
                input_file_id=batch_input_file.id,
                endpoint="/v1/classify",
                metadata={"description": "Vessel SDK batch classification"}
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
                    return ClassifyResponse(
                        success=False,
                        classifications=[],
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
            
            # Extract classifications
            classifications_data = []
            total_tokens = 0
            for result in results:
                classification_list = result['response']['body']['classifications']
                # Should be a single classification per result (since we sent one text per task)
                if classification_list:
                    classifications_data.append(DictLike(classification_list[0]))
                total_tokens += result['response']['body']['usage']['prompt_tokens']
            
            # Calculate metrics
            processing_time = round(getattr(batch, 'processing_time', end_time - start_time), 5)
            throughput = round(total_tokens / processing_time, 5) if processing_time > 0 else 0
            
            # Calculate costs (pricing is per 1M tokens)
            input_cost_value = total_tokens * input_cost_per_1m / 1_000_000
            output_cost_value = 0.0  # Classification doesn't have output tokens
            exact_cost_value = input_cost_value + output_cost_value
            
            # Format costs
            input_cost, output_cost, total_cost = format_costs_together(
                input_cost_value, output_cost_value
            )
            
            if verbose:
                print(f"✅ Classification completed successfully")
                print(f"   Processing time: {processing_time:.2f} seconds")
                print(f"   Throughput: {throughput:.2f} tokens/sec")
            
            # Cleanup temp file
            import os
            try:
                os.unlink(batch_filename)
            except:
                pass
            
            return ClassifyResponse(
                success=True,
                classifications=classifications_data,
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
            return ClassifyResponse(
                success=False,
                classifications=[],
                input_tokens=0,
                throughput=0,
                processing_time=0,
                error=str(e)
            )
    
    def create_async(
        self,
        model: str,
        texts: List[str]
    ) -> ClassifyBatchJob:
        """Create an asynchronous batch job for classification.
        
        This method creates a batch job and returns immediately without waiting.
        The returned ClassifyBatchJob object can be used to poll status and retrieve results.
        
        Args:
            model: The model to use for classification.
            texts: A list of strings to classify.
        
        Returns:
            ClassifyBatchJob: A batch job object with .id, .poll(), and .get() methods.
        
        Raises:
            Exception: If the batch job creation fails.
        """
        # Wait for rate limit
        self._rate_limiter.wait()
        
        # Validate input
        if not texts or len(texts) == 0:
            raise ValueError("At least one text is required")
        
        # Get model pricing
        input_cost_per_1m, output_cost_per_1m = self._get_model_pricing(model)
        
        # Create batch tasks
        batch_tasks = []
        for i, text in enumerate(texts):
            task = {
                "custom_id": f"classify_task_{i}",
                "method": "POST",
                "url": "/v1/classify",
                "body": {
                    "model": model,
                    "texts": [text]
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
            endpoint="/v1/classify",
            metadata={"description": "Vessel SDK batch classification"}
        )
        
        # Cleanup temp file
        import os
        try:
            os.unlink(batch_filename)
        except:
            pass
        
        # Return batch job object
        return ClassifyBatchJob(
            batch_id=batch_job.id,
            client=self._client,
            model=model,
            input_cost_per_1m=input_cost_per_1m,
            num_requests=len(texts)
        )
