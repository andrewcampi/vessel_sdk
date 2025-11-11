"""Chat resource for the Vessel SDK."""

from typing import TYPE_CHECKING, List, Dict, Union
import json
import time
import re
from ..types import ChatCompletionsResponse, DictLike, format_costs_together, ChatBatchJob

if TYPE_CHECKING:
    from openai import OpenAI
    from ..client import RateLimiter


class Completions:
    """Completions sub-resource for chat."""
    
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
        messages: Union[List[Dict[str, str]], List[List[Dict[str, str]]]],
        verbose: bool = False,
        **kwargs
    ) -> ChatCompletionsResponse:
        """Create chat completions for the given messages.
        
        Args:
            model: The model to use for chat completions.
            messages: A list of message dicts or list of lists of message dicts for batch.
            verbose: Whether to print progress information.
            **kwargs: Additional arguments to pass to the API (e.g., max_tokens, temperature).
        
        Returns:
            ChatCompletionsResponse: Response containing chat completion data and metrics.
        """
        try:
            # Wait for rate limit
            self._rate_limiter.wait()
            
            # Get model pricing
            input_cost_per_1m, output_cost_per_1m = self._get_model_pricing(model)
            
            # Determine if this is a single request or batch
            is_batch = isinstance(messages[0], list)
            
            if not is_batch:
                # Single chat completion
                if verbose:
                    print("Creating single chat completion...")
                
                start_time = time.time()
                response = self._client.chat.completions.create(
                    model=model,
                    messages=messages,
                    **kwargs
                )
                end_time = time.time()
                
                processing_time = round(end_time - start_time, 5)
                input_tokens = response.usage.prompt_tokens
                output_tokens = response.usage.completion_tokens
                total_tokens = response.usage.total_tokens
                throughput = round(total_tokens / processing_time, 5) if processing_time > 0 else 0
                
                # Calculate costs (pricing is per 1M tokens)
                input_cost_value = input_tokens * input_cost_per_1m / 1_000_000
                output_cost_value = output_tokens * output_cost_per_1m / 1_000_000
                exact_cost_value = input_cost_value + output_cost_value
                
                # Format costs (total is calculated as rounded input + rounded output)
                input_cost, output_cost, total_cost = format_costs_together(
                    input_cost_value, output_cost_value
                )
                
                # Use DictLike for JSON serialization support
                message = DictLike(
                    role=response.choices[0].message.role,
                    content=response.choices[0].message.content
                )
                data = [DictLike(message=message, index=0)]
                
                if verbose:
                    print(f"✅ Completed in {processing_time:.2f} seconds")
                
                return ChatCompletionsResponse(
                    success=True,
                    data=data,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    throughput=throughput,
                    processing_time=processing_time,
                    input_cost=input_cost,
                    output_cost=output_cost,
                    total_cost=total_cost,
                    exact_cost=exact_cost_value,
                    error=None
                )
            
            # Batch processing
            messages_list = messages
            if verbose:
                print(f"Creating batch of {len(messages_list)} chat completions...")
            
            # Create batch tasks
            batch_tasks = []
            for i, msg_list in enumerate(messages_list):
                task = {
                    "custom_id": f"chat_task_{i}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": model,
                        "messages": msg_list,
                        **kwargs
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
                endpoint="/v1/chat/completions",
                metadata={"description": "Vessel SDK batch chat completions"}
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
                    return ChatCompletionsResponse(
                        success=False,
                        data=[],
                        input_tokens=0,
                        output_tokens=0,
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
            
            # Extract chat completions
            chat_data = []
            total_input_tokens = 0
            total_output_tokens = 0
            for result in results:
                message_content = result['response']['body']['choices'][0]['message']['content']
                message_role = result['response']['body']['choices'][0]['message']['role']
                index = int(result['custom_id'].split('_')[-1])
                
                # Use DictLike for JSON serialization support
                message = DictLike(role=message_role, content=message_content)
                chat_choice = DictLike(message=message, index=index)
                chat_data.append(chat_choice)
                
                total_input_tokens += result['response']['body']['usage']['prompt_tokens']
                total_output_tokens += result['response']['body']['usage']['completion_tokens']
            
            # Sort by index to maintain order
            chat_data.sort(key=lambda x: x.index)
            
            # Calculate metrics
            processing_time = round(getattr(batch, 'processing_time', end_time - start_time), 5)
            total_tokens = total_input_tokens + total_output_tokens
            throughput = round(total_tokens / processing_time, 5) if processing_time > 0 else 0
            
            # Calculate costs (pricing is per 1M tokens)
            input_cost_value = total_input_tokens * input_cost_per_1m / 1_000_000
            output_cost_value = total_output_tokens * output_cost_per_1m / 1_000_000
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
            
            return ChatCompletionsResponse(
                success=True,
                data=chat_data,
                input_tokens=total_input_tokens,
                output_tokens=total_output_tokens,
                throughput=throughput,
                processing_time=processing_time,
                input_cost=input_cost,
                output_cost=output_cost,
                total_cost=total_cost,
                exact_cost=exact_cost_value,
                error=None
            )
            
        except Exception as e:
            return ChatCompletionsResponse(
                success=False,
                data=[],
                input_tokens=0,
                output_tokens=0,
                throughput=0,
                processing_time=0,
                error=str(e)
            )
    
    def create_async(
        self,
        model: str,
        messages: List[List[Dict[str, str]]],
        **kwargs
    ) -> ChatBatchJob:
        """Create an asynchronous batch job for chat completions.
        
        This method creates a batch job and returns immediately without waiting.
        The returned ChatBatchJob object can be used to poll status and retrieve results.
        
        Args:
            model: The model to use for chat completions.
            messages: A list of lists of message dicts for batch processing.
            **kwargs: Additional arguments to pass to the API (e.g., max_tokens, temperature).
        
        Returns:
            ChatBatchJob: A batch job object with .id, .poll(), and .get() methods.
        
        Raises:
            Exception: If the batch job creation fails.
        """
        # Wait for rate limit
        self._rate_limiter.wait()
        
        # Get model pricing
        input_cost_per_1m, output_cost_per_1m = self._get_model_pricing(model)
        
        # Create batch tasks
        batch_tasks = []
        for i, msg_list in enumerate(messages):
            task = {
                "custom_id": f"chat_task_{i}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": model,
                    "messages": msg_list,
                    **kwargs
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
            endpoint="/v1/chat/completions",
            metadata={"description": "Vessel SDK batch chat completions"}
        )
        
        # Cleanup temp file
        import os
        try:
            os.unlink(batch_filename)
        except:
            pass
        
        # Return batch job object
        return ChatBatchJob(
            batch_id=batch_job.id,
            client=self._client,
            model=model,
            input_cost_per_1m=input_cost_per_1m,
            output_cost_per_1m=output_cost_per_1m,
            num_requests=len(messages)
        )


class Chat:
    """Resource for chat completions."""
    
    def __init__(self, client: "OpenAI", rate_limiter: "RateLimiter"):
        self.completions = Completions(client, rate_limiter)

