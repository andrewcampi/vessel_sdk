"""Type definitions for the Vessel SDK."""

from typing import List, Dict, Any, Optional, TYPE_CHECKING
from dataclasses import dataclass, asdict
import json
import math

if TYPE_CHECKING:
    from openai import OpenAI


def round_cost(cost: float) -> float:
    """Round cost, keeping enough precision for accurate display.
    
    Args:
        cost: The cost in USD as a float.
    
    Returns:
        Rounded cost as a float.
    """
    if cost == 0:
        return 0.0
    if cost >= 0.01:
        return round(cost, 2)
    
    # For costs smaller than a cent, round to 2 places after the first non-zero digit
    # This ensures math adds up visually (e.g., 0.000002 + 0.000009 = 0.000011)
    decimal_places = -math.floor(math.log10(abs(cost))) + 1
    return round(cost, decimal_places)


def format_cost(cost: float) -> str:
    """Format cost for display with appropriate precision.
    
    Args:
        cost: The cost in USD as a float.
    
    Returns:
        Formatted cost string (e.g., "$0.00", "$0.000011").
    """
    if cost == 0:
        return "$0.00"
    if cost >= 0.01:
        return f"${cost:.2f}"
    
    # For costs smaller than a cent, show 2 places after the first non-zero digit
    # This ensures the math adds up visually when displaying input + output = total
    decimal_places = -math.floor(math.log10(abs(cost))) + 1
    return f"${cost:.{decimal_places}f}"


def format_costs_together(input_cost: float, output_cost: float) -> tuple[str, str, str]:
    """Format input, output, and total costs so the math visually adds up.
    
    Rounds input and output individually, then calculates total as their sum.
    All three values are formatted with the same precision.
    
    Args:
        input_cost: Input cost in USD.
        output_cost: Output cost in USD.
    
    Returns:
        Tuple of formatted (input_cost, output_cost, total_cost) strings.
    """
    # Round each cost individually
    rounded_input = round_cost(input_cost)
    rounded_output = round_cost(output_cost)
    
    # Total is the sum of the rounded values
    total = rounded_input + rounded_output
    
    # Determine the precision needed (use the smallest non-zero value)
    costs = [c for c in [rounded_input, rounded_output, total] if c > 0]
    if not costs:
        return "$0.00", "$0.00", "$0.00"
    
    # If all costs >= $0.01, use standard 2 decimal places
    if min(costs) >= 0.01:
        return f"${rounded_input:.2f}", f"${rounded_output:.2f}", f"${total:.2f}"
    
    # For smaller costs, find precision based on smallest value
    # Use 2 decimal places after the first non-zero digit
    min_cost = min(costs)
    decimal_places = -math.floor(math.log10(abs(min_cost))) + 1
    
    # Format all three with the same precision
    return (
        f"${rounded_input:.{decimal_places}f}",
        f"${rounded_output:.{decimal_places}f}",
        f"${total:.{decimal_places}f}"
    )


class DictLike(dict):
    """A dict that also supports attribute access."""
    
    def __getattr__(self, key):
        """Get attribute."""
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{key}'")
    
    def __setattr__(self, key, value):
        """Set attribute."""
        self[key] = value
    
    def __delattr__(self, key):
        """Delete attribute."""
        try:
            del self[key]
        except KeyError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{key}'")


@dataclass
class Model:
    """Represents a model in the Vessel API."""
    name: str
    type: str
    input_cost: float
    output_cost: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class ModelsListResponse:
    """Response from the models.list() endpoint."""
    data: List[Model]


@dataclass
class Account:
    """Represents an account in the Vessel API."""
    email: str
    credits: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class AccountRetrieveResponse:
    """Response from the accounts.retrieve() endpoint."""
    data: Account


@dataclass
class EmbeddingData:
    """Represents a single embedding result."""
    embedding: List[float]
    index: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class ChatMessage:
    """Represents a chat message."""
    role: str
    content: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class ChatChoice:
    """Represents a chat completion choice."""
    message: ChatMessage
    index: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class EmbeddingsResponse:
    """Response from the embeddings.create() endpoint."""
    success: bool
    data: List[EmbeddingData]
    input_tokens: int
    throughput: float
    processing_time: float
    input_cost: str = "$0.00"
    output_cost: str = "$0.00"
    total_cost: str = "$0.00"
    exact_cost: float = 0.0
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'success': self.success,
            'data': [item.to_dict() for item in self.data],
            'input_tokens': self.input_tokens,
            'throughput': self.throughput,
            'processing_time': self.processing_time,
            'input_cost': self.input_cost,
            'output_cost': self.output_cost,
            'total_cost': self.total_cost,
            'exact_cost': self.exact_cost,
            'error': self.error
        }


@dataclass
class ChatCompletionsResponse:
    """Response from the chat.completions.create() endpoint."""
    success: bool
    data: List[ChatChoice]
    input_tokens: int
    output_tokens: int
    throughput: float
    processing_time: float
    input_cost: str = "$0.00"
    output_cost: str = "$0.00"
    total_cost: str = "$0.00"
    exact_cost: float = 0.0
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'success': self.success,
            'data': [item.to_dict() for item in self.data],
            'input_tokens': self.input_tokens,
            'output_tokens': self.output_tokens,
            'throughput': self.throughput,
            'processing_time': self.processing_time,
            'input_cost': self.input_cost,
            'output_cost': self.output_cost,
            'total_cost': self.total_cost,
            'exact_cost': self.exact_cost,
            'error': self.error
        }


@dataclass
class ClassifyResponse:
    """Response from the classify.create() endpoint."""
    success: bool
    classifications: List[DictLike]  # Each item has is_ai and confidence
    input_tokens: int
    throughput: float
    processing_time: float
    input_cost: str = "$0.00"
    output_cost: str = "$0.00"
    total_cost: str = "$0.00"
    exact_cost: float = 0.0
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'success': self.success,
            'classifications': [dict(item) for item in self.classifications],
            'input_tokens': self.input_tokens,
            'throughput': self.throughput,
            'processing_time': self.processing_time,
            'input_cost': self.input_cost,
            'output_cost': self.output_cost,
            'total_cost': self.total_cost,
            'exact_cost': self.exact_cost,
            'error': self.error
        }


@dataclass
class BatchStatus:
    """Status information for a batch job."""
    finished: bool
    status: str


class ChatBatchJob:
    """Represents an asynchronous batch job for chat completions."""
    
    def __init__(
        self,
        batch_id: str,
        client: "OpenAI",
        model: str,
        input_cost_per_1m: float,
        output_cost_per_1m: float,
        num_requests: int
    ):
        """Initialize a chat batch job.
        
        Args:
            batch_id: The batch job ID.
            client: The OpenAI client instance.
            model: The model name.
            input_cost_per_1m: Input cost per 1M tokens.
            output_cost_per_1m: Output cost per 1M tokens.
            num_requests: Number of requests in the batch.
        """
        self.id = batch_id
        self._client = client
        self._model = model
        self._input_cost_per_1m = input_cost_per_1m
        self._output_cost_per_1m = output_cost_per_1m
        self._num_requests = num_requests
        self._cached_status: Optional[BatchStatus] = None
    
    def poll(self) -> BatchStatus:
        """Poll the batch job status.
        
        Returns:
            BatchStatus: Current status of the batch job.
        """
        import time
        batch = self._client.batches.retrieve(self.id)
        finished = batch.status in ["completed", "failed", "cancelled"]
        self._cached_status = BatchStatus(finished=finished, status=batch.status)
        return self._cached_status
    
    def get(self) -> ChatCompletionsResponse:
        """Get the results of the batch job.
        
        This method blocks until the batch is complete.
        
        Returns:
            ChatCompletionsResponse: The response containing all chat completions.
        """
        import time
        
        # Wait for completion
        while True:
            batch = self._client.batches.retrieve(self.id)
            
            if batch.status == "completed":
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
            elif batch.status == "cancelled":
                return ChatCompletionsResponse(
                    success=False,
                    data=[],
                    input_tokens=0,
                    output_tokens=0,
                    throughput=0,
                    processing_time=0,
                    error="Batch processing was cancelled"
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
        processing_time = round(getattr(batch, 'processing_time', 0), 5)
        total_tokens = total_input_tokens + total_output_tokens
        throughput = round(total_tokens / processing_time, 5) if processing_time > 0 else 0
        
        # Calculate costs (pricing is per 1M tokens)
        input_cost_value = total_input_tokens * self._input_cost_per_1m / 1_000_000
        output_cost_value = total_output_tokens * self._output_cost_per_1m / 1_000_000
        exact_cost_value = input_cost_value + output_cost_value
        
        # Format costs (total is calculated as rounded input + rounded output)
        input_cost, output_cost, total_cost = format_costs_together(
            input_cost_value, output_cost_value
        )
        
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


class EmbeddingsBatchJob:
    """Represents an asynchronous batch job for embeddings."""
    
    def __init__(
        self,
        batch_id: str,
        client: "OpenAI",
        model: str,
        input_cost_per_1m: float,
        num_requests: int
    ):
        """Initialize an embeddings batch job.
        
        Args:
            batch_id: The batch job ID.
            client: The OpenAI client instance.
            model: The model name.
            input_cost_per_1m: Input cost per 1M tokens.
            num_requests: Number of requests in the batch.
        """
        self.id = batch_id
        self._client = client
        self._model = model
        self._input_cost_per_1m = input_cost_per_1m
        self._num_requests = num_requests
        self._cached_status: Optional[BatchStatus] = None
    
    def poll(self) -> BatchStatus:
        """Poll the batch job status.
        
        Returns:
            BatchStatus: Current status of the batch job.
        """
        import time
        batch = self._client.batches.retrieve(self.id)
        finished = batch.status in ["completed", "failed", "cancelled"]
        self._cached_status = BatchStatus(finished=finished, status=batch.status)
        return self._cached_status
    
    def get(self) -> EmbeddingsResponse:
        """Get the results of the batch job.
        
        This method blocks until the batch is complete.
        
        Returns:
            EmbeddingsResponse: The response containing all embeddings.
        """
        import time
        
        # Wait for completion
        while True:
            batch = self._client.batches.retrieve(self.id)
            
            if batch.status == "completed":
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
            elif batch.status == "cancelled":
                return EmbeddingsResponse(
                    success=False,
                    data=[],
                    input_tokens=0,
                    throughput=0,
                    processing_time=0,
                    error="Batch processing was cancelled"
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
        processing_time = round(getattr(batch, 'processing_time', 0), 5)
        throughput = round(total_tokens / processing_time, 5) if processing_time > 0 else 0
        
        # Calculate costs (pricing is per 1M tokens)
        input_cost_value = total_tokens * self._input_cost_per_1m / 1_000_000
        output_cost_value = 0.0  # Embeddings don't have output tokens
        exact_cost_value = input_cost_value + output_cost_value
        
        # Format costs (total is calculated as rounded input + rounded output)
        input_cost, output_cost, total_cost = format_costs_together(
            input_cost_value, output_cost_value
        )
        
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


class ClassifyBatchJob:
    """Represents an asynchronous batch job for classification."""
    
    def __init__(
        self,
        batch_id: str,
        client: "OpenAI",
        model: str,
        input_cost_per_1m: float,
        num_requests: int
    ):
        """Initialize a classification batch job.
        
        Args:
            batch_id: The batch job ID.
            client: The OpenAI client instance.
            model: The model name.
            input_cost_per_1m: Input cost per 1M tokens.
            num_requests: Number of requests in the batch.
        """
        self.id = batch_id
        self._client = client
        self._model = model
        self._input_cost_per_1m = input_cost_per_1m
        self._num_requests = num_requests
        self._cached_status: Optional[BatchStatus] = None
    
    def poll(self) -> BatchStatus:
        """Poll the batch job status.
        
        Returns:
            BatchStatus: Current status of the batch job.
        """
        import time
        batch = self._client.batches.retrieve(self.id)
        finished = batch.status in ["completed", "failed", "cancelled"]
        self._cached_status = BatchStatus(finished=finished, status=batch.status)
        return self._cached_status
    
    def get(self) -> ClassifyResponse:
        """Get the results of the batch job.
        
        This method blocks until the batch is complete.
        
        Returns:
            ClassifyResponse: The response containing all classifications.
        """
        import time
        
        # Wait for completion
        while True:
            batch = self._client.batches.retrieve(self.id)
            
            if batch.status == "completed":
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
            elif batch.status == "cancelled":
                return ClassifyResponse(
                    success=False,
                    classifications=[],
                    input_tokens=0,
                    throughput=0,
                    processing_time=0,
                    error="Batch processing was cancelled"
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
            # Should be a single classification per result
            if classification_list:
                classifications_data.append(DictLike(classification_list[0]))
            total_tokens += result['response']['body']['usage']['prompt_tokens']
        
        # Calculate metrics
        processing_time = round(getattr(batch, 'processing_time', 0), 5)
        throughput = round(total_tokens / processing_time, 5) if processing_time > 0 else 0
        
        # Calculate costs (pricing is per 1M tokens)
        input_cost_value = total_tokens * self._input_cost_per_1m / 1_000_000
        output_cost_value = 0.0  # Classification doesn't have output tokens
        exact_cost_value = input_cost_value + output_cost_value
        
        # Format costs (total is calculated as rounded input + rounded output)
        input_cost, output_cost, total_cost = format_costs_together(
            input_cost_value, output_cost_value
        )
        
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
