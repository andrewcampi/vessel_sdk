"""Main Vessel client."""

import time
from openai import OpenAI
from .resources import Models, Accounts, Embeddings, Chat, Classify


class RateLimiter:
    """Shared rate limiter for all API requests."""
    
    def __init__(self, seconds: float = 10.0):
        """Initialize the rate limiter.
        
        Args:
            seconds: Minimum seconds to wait between requests.
        """
        self._last_request_time = 0
        self._rate_limit_seconds = seconds
    
    def wait(self):
        """Wait if necessary to respect rate limits."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._rate_limit_seconds:
            wait_time = self._rate_limit_seconds - elapsed
            time.sleep(wait_time)
        self._last_request_time = time.time()


class Vessel:
    """Main client for interacting with the Vessel API.
    
    This client provides a simplified interface for using the Vessel API,
    automatically handling batch processing, rate limiting, and result parsing.
    
    Args:
        base_url: The base URL for the Vessel API.
        api_key: The API key for authentication.
    
    Example:
        >>> client = Vessel(
        ...     base_url="https://vessel.acampi.dev/v1",
        ...     api_key="your-api-key"
        ... )
        >>> response = client.embeddings.create(
        ...     model="vessel-embedding-nano",
        ...     input="Hello, world!"
        ... )
    """
    
    def __init__(self, base_url: str, api_key: str):
        """Initialize the Vessel client.
        
        Args:
            base_url: The base URL for the Vessel API.
            api_key: The API key for authentication.
        """
        self._openai_client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        self._base_url = base_url
        
        # Create shared rate limiter (11.25 seconds for safety buffer)
        self._rate_limiter = RateLimiter(seconds=11.25)
        
        # Initialize resources with shared rate limiter
        self.models = Models(self._openai_client)
        self.accounts = Accounts(self._openai_client, base_url)
        self.embeddings = Embeddings(self._openai_client, self._rate_limiter)
        self.chat = Chat(self._openai_client, self._rate_limiter)
        self.classify = Classify(self._openai_client, self._rate_limiter)

