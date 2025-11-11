"""Accounts resource for the Vessel SDK."""

from typing import TYPE_CHECKING
from ..types import AccountRetrieveResponse, Account

if TYPE_CHECKING:
    from openai import OpenAI


class Accounts:
    """Resource for interacting with account information."""
    
    def __init__(self, client: "OpenAI", base_url: str):
        self._client = client
        self._base_url = base_url
    
    def retrieve(self) -> AccountRetrieveResponse:
        """Retrieve account information.
        
        Returns:
            AccountRetrieveResponse: Response containing account details.
        """
        import requests
        
        # Make a direct request to the account endpoint
        # Since OpenAI client doesn't have an accounts endpoint, we need to make a custom request
        response = requests.get(
            f"{self._base_url}/account",
            headers={
                "Authorization": f"Bearer {self._client.api_key}"
            }
        )
        
        if response.status_code == 200:
            data = response.json()
            account = Account(
                email=data.get('email', ''),
                credits=data.get('credits', 0.0)
            )
            return AccountRetrieveResponse(data=account)
        else:
            # If request fails, try to get error details
            try:
                error_data = response.json()
                print(f"Warning: Failed to retrieve account: {error_data}")
            except:
                print(f"Warning: Failed to retrieve account (status: {response.status_code})")
            
            # Return default values if the request fails
            account = Account(email='', credits=0.0)
            return AccountRetrieveResponse(data=account)

