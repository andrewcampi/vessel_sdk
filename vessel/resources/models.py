"""Models resource for the Vessel SDK."""

from typing import TYPE_CHECKING
from ..types import ModelsListResponse, Model

if TYPE_CHECKING:
    from openai import OpenAI


class Models:
    """Resource for interacting with models."""
    
    def __init__(self, client: "OpenAI"):
        self._client = client
    
    def list(self) -> ModelsListResponse:
        """List all available models.
        
        Returns:
            ModelsListResponse: Response containing list of models with their details.
        """
        response = self._client.models.list()
        
        models = []
        for model in response.data:
            # Extract cost information from the model object's pricing dict
            pricing = getattr(model, 'pricing', {})
            input_cost = pricing.get('input', 0.0) if pricing else 0.0
            output_cost = pricing.get('output', 0.0) if pricing else 0.0
            
            models.append(Model(
                name=model.id,
                type=getattr(model, 'type', 'unknown'),
                input_cost=input_cost,
                output_cost=output_cost
            ))
        
        return ModelsListResponse(data=models)

