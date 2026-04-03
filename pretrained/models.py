# from transformers import pipeline

# pipe = pipeline("text-generation", model="codeparrot/codeparrot-small")

# # Run the environment locally from the space
# # docker run -d -p 8000:8000 registry.hf.space/openenv-echo-env:latest

# classifier = pipeline('sentiment-analysis')
# print(classifier("I love using Transformers!"))

# models.py
from pydantic import Field
from openenv.core.env_server.types import Action, Observation

class MyAction(Action):
    """Your custom action."""
    command: str = Field(..., description="Command to execute")
    parameters: dict = Field(default_factory=dict, description="Command parameters")

class MyObservation(Observation):
    """Your custom observation."""
    result: str = Field(..., description="Result of the action")
    success: bool = Field(..., description="Whether the action succeeded")