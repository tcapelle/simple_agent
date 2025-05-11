from typing import List, Dict, Any
from pydantic import BaseModel, Field
class AgentState(BaseModel):
    """Manages the state of the agent."""
    messages: List[Dict[str, Any]] = Field(default_factory=list)
    step: int = Field(default=0)
    final_assistant_content: str | None = None # Populated at the end of a run