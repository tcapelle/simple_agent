from typing import List, Dict, Any

class AgentState:
    """Represents the state of the agent's conversation."""
    
    def __init__(self, history: List[Dict[str, Any]] = None):
        """
        Initialize agent state.
        
        Args:
            history: List of conversation messages
        """
        self.history = history or []

    def __str__(self):
        return f"AgentState(history_length={len(self.history)})" 
