from typing import List, Dict
from datetime import datetime


class ConversationMemory:
    """Manages conversation history and context"""
    
    def __init__(self, max_history: int = 10):
        self.history: List[Dict] = []
        self.max_history = max_history
    
    def add_exchange(self, user_query: str, assistant_response: str):
        """Store user query and bot response"""
        self.history.append({
            "timestamp": datetime.now().isoformat(),
            "user": user_query,
            "assistant": assistant_response
        })
        # Keep only recent history
        if len(self.history) > self.max_history:
            self.history.pop(0)
    
    def get_context(self) -> str:
        """Get conversation context for prompt"""
        if not self.history:
            return "This is the start of the conversation."
        
        context = "Previous conversation:\n"
        for exchange in self.history[-5:]:  # Last 5 exchanges
            context += f"User: {exchange['user']}\nAssistant: {exchange['assistant']}\n\n"
        return context
    
    def clear(self):
        """Clear conversation history"""
        self.history = []
    
    def get_history(self) -> List[Dict]:
        """Get full conversation history"""
        return self.history.copy()
    
    def get_recent_context(self, num_exchanges: int = 3) -> str:
        """Get recent exchanges as context"""
        if not self.history:
            return ""
        
        recent = self.history[-num_exchanges:]
        context_lines = []
        for exchange in recent:
            context_lines.append(f"User: {exchange['user']}")
            context_lines.append(f"Assistant: {exchange['assistant']}")
        
        return "\n".join(context_lines)
