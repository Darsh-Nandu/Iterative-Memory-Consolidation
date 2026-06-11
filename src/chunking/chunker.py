from typing import List, Dict, Any
from transformers import AutoTokenizer

class StructuralChunker:
    """
    Processes broad continuous text spaces into atomic tokens 
    and handles deterministic structural layouts.
    """
    def __init__(self, strategy: str = "recursive", size: int = 1024, overlap: int = 128):
        self.strategy = strategy
        self.size = size
        self.overlap = overlap
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")

    def chunk_text(self, text: str) -> List[Dict[str, Any]]:
        tokens = self.tokenizer.encode(text)
        chunks = []
        start = 0
        chunk_idx = 0

        while start < len(tokens):
            end = min(start + self.size, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = self.tokenizer.decode(chunk_tokens)
            
            chunks.append({
                "index": chunk_idx,
                "text": chunk_text,
                "token_count": len(chunk_tokens),
                "start_token": start,
                "end_token": end
            })
            
            if end == len(tokens):
                break
            start += self.size - self.overlap
            chunk_idx += 1

        return chunks