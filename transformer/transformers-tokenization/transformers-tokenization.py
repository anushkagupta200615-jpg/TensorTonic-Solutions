import re
import numpy as np
from typing import List, Dict

class SimpleTokenizer:
    """
    A word-level tokenizer with special tokens.
    """
    
    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 0
        
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
    
    def build_vocab(self, texts: List[str]) -> None:
        self.word_to_id = {
            self.pad_token: 0,
            self.unk_token: 1,
            self.bos_token: 2,
            self.eos_token: 3
        }
        self.id_to_word = {v: k for k, v in self.word_to_id.items()}
        
        words = set()
        for text in texts:
            text = text.lower()
            tokens = re.findall(r"\b\w+\b", text)
            words.update(tokens)
        
        for i, word in enumerate(sorted(words), start=4):
            self.word_to_id[word] = i
            self.id_to_word[i] = word
        
        self.vocab_size = len(self.word_to_id)
    
    def encode(self, text: str) -> List[int]:
        text = text.lower()
        tokens = re.findall(r"\b\w+\b", text)
        ids = []
        for token in tokens:
            ids.append(self.word_to_id.get(token, self.word_to_id[self.unk_token]))
        return ids
    
    def decode(self, ids: List[int]) -> str:
        words = []
        for i in ids:
            if i in self.id_to_word:
                tok = self.id_to_word[i]
                if tok not in {self.pad_token, self.bos_token, self.eos_token}:
                    words.append(tok)
            else:
                words.append(self.unk_token)
        return " ".join(words)
