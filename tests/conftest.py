import os
import sys
import zlib

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


class HashEmbedder:
    """Deterministic bag-of-words embedder: shared words give high cosine similarity."""

    dim = 64

    def encode(self, text, normalize_embeddings=True):
        vec = np.zeros(self.dim)
        for word in str(text).lower().replace(".", " ").replace(",", " ").split():
            vec[zlib.crc32(word.encode()) % self.dim] += 1.0
        norm = np.linalg.norm(vec)
        return vec / norm if norm else vec

    def get_sentence_embedding_dimension(self):
        return self.dim


class FakeEngine:
    """Stands in for the subject model. Records every prompt it is given."""

    prompt_format = "mistral-v03"
    max_context = 16384

    def __init__(self, reply="A plain answer.", structured_reply="NO_NEW_DIRECTIVE"):
        self.embedder = HashEmbedder()
        self.reply = reply
        self.structured_reply = structured_reply
        self.prompts = []
        self.batch_prompts = []
        self.biases = ({}, [])
        self.scope = None

    def set_seed_scope(self, scope):
        self.scope = scope

    def count_tokens(self, text):
        return len(str(text).split())

    def generate(self, prompt, max_new_tokens=1024, temperature=0.7, top_p=0.9):
        self.prompts.append(prompt)
        return self.reply(prompt) if callable(self.reply) else self.reply

    def generate_structured(self, prompt, max_new_tokens=512, temperature=0.3):
        self.prompts.append(prompt)
        if callable(self.structured_reply):
            return self.structured_reply(prompt)
        return self.structured_reply

    def generate_batch(self, prompts, max_new_tokens=512, temperature=0.3):
        self.batch_prompts.extend(prompts)
        return ["The response was accurate and brief." for _ in prompts]

    def get_embedding(self, text):
        return self.embedder.encode(text)

    def cosine_similarity(self, a, b):
        return float(np.dot(a, b))

    def set_logit_biases(self, bias_data):
        self.biases = bias_data

    def suppress_tokens(self, phrases):
        return ({}, [list(p) for p in phrases])


@pytest.fixture
def fake_engine():
    return FakeEngine()


@pytest.fixture
def memory_store(tmp_path, fake_engine):
    from core.memory_store import MemoryStore
    return MemoryStore(persist_dir=str(tmp_path / "memory_db"), engine=fake_engine)
