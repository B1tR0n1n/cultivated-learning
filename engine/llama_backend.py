"""llama.cpp server backends for the cognitive shell.

LlamaServerEngine is a drop-in for InferenceEngine (engine/inference.py) that
generates through a running llama-server instead of in-process transformers.
It keeps the same interface and the same sampling semantics:

- do_sample with temperature, top_p, top_k=50 (the transformers default),
  repetition_penalty 1.1
- flat logit biases for single-token phrases
- sequence suppressions for multi-token phrases: the final token of a phrase
  is penalised only when the context already ends with the rest of it, exactly
  like LogitBiasProcessor. llama-server has no conditional bias, so this is
  done by backtracking: generate, find the first token that completes a
  suppressed phrase, cut there, resample that one position with the bias, and
  continue.

ChatWriter is a second backend for a different model family. It only writes
(reflection and observation text); it takes the same "[INST] ... [/INST]"
prompts the shell builds and sends them through the model's own chat template.

Every request carries a seed derived from a scope string and a call counter,
so two runs that share a scope get the same random draws for the same call.
"""
import json
import urllib.request
import zlib

import numpy as np

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def first_violation(context, new_tokens, sequences):
    """Index of the first token in new_tokens that completes a suppressed sequence.

    Args:
        context: token ids already fixed (prompt + accepted generation)
        new_tokens: freshly generated token ids
        sequences: list of token-id sequences, each of length >= 2

    Returns:
        index into new_tokens, or None if nothing in new_tokens completes a sequence.
    """
    if not sequences:
        return None
    longest = max(len(s) for s in sequences)
    tail = list(context[-(longest - 1):]) if longest > 1 else []
    for k, tok in enumerate(new_tokens):
        window = tail + list(new_tokens[:k])
        for seq in sequences:
            if len(seq) < 2 or tok != seq[-1]:
                continue
            prefix = list(seq[:-1])
            if window[-len(prefix):] == prefix:
                return k
    return None


def banned_at(context, sequences):
    """Final tokens of every suppressed sequence whose prefix the context ends with."""
    banned = set()
    for seq in sequences:
        if len(seq) < 2:
            continue
        prefix = list(seq[:-1])
        if list(context[-len(prefix):]) == prefix:
            banned.add(seq[-1])
    return banned


def strip_inst(prompt):
    """Turn a Mistral-format "[INST] ... [/INST]" prompt into plain message text."""
    return prompt.strip().removeprefix("[INST]").removesuffix("[/INST]").strip()


class _Seeder:
    """Deterministic per-call seeds: crc32(scope:counter)."""

    def __init__(self, scope="0"):
        self.set_scope(scope)

    def set_scope(self, scope):
        self.scope = str(scope)
        self.counter = 0

    def next(self):
        seed = zlib.crc32(f"{self.scope}:{self.counter}".encode()) & 0x7FFFFFFF
        self.counter += 1
        return seed


def _post(base_url, path, payload, timeout=600):
    request = urllib.request.Request(
        base_url.rstrip("/") + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


class LlamaServerEngine:
    """InferenceEngine-compatible wrapper around a llama-server endpoint."""

    def __init__(self, base_url="http://127.0.0.1:8091", max_context=16384,
                 embedding_model_path=DEFAULT_EMBEDDING_MODEL, prompt_format="mistral-v03",
                 post=None, embedder=None):
        self.base_url = base_url
        self.max_context = max_context
        self.embedding_model_path = embedding_model_path
        self._prompt_format = prompt_format
        self._post_fn = post or (lambda path, payload: _post(self.base_url, path, payload))
        self.embedding_model = embedder
        self._flat_biases = {}
        self._sequence_suppressions = []
        self.seeder = _Seeder()
        self.truncations = 0
        self.backtracks = 0
        self.calls = 0
        self._special = set()
        self._token_counts = {}

    @property
    def prompt_format(self):
        return self._prompt_format

    def load(self):
        if self.embedding_model is None:
            from sentence_transformers import SentenceTransformer
            self.embedding_model = SentenceTransformer(self.embedding_model_path, device="cpu")
        props = self._post_fn("/tokenize", {"content": "", "add_special": True})
        self._bos = props["tokens"][:1]
        # transformers decoded with skip_special_tokens=True; drop EOS the same way.
        self._special = set(self._tokenize("</s>")) if self._tokenize("</s>") else set()
        if len(self._special) != 1:
            self._special = set()
        print(f"LlamaServerEngine ready: {self.base_url} (max context {self.max_context})", flush=True)
        return self

    def set_seed_scope(self, scope):
        self.seeder.set_scope(scope)

    # --- tokens ---

    def _tokenize(self, text, add_special=False):
        return self._post_fn("/tokenize", {"content": text, "add_special": add_special})["tokens"]

    def count_tokens(self, text):
        # The assembler counts the same memory lines on every turn; cache them.
        if text not in self._token_counts:
            if len(self._token_counts) > 8192:
                self._token_counts.clear()
            self._token_counts[text] = len(self._tokenize(text))
        return self._token_counts[text]

    def _detokenize(self, tokens):
        if not tokens:
            return ""
        return self._post_fn("/detokenize", {"tokens": list(tokens)})["content"]

    # --- generation ---

    def _complete(self, tokens, n_predict, temperature, top_p, biases):
        self.calls += 1
        payload = {
            "prompt": list(tokens),
            "n_predict": n_predict,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": 50,
            "min_p": 0.0,
            "repeat_penalty": 1.1,
            "repeat_last_n": -1,
            "seed": self.seeder.next(),
            "return_tokens": True,
            "cache_prompt": True,
        }
        if biases:
            payload["logit_bias"] = [[int(t), float(b)] for t, b in biases.items()]
        result = self._post_fn("/completion", payload)
        tokens = [t for t in result.get("tokens", []) if t not in self._special]
        return tokens, result.get("stop_type") in ("eos", "word")

    def _prompt_tokens(self, prompt, max_new_tokens):
        tokens = self._tokenize(prompt, add_special=True)
        limit = self.max_context - max_new_tokens
        if len(tokens) > limit:
            # Keep BOS and the end of the prompt (the user turn and [/INST]).
            self.truncations += 1
            print(f"  WARNING: prompt of {len(tokens)} tokens truncated to {limit} (left side)", flush=True)
            tokens = tokens[:1] + tokens[-(limit - 1):]
        return tokens

    def generate(self, prompt, max_new_tokens=1024, temperature=0.7, top_p=0.9,
                 apply_biases=True):
        context = self._prompt_tokens(prompt, max_new_tokens)
        flat = dict(self._flat_biases) if apply_biases else {}
        sequences = self._sequence_suppressions if apply_biases else []
        generated = []
        while len(generated) < max_new_tokens:
            new, stopped = self._complete(context + generated, max_new_tokens - len(generated),
                                          temperature, top_p, flat)
            k = first_violation(context + generated, new, sequences)
            if k is None:
                generated += new
                break
            self.backtracks += 1
            generated += new[:k]
            biases = dict(flat)
            for token in banned_at(context + generated, sequences):
                biases[token] = biases.get(token, 0.0) - 10.0
            one, stopped = self._complete(context + generated, 1, temperature, top_p, biases)
            generated += one
            if stopped or not one:
                break
        return self._detokenize(generated).strip()

    def generate_structured(self, prompt, max_new_tokens=512, temperature=0.3):
        return self.generate(prompt, max_new_tokens=max_new_tokens, temperature=temperature, top_p=0.95)

    def generate_batch(self, prompts, max_new_tokens=512, temperature=0.3):
        # The transformers batch path never applied logit biases; neither does this.
        return [self.generate(p, max_new_tokens=max_new_tokens, temperature=temperature,
                              top_p=0.95, apply_biases=False) for p in prompts]

    # --- logit bias ---

    def set_logit_biases(self, bias_data):
        self._flat_biases, self._sequence_suppressions = bias_data

    def suppress_tokens(self, phrases):
        flat_biases = {}
        sequence_suppressions = []
        for phrase in phrases:
            phrase = phrase.strip("\n")
            if not phrase.strip():
                continue
            token_ids = self._tokenize(phrase)
            if not token_ids:
                continue
            if len(token_ids) == 1:
                flat_biases[token_ids[0]] = -10.0
            else:
                sequence_suppressions.append(token_ids)
        return flat_biases, sequence_suppressions

    # --- embeddings ---

    def get_embedding(self, text):
        return self.embedding_model.encode(text, normalize_embeddings=True)

    def get_embedding_dimension(self):
        return self.embedding_model.get_sentence_embedding_dimension()

    def cosine_similarity(self, vec_a, vec_b):
        a = np.asarray(vec_a)
        b = np.asarray(vec_b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))


class ChatWriter:
    """Writer for a different model family, through its own chat template.

    Implements only what the reflection engine calls: generate_structured and
    generate_batch. Thinking is switched off so the output is the answer itself.
    """

    def __init__(self, base_url="http://127.0.0.1:8092", post=None, label="observer"):
        self.base_url = base_url
        self.label = label
        self._post_fn = post or (lambda path, payload: _post(self.base_url, path, payload))
        self.seeder = _Seeder()
        self.calls = 0

    def set_seed_scope(self, scope):
        self.seeder.set_scope(scope)

    def _chat(self, prompt, max_new_tokens, temperature, top_p):
        self.calls += 1
        result = self._post_fn("/v1/chat/completions", {
            "messages": [{"role": "user", "content": strip_inst(prompt)}],
            "max_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "seed": self.seeder.next(),
            "chat_template_kwargs": {"enable_thinking": False},
        })
        return (result["choices"][0]["message"].get("content") or "").strip()

    def generate_structured(self, prompt, max_new_tokens=512, temperature=0.3):
        return self._chat(prompt, max_new_tokens, temperature, 0.95)

    def generate(self, prompt, max_new_tokens=1024, temperature=0.7, top_p=0.9):
        return self._chat(prompt, max_new_tokens, temperature, top_p)

    def generate_batch(self, prompts, max_new_tokens=512, temperature=0.3):
        return [self._chat(p, max_new_tokens, temperature, 0.95) for p in prompts]
