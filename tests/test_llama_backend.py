from engine.llama_backend import (
    ChatWriter,
    LlamaServerEngine,
    _Seeder,
    banned_at,
    first_violation,
    strip_inst,
)
from tests.conftest import HashEmbedder


def test_first_violation_finds_token_that_completes_sequence():
    # sequence "Of course" = [7, 8]; context ends with 7, new tokens start with 8
    assert first_violation([1, 2, 7], [8, 9], [[7, 8]]) == 0
    assert first_violation([1, 2], [7, 8, 9], [[7, 8]]) == 1


def test_first_violation_ignores_partial_and_unrelated_tokens():
    assert first_violation([1, 2], [7, 9, 8], [[7, 8]]) is None
    assert first_violation([1, 2], [8, 8], [[7, 8]]) is None
    assert first_violation([1], [5], []) is None


def test_first_violation_with_longer_prefix_spanning_context():
    seq = [4, 5, 6, 7]
    assert first_violation([4, 5], [6, 7], [seq]) == 1
    assert first_violation([5], [6, 7], [seq]) is None


def test_banned_at_returns_final_tokens_whose_prefix_matches():
    assert banned_at([1, 7], [[7, 8], [7, 9], [3, 4]]) == {8, 9}
    assert banned_at([1, 2], [[7, 8]]) == set()


def test_strip_inst():
    assert strip_inst("[INST] hello there [/INST]") == "hello there"
    assert strip_inst("plain") == "plain"


def test_seeder_is_deterministic_per_scope():
    a, b = _Seeder("s1:4"), _Seeder("s1:4")
    assert [a.next() for _ in range(3)] == [b.next() for _ in range(3)]
    c = _Seeder("s2:4")
    assert c.next() != _Seeder("s1:4").next()


class FakeServer:
    """Scripted llama-server: each /completion pops the next token list."""

    def __init__(self, completions, vocab=None):
        self.completions = list(completions)
        self.requests = []
        self.vocab = vocab or {}

    def __call__(self, path, payload):
        self.requests.append((path, payload))
        if path == "/tokenize":
            text = payload["content"]
            toks = [self.vocab.get(w, 100 + len(w)) for w in text.split()] if text else []
            return {"tokens": ([1] if payload.get("add_special") else []) + toks}
        if path == "/detokenize":
            return {"content": " ".join(str(t) for t in payload["tokens"])}
        if path == "/completion":
            tokens, stop = self.completions.pop(0)
            return {"tokens": tokens, "stop_type": stop}
        raise AssertionError(path)


def make_engine(server):
    return LlamaServerEngine(post=server, embedder=HashEmbedder()).load()


def test_generate_without_suppression_is_one_call():
    server = FakeServer([([10, 11, 12], "eos")])
    engine = make_engine(server)
    assert engine.generate("hi") == "10 11 12"
    completions = [p for path, p in server.requests if path == "/completion"]
    assert len(completions) == 1
    assert completions[0]["prompt"][0] == 1  # BOS kept


def test_generate_backtracks_and_resamples_suppressed_sequence():
    # Suppress [7, 8]. First draw produces 7 8 9; engine must cut before 8,
    # resample one token with a -10 bias on 8, then continue.
    server = FakeServer([
        ([5, 7, 8, 9], "eos"),
        ([6], "limit"),
        ([20, 21], "eos"),
    ])
    engine = make_engine(server)
    engine.set_logit_biases(({}, [[7, 8]]))
    out = engine.generate("hi", max_new_tokens=50)
    assert out == "5 7 6 20 21"
    completions = [p for path, p in server.requests if path == "/completion"]
    resample = completions[1]
    assert resample["n_predict"] == 1
    assert [8, -10.0] in resample["logit_bias"]
    assert engine.backtracks == 1


def test_generate_batch_ignores_logit_biases():
    server = FakeServer([([5, 7, 8], "eos")])
    engine = make_engine(server)
    engine.set_logit_biases(({99: -10.0}, [[7, 8]]))
    assert engine.generate_batch(["x"]) == ["5 7 8"]
    completion = next(p for path, p in server.requests if path == "/completion")
    assert "logit_bias" not in completion


def test_flat_bias_is_sent_on_generate():
    server = FakeServer([([5], "eos")])
    engine = make_engine(server)
    engine.set_logit_biases(({99: -10.0}, []))
    engine.generate("x")
    completion = next(p for path, p in server.requests if path == "/completion")
    assert completion["logit_bias"] == [[99, -10.0]]


def test_long_prompt_is_truncated_from_the_left_keeping_bos():
    server = FakeServer([([5], "eos")])
    engine = LlamaServerEngine(post=server, embedder=HashEmbedder(), max_context=10).load()
    engine.generate("a b c d e f g h i j k l", max_new_tokens=4)
    prompt = next(p for path, p in server.requests if path == "/completion")["prompt"]
    assert len(prompt) == 6 and prompt[0] == 1
    assert engine.truncations == 1


def test_suppress_tokens_splits_single_and_multi_token_phrases():
    server = FakeServer([], vocab={"crucial": 42})
    engine = make_engine(server)
    flat, seqs = engine.suppress_tokens(["crucial", "it is important"])
    assert flat == {42: -10.0}
    assert len(seqs) == 1 and len(seqs[0]) == 3


def test_seed_scope_controls_request_seeds():
    server = FakeServer([([5], "eos"), ([5], "eos")])
    engine = make_engine(server)
    engine.set_seed_scope("run1:3")
    engine.generate("x")
    engine.set_seed_scope("run1:3")
    engine.generate("x")
    seeds = [p["seed"] for path, p in server.requests if path == "/completion"]
    assert seeds[0] == seeds[1]


def test_chat_writer_strips_inst_and_disables_thinking():
    calls = []

    def post(path, payload):
        calls.append((path, payload))
        return {"choices": [{"message": {"content": "  An observation. "}}]}

    writer = ChatWriter(post=post)
    assert writer.generate_structured("[INST] Watch this. [/INST]") == "An observation."
    assert writer.generate_batch(["[INST] a [/INST]", "[INST] b [/INST]"]) == ["An observation."] * 2
    path, payload = calls[0]
    assert path == "/v1/chat/completions"
    assert payload["messages"][0]["content"] == "Watch this."
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert writer.calls == 3


def test_eos_and_special_tokens_never_reach_the_text():
    # llama-server returns the EOS token in "tokens"; transformers decoded with
    # skip_special_tokens=True, so the text must not contain it.
    class Server(FakeServer):
        def __call__(self, path, payload):
            if path == "/tokenize" and payload["content"] == "</s>":
                return {"tokens": [2]}
            return super().__call__(path, payload)

    server = Server([([10, 11, 2], "eos")])
    engine = make_engine(server)
    assert engine.generate("hi") == "10 11"
