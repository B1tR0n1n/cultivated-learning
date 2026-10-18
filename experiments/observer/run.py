"""One run of the observer experiment: one arm, one seed, the 100-prompt protocol.

    python -m experiments.observer.run --arm PEER --seed 3 --out runs/PEER-s3

Arms (see PREREG.md):
    SELF       shipped design: the agent reflects on itself
    SELF_PLUS  first-person reflection given the observer's evidence too
    PEER       written from outside by the agent's own weights
    OTHER      written from outside by a different model family
"""
import argparse
import json
import os
import subprocess
import sys
import time

from core.bias_builder import BiasBuilder
from core.consolidation import ConsolidationEngine
from core.context_assembler import ContextAssembler
from core.corrections import apply_correction
from core.interaction_loop import InteractionLoop
from core.memory_store import MemoryStore
from engine.llama_backend import ChatWriter, LlamaServerEngine
from experiments.observer import metrics

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.dirname(os.path.abspath(__file__))

ARMS = {
    "SELF": {"perspective": "self", "writer": "subject"},
    "SELF_PLUS": {"perspective": "self_informed", "writer": "subject"},
    "PEER": {"perspective": "observer", "writer": "subject"},
    "OTHER": {"perspective": "observer", "writer": "observer"},
}


class RecordingWriter:
    """Passes writer calls through and keeps every prompt and output."""

    def __init__(self, inner):
        self.inner = inner
        self.log = []
        self.interaction = 0

    def set_seed_scope(self, scope):
        self.inner.set_seed_scope(scope)

    def _record(self, kind, prompt, output):
        self.log.append({"interaction": self.interaction, "kind": kind,
                         "prompt": prompt, "output": output})

    def generate_structured(self, prompt, max_new_tokens=512, temperature=0.3):
        out = self.inner.generate_structured(prompt, max_new_tokens=max_new_tokens, temperature=temperature)
        self._record("structured", prompt, out)
        return out

    def generate_batch(self, prompts, max_new_tokens=512, temperature=0.3):
        outs = self.inner.generate_batch(prompts, max_new_tokens=max_new_tokens, temperature=temperature)
        for p, o in zip(prompts, outs):
            self._record("batch", p, o)
        return outs


def build(arm, subject_url, observer_url, out_dir, system_prompt_path, engine=None, writer=None):
    """Assemble the shell for one arm. engine/writer can be injected for tests."""
    spec = ARMS[arm]
    engine = engine or LlamaServerEngine(subject_url).load()
    if writer is None:
        writer = engine if spec["writer"] == "subject" else ChatWriter(observer_url)
    recorder = RecordingWriter(writer)
    memory = MemoryStore(persist_dir=os.path.join(out_dir, "memory_db"), engine=engine)
    assembler = ContextAssembler(engine=engine, memory_store=memory,
                                 system_prompt_path=system_prompt_path)
    loop = InteractionLoop(
        engine=engine, memory_store=memory, assembler=assembler,
        log_dir=os.path.join(out_dir, "interaction_log"),
        bias_builder=BiasBuilder(engine, memory),
        reflection_kwargs={"writer": recorder, "perspective": spec["perspective"]},
    )
    return engine, writer, recorder, memory, loop, ConsolidationEngine(engine, memory)


def git_commit():
    try:
        return subprocess.check_output(["git", "-C", ROOT, "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def dump_memories(memory):
    data = memory.collection.get(include=["documents", "metadatas"])
    return [{"id": i, "content": d, **m} for i, d, m in zip(data["ids"], data["documents"], data["metadatas"])]


def run(arm, seed, out_dir, prompts, schedule, subject_url, observer_url,
        system_prompt_path, limit=None, engine=None, writer=None):
    os.makedirs(out_dir, exist_ok=True)
    progress_path = os.path.join(out_dir, "progress.jsonl")
    if os.path.exists(progress_path):
        raise SystemExit(f"{out_dir} already has a partial run; move it aside to rerun")

    engine, writer, recorder, memory, loop, consolidator = build(
        arm, subject_url, observer_url, out_dir, system_prompt_path, engine, writer)
    by_after = {}
    for c in schedule["corrections"]:
        by_after.setdefault(c["after"], []).append(c)

    started = time.time()
    records = []
    for entry in prompts[:limit]:
        n = entry["number"]
        engine.set_seed_scope(f"{seed}:{n}")
        if writer is not engine:
            writer.set_seed_scope(f"{seed}:{n}:observer")
        recorder.interaction = n

        t0 = time.time()
        response = loop.chat(entry["prompt"])
        record = {
            "number": n,
            "prompt": entry["prompt"],
            "response": response,
            "elapsed": round(time.time() - t0, 2),
            "directives": [{"content": d.content, "origin": d.origin}
                           for d in loop.reflection_engine.get_directives()],
            "memory_count": memory.collection.count(),
            "corrections": [],
        }
        for c in by_after.get(n, []):
            record["corrections"].append({"type": c["type"], "content": c["content"],
                                          "result": apply_correction(loop, c["type"], c["content"])})
        if n in schedule["decay_after"]:
            memory.decay_pass()
        if n in schedule["consolidate_after"]:
            consolidator.consolidate()
        records.append(record)
        with open(progress_path, "a") as f:
            f.write(json.dumps(record) + "\n")
        print(f"[{arm} s{seed}] {n:3d}  {record['elapsed']:6.1f}s  mem={record['memory_count']}", flush=True)

    responses = {r["number"]: r["response"] for r in records}
    summary = {
        "arm": arm,
        "seed": seed,
        "perspective": ARMS[arm]["perspective"],
        "writer": ARMS[arm]["writer"],
        "completed": len(records),
        "endpoints": metrics.endpoint_rates(responses, schedule["endpoints"]) if limit is None else None,
        "crucial_after_40": metrics.phrase_count(responses, "crucial", after=40),
        "important_to_note_after_42": metrics.phrase_count(responses, "it is important to note", after=42),
        "memory_stats": memory.get_stats(),
        "final_directives": records[-1]["directives"] if records else [],
        "engine": {"calls": getattr(engine, "calls", None), "backtracks": getattr(engine, "backtracks", None),
                   "truncations": getattr(engine, "truncations", None)},
        "writer_calls": len(recorder.log),
        "git_commit": git_commit(),
        "started": started,
        "seconds": round(time.time() - started, 1),
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(out_dir, "writer_log.json"), "w") as f:
        json.dump(recorder.log, f, indent=1)
    with open(os.path.join(out_dir, "memories.json"), "w") as f:
        json.dump(dump_memories(memory), f, indent=1)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm", required=True, choices=sorted(ARMS))
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--out", required=True)
    parser.add_argument("--prompts", default=os.path.join(ROOT, "data", "test_prompts_100.json"))
    parser.add_argument("--schedule", default=os.path.join(HERE, "schedule.json"))
    parser.add_argument("--system-prompt", default=os.path.join(ROOT, "sable_system_prompt.txt"))
    parser.add_argument("--subject-url", default="http://127.0.0.1:8091")
    parser.add_argument("--observer-url", default="http://127.0.0.1:8092")
    parser.add_argument("--limit", type=int, default=None, help="first N prompts only (smoke test)")
    args = parser.parse_args(argv)

    if not os.path.exists(args.system_prompt):
        sys.exit(f"system prompt not found: {args.system_prompt}")
    with open(args.prompts) as f:
        prompts = json.load(f)
    with open(args.schedule) as f:
        schedule = json.load(f)
    summary = run(args.arm, args.seed, args.out, prompts, schedule, args.subject_url,
                  args.observer_url, args.system_prompt, limit=args.limit)
    print(json.dumps({k: summary[k] for k in ("arm", "seed", "completed", "endpoints", "seconds")}, indent=2))


if __name__ == "__main__":
    main()
