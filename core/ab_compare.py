import time
import json
import os


_VANILLA_SYSTEM_PROMPT = "You are a helpful assistant."


class ABCompare:
    """A/B comparison: full pipeline vs vanilla generation on identical inputs.

    For each prompt, generates two responses:
      A) Full pipeline — memory retrieval, directives, context assembly
      B) Vanilla — basic system prompt + user message, no memory or directives

    Stores results to a JSON file and returns a summary of token counts
    and generation times for both paths.
    """

    def __init__(self, engine, interaction_loop, output_path="data/ab_results.json"):
        self.engine = engine
        self.loop = interaction_loop
        self.output_path = output_path

    def run_comparison(self, prompts):
        """Run A/B comparison on a list of prompt strings.

        Args:
            prompts: List of user message strings to test.

        Returns:
            dict with per-prompt results and aggregate summary.
        """
        results = []

        for prompt_text in prompts:
            entry = self._compare_single(prompt_text)
            results.append(entry)

        # Aggregate summary
        pipeline_tokens = sum(r["pipeline"]["response_tokens"] for r in results)
        vanilla_tokens = sum(r["vanilla"]["response_tokens"] for r in results)
        pipeline_time = sum(r["pipeline"]["elapsed"] for r in results)
        vanilla_time = sum(r["vanilla"]["elapsed"] for r in results)

        summary = {
            "total_prompts": len(prompts),
            "pipeline_total_tokens": pipeline_tokens,
            "vanilla_total_tokens": vanilla_tokens,
            "pipeline_total_time": round(pipeline_time, 2),
            "vanilla_total_time": round(vanilla_time, 2),
            "pipeline_avg_time": round(pipeline_time / len(prompts), 2) if prompts else 0,
            "vanilla_avg_time": round(vanilla_time / len(prompts), 2) if prompts else 0,
        }

        output = {"summary": summary, "results": results}

        os.makedirs(os.path.dirname(self.output_path) or ".", exist_ok=True)
        with open(self.output_path, "w") as f:
            json.dump(output, f, indent=2)

        print(f"A/B comparison complete: {len(prompts)} prompts")
        print(f"  Pipeline: {pipeline_tokens} tokens, {pipeline_time:.2f}s total")
        print(f"  Vanilla:  {vanilla_tokens} tokens, {vanilla_time:.2f}s total")
        print(f"  Results saved to {self.output_path}")

        return output

    def _compare_single(self, prompt_text):
        """Generate both pipeline and vanilla responses for a single prompt."""

        # A) Full pipeline
        t0 = time.time()
        pipeline_response = self.loop.chat(prompt_text)
        pipeline_elapsed = time.time() - t0
        pipeline_tokens = self.engine.count_tokens(pipeline_response)

        # B) Vanilla (no memory, no directives)
        vanilla_prompt = f"[INST] {_VANILLA_SYSTEM_PROMPT}\nUser: {prompt_text} [/INST]"
        t0 = time.time()
        vanilla_response = self.engine.generate(vanilla_prompt)
        vanilla_elapsed = time.time() - t0
        vanilla_tokens = self.engine.count_tokens(vanilla_response)

        return {
            "prompt": prompt_text,
            "timestamp": time.time(),
            "pipeline": {
                "response": pipeline_response,
                "response_tokens": pipeline_tokens,
                "elapsed": round(pipeline_elapsed, 2),
            },
            "vanilla": {
                "response": vanilla_response,
                "response_tokens": vanilla_tokens,
                "elapsed": round(vanilla_elapsed, 2),
            },
        }
