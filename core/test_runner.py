import time
import json
import os


_VANILLA_SYSTEM_PROMPT = "You are a helpful assistant."

# Prompt numbers that trigger decay passes
_DECAY_CHECKPOINTS = {25, 50, 75, 100}
# Prompt numbers that trigger consolidation passes
_CONSOLIDATION_CHECKPOINTS = {50, 100}


class TestRunner:
    """Automated test runner for the 100-prompt longitudinal evaluation.

    Drives an InteractionLoop through a sequence of prompts, collects
    interactive ratings, runs checkpoint automation (decay, consolidation,
    directive snapshots), tracks cumulative stats, and writes full results
    to a JSON file.
    """

    def __init__(self, loop, memory, consolidator,
                 output_path="data/test_results.json",
                 ab_output_path="data/ab_comparison.json"):
        self.loop = loop
        self.memory = memory
        self.consolidator = consolidator
        self.output_path = output_path
        self.ab_output_path = ab_output_path
        self.prompts = []
        self._results = []  # stored for A/B comparison lookups

    def load_prompts(self, path):
        """Read a JSON prompt file produced from the 100-prompt protocol."""
        with open(path, "r") as f:
            self.prompts = json.load(f)
        print(f"Loaded {len(self.prompts)} test prompts from {path}")

    # ------------------------------------------------------------------
    # Main run
    # ------------------------------------------------------------------

    def run(self, pause_for_rating=True):
        """Run the full prompt sequence.

        Args:
            pause_for_rating: If True, pause after each response and prompt
                the user for a rating (1-5), skip ('s'), or quit ('q').
                If False, run all prompts unattended with no rating input.
        """
        if not self.prompts:
            print("No prompts loaded. Call load_prompts() first.")
            return

        results = []
        stats = _RunStats()
        quit_early = False

        for entry in self.prompts:
            prompt_num = entry.get("number", 0)
            prompt_text = entry.get("prompt", "")
            category = entry.get("category", "")
            expected = entry.get("expected_behavior")

            # Run interaction
            step_start = time.time()
            response = self.loop.chat(prompt_text)
            step_elapsed = time.time() - step_start

            verify_status = self.loop._last_verify_status
            memory_count = self.memory.collection.count()

            if verify_status == "UNVERIFIED":
                stats.unverified += 1

            # Show expected behavior hint
            if pause_for_rating and expected:
                print(f"  Expected: {expected}")

            # Collect rating
            rating = None
            correction = None

            if pause_for_rating:
                rating, correction, should_quit = _prompt_for_rating()
                if should_quit:
                    quit_early = True
            # else: unattended — no rating input

            # Process feedback if rated
            directives_killed = 0
            if rating is not None:
                directive_count_before = 0
                if self.loop.reflection_engine:
                    directive_count_before, _ = self.loop.reflection_engine.get_directive_count()

                self.loop.feedback(rating, correction=correction)

                directive_count_after = 0
                if self.loop.reflection_engine:
                    directive_count_after, _ = self.loop.reflection_engine.get_directive_count()
                directives_killed = max(0, directive_count_before - directive_count_after)

                stats.record_rating(rating, category)
                if correction:
                    stats.corrections += 1
                stats.directives_killed += directives_killed

            stats.completed += 1

            # Store result
            results.append({
                "number": prompt_num,
                "prompt": prompt_text,
                "category": category,
                "expected_behavior": expected,
                "response": response,
                "rating": rating,
                "correction": correction,
                "verify_status": verify_status,
                "memory_count": memory_count,
                "elapsed": round(step_elapsed, 2),
                "timestamp": time.time(),
            })

            # --- Checkpoint automation ---

            # Decay pass at 25, 50, 75, 100
            if prompt_num in _DECAY_CHECKPOINTS:
                print(f"\n--- Checkpoint: decay pass (after prompt {prompt_num}) ---")
                self.memory.decay_pass()

            # Consolidation at 50, 100
            if prompt_num in _CONSOLIDATION_CHECKPOINTS:
                print(f"\n--- Checkpoint: consolidation (after prompt {prompt_num}) ---")
                new_semantic = self.consolidator.consolidate()
                print(f"  Consolidation produced {len(new_semantic)} new semantic memories")

            # Directive snapshot every 10 prompts
            if prompt_num > 0 and prompt_num % 10 == 0:
                self._print_directive_snapshot(prompt_num)

            # Progress summary every 10 interactions
            if stats.completed % 10 == 0:
                stats.print_progress(len(self.prompts))

            if quit_early:
                print(f"\nRun stopped early at prompt {stats.completed}/{len(self.prompts)}")
                break

        # --- Final summary ---
        memory_stats = self.memory.get_stats()
        directive_list = []
        if self.loop.reflection_engine:
            directive_units = self.loop.reflection_engine.get_directives()
            directive_list = [d.content for d in directive_units]

        summary = stats.summary(len(self.prompts))
        summary["memory_stats"] = memory_stats
        summary["directive_list"] = directive_list

        _print_final_summary(summary, memory_stats, directive_list)

        # Write results
        output = {"summary": summary, "results": results}
        os.makedirs(os.path.dirname(self.output_path) or ".", exist_ok=True)
        with open(self.output_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to {self.output_path}")

        # Cache results for A/B lookups
        self._results = results

        return output

    # ------------------------------------------------------------------
    # A/B comparison
    # ------------------------------------------------------------------

    def run_ab_comparison(self, prompt_numbers):
        """Re-run selected prompts through vanilla generation for A/B comparison.

        Uses engine.generate() with a basic system prompt and the user message
        only — no memory, no directives, no logit bias.

        Args:
            prompt_numbers: List of prompt numbers to compare, e.g. [1, 5, 10, 25, 50, 100].
        """
        # Build lookup from main run results
        result_by_num = {r["number"]: r for r in self._results}

        # Build prompt lookup from loaded prompts
        prompt_by_num = {p["number"]: p for p in self.prompts}

        comparisons = []

        for num in prompt_numbers:
            entry = prompt_by_num.get(num)
            if not entry:
                print(f"  Prompt {num} not found in loaded prompts — skipping")
                continue

            prompt_text = entry["prompt"]
            framework_result = result_by_num.get(num)

            # Vanilla generation — no memory, no directives
            vanilla_prompt = f"[INST] {_VANILLA_SYSTEM_PROMPT}\nUser: {prompt_text} [/INST]"
            t0 = time.time()
            vanilla_response = self.loop.engine.generate(vanilla_prompt)
            vanilla_elapsed = time.time() - t0

            comparisons.append({
                "number": num,
                "prompt": prompt_text,
                "category": entry.get("category", ""),
                "framework_response": framework_result["response"] if framework_result else None,
                "framework_rating": framework_result["rating"] if framework_result else None,
                "vanilla_response": vanilla_response,
                "vanilla_elapsed": round(vanilla_elapsed, 2),
                "timestamp": time.time(),
            })

            print(f"  A/B #{num}: framework={'rated ' + str(framework_result['rating']) if framework_result and framework_result.get('rating') else 'unrated'}"
                  f" | vanilla={len(vanilla_response)} chars, {vanilla_elapsed:.2f}s")

        output = {
            "prompt_numbers": prompt_numbers,
            "total": len(comparisons),
            "comparisons": comparisons,
        }

        os.makedirs(os.path.dirname(self.ab_output_path) or ".", exist_ok=True)
        with open(self.ab_output_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nA/B comparison saved to {self.ab_output_path}")

        return output

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _print_directive_snapshot(self, prompt_num):
        """Print the current directive set."""
        if not self.loop.reflection_engine:
            return
        directives = self.loop.reflection_engine.get_directives()
        count, cap = self.loop.reflection_engine.get_directive_count()
        print(f"\n--- Directive snapshot (after prompt {prompt_num}): {count}/{cap} ---")
        if directives:
            for i, d in enumerate(directives):
                print(f"  {i+1}. [{d.origin}] {d.content}")
        else:
            print("  (none)")
        print()


class _RunStats:
    """Accumulates stats during a test run."""

    def __init__(self):
        self.completed = 0
        self.ratings = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        self.rated_count = 0
        self.rating_sum = 0
        self.category_ratings = {}  # category -> [list of ratings]
        self.corrections = 0
        self.directives_killed = 0
        self.unverified = 0

    def record_rating(self, rating, category=""):
        self.ratings[rating] = self.ratings.get(rating, 0) + 1
        self.rated_count += 1
        self.rating_sum += rating
        if category:
            self.category_ratings.setdefault(category, []).append(rating)

    @property
    def avg_rating(self):
        return self.rating_sum / self.rated_count if self.rated_count else 0.0

    def print_progress(self, total):
        excellent = self.ratings.get(4, 0) + self.ratings.get(5, 0)
        fail = self.ratings.get(1, 0) + self.ratings.get(2, 0)
        exc_pct = (excellent / self.rated_count * 100) if self.rated_count else 0
        fail_pct = (fail / self.rated_count * 100) if self.rated_count else 0

        print(f"\n\u2500\u2500 Progress: {self.completed}/{total} " + "\u2500" * 29)
        print(f"  Avg rating:  {self.avg_rating:.2f} | Excellent(4-5): {exc_pct:.0f}% | Fail(1-2): {fail_pct:.0f}%")
        print(f"  Corrections: {self.corrections} | Directives killed: {self.directives_killed}")
        print(f"  Unverified:  {self.unverified} episodes flagged")
        print("\u2500" * 48 + "\n")

    def summary(self, total):
        excellent = self.ratings.get(4, 0) + self.ratings.get(5, 0)
        fail = self.ratings.get(1, 0) + self.ratings.get(2, 0)

        category_avgs = {}
        for cat, vals in self.category_ratings.items():
            category_avgs[cat] = {
                "count": len(vals),
                "avg": round(sum(vals) / len(vals), 2) if vals else 0,
            }

        rating_pcts = {}
        for r in range(1, 6):
            count = self.ratings.get(r, 0)
            rating_pcts[str(r)] = {
                "count": count,
                "pct": round(count / self.rated_count * 100, 1) if self.rated_count else 0,
            }

        return {
            "total_prompts": total,
            "completed": self.completed,
            "rated": self.rated_count,
            "avg_rating": round(self.avg_rating, 2),
            "excellent_pct": round(excellent / self.rated_count * 100, 1) if self.rated_count else 0,
            "fail_pct": round(fail / self.rated_count * 100, 1) if self.rated_count else 0,
            "corrections": self.corrections,
            "directives_killed": self.directives_killed,
            "unverified_episodes": self.unverified,
            "rating_distribution": rating_pcts,
            "category_breakdown": category_avgs,
        }


def _prompt_for_rating():
    """Interactive rating prompt. Returns (rating_or_None, correction_or_None, should_quit)."""
    while True:
        raw = input("  Rating (1-5), 's' to skip, 'q' to quit: ").strip().lower()
        if raw == "s":
            return None, None, False
        if raw == "q":
            return None, None, True
        if raw in ("1", "2", "3", "4", "5"):
            rating = int(raw)
            correction_raw = input("  Correction (enter to skip): ").strip()
            correction = correction_raw if correction_raw else None
            return rating, correction, False
        print("  Invalid input. Enter 1-5, 's', or 'q'.")


def _print_final_summary(summary, memory_stats, directive_list):
    print(f"\n{'=' * 60}")
    print(f"  FINAL SUMMARY — 100-Prompt Longitudinal Evaluation")
    print(f"{'=' * 60}")
    print(f"  Completed:   {summary['completed']}/{summary['total_prompts']} prompts")
    print(f"  Rated:       {summary['rated']}")
    print(f"  Avg rating:  {summary['avg_rating']}")
    print(f"  Excellent:   {summary['excellent_pct']}%  |  Fail: {summary['fail_pct']}%")
    print(f"  Corrections: {summary['corrections']}")
    print(f"  Dir. killed: {summary['directives_killed']}")
    print(f"  Unverified:  {summary['unverified_episodes']} episodes")

    # Memory stats
    print(f"\n  --- Memory ---")
    print(f"  Total memories: {memory_stats.get('total', 0)}")
    by_type = memory_stats.get("by_type", {})
    for t, count in sorted(by_type.items()):
        print(f"    {t:12s}: {count}")
    print(f"  Avg salience:   {memory_stats.get('avg_salience', 0):.4f}")
    print(f"  Superseded:     {memory_stats.get('superseded', 0)}")

    # Directives
    print(f"\n  --- Active Directives ({len(directive_list)}) ---")
    if directive_list:
        for i, d in enumerate(directive_list):
            print(f"    {i+1}. {d}")
    else:
        print("    (none)")

    # Rating distribution
    dist = summary["rating_distribution"]
    print(f"\n  --- Rating Distribution ---")
    for r in ("1", "2", "3", "4", "5"):
        d = dist[r]
        bar = "\u2588" * int(d["pct"] / 2)
        print(f"    {r}: {bar} {d['pct']}% ({d['count']})")

    # Per-category
    cats = summary["category_breakdown"]
    if cats:
        print(f"\n  --- Per-Category Averages ---")
        for cat, data in sorted(cats.items()):
            print(f"    {cat:30s}  avg={data['avg']:.2f}  n={data['count']}")

    print(f"{'=' * 60}")
