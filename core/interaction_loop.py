import time
import json
import uuid
from core.memory_store import MemoryUnit, MemoryType
from core.reflection import ReflectionEngine
from evaluation.metrics import EvaluationMetrics

# Phrases that indicate the user is asking for fabricated/hypothetical content.
# Episodic memories created during these interactions are tagged "speculative"
# and stored at low salience so they don't pollute factual retrieval.
_FABRICATION_TRIGGERS = {
    "make up", "made up", "imagine", "invent", "invented",
    "pretend", "create a fictional", "hypothetical",
    "make something up", "come up with a fictional", "fictional scenario",
}


class InteractionLoop:
    """Main loop: assembles context, generates response, stores memories, reflects, measures.
    
    v5 changes:
    - feedback() accepts optional target_query to aim salience adjustments
      at a specific topic instead of only the last interaction
    """

    def __init__(self, engine, memory_store, assembler, log_dir=None,
                 metrics_dir=None, reflect=True, bias_builder=None):
        self.engine = engine
        self.memory = memory_store
        self.assembler = assembler
        self.log_dir = log_dir
        self.history = []
        self.interaction_count = 0
        self.reflect_enabled = reflect
        self.reflection_engine = None
        self.metrics = None
        self.bias_builder = bias_builder
        self.last_interaction_id = None
        self.last_response = ""

        if self.reflect_enabled:
            self.reflection_engine = ReflectionEngine(engine, memory_store)

        if metrics_dir:
            self.metrics = EvaluationMetrics(metrics_dir, memory_store)

    def chat(self, user_message):
        self.interaction_count += 1
        interaction_id = str(uuid.uuid4())
        start_time = time.time()

        directives = None
        if self.reflection_engine:
            directives = self.reflection_engine.get_directives()

        prompt = self.assembler.assemble(
            user_message=user_message,
            conversation_history=self.history,
            directives=directives,
        )

        prompt_tokens = self.engine.count_tokens(prompt)

        if self.bias_builder:
            self.engine.set_logit_biases(self.bias_builder.build_bias_map())

        response = self.engine.generate(prompt)
        self.last_interaction_id = interaction_id
        self.last_response = response
        elapsed = time.time() - start_time
        response_tokens = self.engine.count_tokens(response)

        self.history.append({"role": "user", "content": user_message})
        self.history.append({"role": "assistant", "content": response})

        if len(self.history) > 20:
            self.history = self.history[-20:]

        user_lower = user_message.lower()
        is_speculative = any(trigger in user_lower for trigger in _FABRICATION_TRIGGERS)
        episodic = MemoryUnit(
            content=f"User: {user_message}\nAssistant: {response[:200]}",
            memory_type=MemoryType.EPISODIC,
            source_interaction_id=interaction_id,
            salience_score=0.2 if is_speculative else 0.5,
            tags=["interaction"] + (["speculative"] if is_speculative else []),
        )
        if is_speculative:
            print(f"  Fabrication detected: episodic stored as speculative (salience=0.2)")
        self.memory.store(episodic)

        reflection_calls = 0
        if self.reflection_engine and self.interaction_count % 3 == 0:
            try:
                reflections = self.reflection_engine.reflect(
                    user_message, response, interaction_id
                )
                if reflections:
                    reflection_calls = len(reflections)
                    print(f"  Reflection: {reflection_calls} new memories generated")
            except Exception as e:
                print(f"  Reflection error (non-fatal): {e}")

        if self.metrics:
            self.metrics.log_compute(
                interaction_id=interaction_id,
                prompt_tokens=prompt_tokens,
                response_tokens=response_tokens,
                elapsed_seconds=elapsed,
                memory_ops=1,
                reflection_calls=reflection_calls,
            )
            self.metrics.log_drift_sample(
                interaction_count=self.interaction_count,
                response=response,
            )
            if directives is not None:
                self.metrics.snapshot_directives(
                    interaction_count=self.interaction_count,
                    directives=directives,
                )

        if self.log_dir:
            self._log(interaction_id, user_message, response, prompt, elapsed)

        return response

    def feedback(self, rating, correction=None, target_query=None):
        """Process explicit feedback.
        
        Args:
            rating: 1-5 scale
            correction: Optional correction text
            target_query: Optional — what the feedback is ABOUT. If provided,
                         salience adjustments target memories related to this
                         query instead of the last interaction. Useful when
                         correcting something from 2+ interactions ago.
                         If not provided, defaults to last user message.
        """
        if not target_query:
            if len(self.history) < 2:
                print("No interaction to rate and no target query provided.")
                return
            target_query = self.history[-2]["content"]

        # Adjust salience of memories related to the target
        recent = self.memory.retrieve(target_query, top_k=3)
        delta = (rating - 3) * 0.1
        for mem in recent:
            self.memory.adjust_salience(mem.id, delta)

        # Store correction as high-salience semantic memory
        if correction:
            correction_mem = MemoryUnit(
                content=f"CORRECTION: {correction}",
                memory_type=MemoryType.SEMANTIC,
                salience_score=0.9,
                confidence=1.0,
                tags=["user_correction", "high_priority"],
            )
            self.memory.store(correction_mem)
            print(f"Stored correction: {correction}")

            # Kill contradicting directives
            if self.reflection_engine:
                killed = self.reflection_engine.check_correction_conflicts(correction)
                if killed:
                    print(f"  Correction killed {len(killed)} conflicting directive(s)")

            # Supersede contradicting memories
            superseded = self.memory.supersede_by_correction(
                correction_text=correction,
                correction_id=correction_mem.id,
                similarity_threshold=0.45,
            )
            if superseded:
                print(f"  Correction superseded {len(superseded)} contradicting memories")

        print(f"Feedback recorded: rating={rating}, target='{target_query[:50]}...', "
              f"adjusted {len(recent)} memories by {delta:+.2f}")

    def score_retrieval(self, relevance_scores):
        if not self.metrics:
            print("Metrics not enabled.")
            return
        if len(self.history) < 2:
            print("No interaction to score.")
            return
        last_user = self.history[-2]["content"]
        retrieved = self.memory.retrieve(last_user, top_k=10)
        if len(relevance_scores) != len(retrieved):
            print(f"Expected {len(retrieved)} scores, got {len(relevance_scores)}")
            return
        precision = self.metrics.score_retrieval(
            interaction_id="manual", query=last_user,
            retrieved_memories=retrieved, relevance_scores=relevance_scores,
        )
        print(f"Retrieval precision: {precision:.3f}")

    def log_adaptation(self, event_type, description):
        if not self.metrics:
            print("Metrics not enabled.")
            return
        self.metrics.log_adaptation(
            interaction_id="manual", event_type=event_type, description=description,
        )
        print(f"Adaptation event logged: {event_type}")

    def evaluation_report(self):
        if not self.metrics:
            print("Metrics not enabled.")
            return
        self.metrics.print_report()

    def status(self):
        stats = self.memory.get_stats()
        directive_count = 0
        max_directives = 0
        if self.reflection_engine:
            directive_count, max_directives = self.reflection_engine.get_directive_count()
        return {
            "interaction_count": self.interaction_count,
            "history_length": len(self.history),
            "memory": stats,
            "active_directives": directive_count,
            "max_directives": max_directives,
            "reflection_enabled": self.reflect_enabled,
            "metrics_enabled": self.metrics is not None,
            "context_window": self.assembler.max_context,
        }

    def update_log_feedback(self, interaction_id, updates):
        """Patch feedback data into an existing interaction log file.

        Args:
            interaction_id: UUID of the interaction to update. No-op if None
                            or the log file doesn't exist.
            updates: Dict with any subset of:
                       rating          → int, overwrites existing value
                       corrections     → list of {type, content} dicts, appended
                       segment_ratings → list of {sentence_index, sentence, rating}, appended
        """
        if not self.log_dir or not interaction_id:
            return
        import os
        path = os.path.join(self.log_dir, f"{interaction_id}.json")
        if not os.path.exists(path):
            return
        with open(path, "r") as f:
            entry = json.load(f)
        if "rating" in updates:
            entry["rating"] = updates["rating"]
        if "corrections" in updates:
            entry.setdefault("corrections", []).extend(updates["corrections"])
        if "segment_ratings" in updates:
            entry.setdefault("segment_ratings", []).extend(updates["segment_ratings"])
        with open(path, "w") as f:
            json.dump(entry, f, indent=2)

    def _log(self, interaction_id, user_message, response, prompt, elapsed):
        import os
        os.makedirs(self.log_dir, exist_ok=True)
        log_entry = {
            "id": interaction_id,
            "timestamp": time.time(),
            "user_message": user_message,
            "response": response,
            "prompt_tokens": self.engine.count_tokens(prompt),
            "response_tokens": self.engine.count_tokens(response),
            "elapsed_seconds": round(elapsed, 2),
            "memory_count": self.memory.collection.count(),
            "rating": None,
            "corrections": [],
            "segment_ratings": [],
        }
        path = os.path.join(self.log_dir, f"{interaction_id}.json")
        with open(path, "w") as f:
            json.dump(log_entry, f, indent=2)
