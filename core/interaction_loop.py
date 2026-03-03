import re
import time
import json
import uuid
import numpy as np
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
        self._last_retrieved = []
        self._last_log_path = None
        self._last_verify_status = "PASS"

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

        # Retrieve memories for this interaction (used by hallucination filter + tracking)
        retrieved = self.memory.retrieve(user_message, top_k=10)
        self._last_retrieved = [{"id": m.id, "content": m.content} for m in retrieved]

        user_lower = user_message.lower()
        is_speculative = any(trigger in user_lower for trigger in _FABRICATION_TRIGGERS)

        # Hallucination pre-filter: check response claims against source material
        is_unverified = False
        if not is_speculative and retrieved:
            reference_text = user_message + "\n" + "\n".join(m.content for m in retrieved)
            ref_lower = reference_text.lower()

            claims = self._extract_claims(response)
            if claims:
                unmatched = sum(1 for c in claims if c not in ref_lower)
                if unmatched / len(claims) > 0.3:
                    is_unverified = True

        if is_speculative:
            salience = 0.2
            tags = ["interaction", "speculative"]
        elif is_unverified:
            salience = 0.3
            tags = ["interaction", "unverified"]
        else:
            salience = 0.5
            tags = ["interaction"]

        episodic = MemoryUnit(
            content=f"User: {user_message}\nAssistant: {response[:200]}",
            memory_type=MemoryType.EPISODIC,
            source_interaction_id=interaction_id,
            salience_score=salience,
            tags=tags,
        )
        self.memory.store(episodic)

        reflection_calls = 0
        reflection_throttled = False
        if self.reflection_engine:
            try:
                reflections = self.reflection_engine.reflect(
                    user_message, response, interaction_id,
                    interaction_count=self.interaction_count,
                )
                if reflections:
                    reflection_calls = len(reflections)
                reflection_throttled = (self.interaction_count % 3 != 0)
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
                    directives=[d.content for d in directives],
                )

        if self.log_dir:
            self._log(interaction_id, user_message, response, prompt, elapsed,
                      directives=directives)

        # Console summary
        directive_count = 0
        if self.reflection_engine:
            directive_count, _ = self.reflection_engine.get_directive_count()
        verify_status = "SPECULATIVE" if is_speculative else ("UNVERIFIED" if is_unverified else "PASS")
        self._last_verify_status = verify_status
        if reflection_throttled:
            refl_desc = f"depth 0 only (throttled) | {reflection_calls} new memor{'y' if reflection_calls == 1 else 'ies'}"
        elif self.reflection_engine:
            refl_desc = f"full pass | {reflection_calls} new memor{'y' if reflection_calls == 1 else 'ies'}"
        else:
            refl_desc = "disabled"
        print(f"\u2500\u2500 Interaction {self.interaction_count} " + "\u2500" * max(0, 46 - len(str(self.interaction_count))))
        print(f"  User:       {user_message[:60]}{'...' if len(user_message) > 60 else ''}")
        print(f"  Response:   {response[:80]}{'...' if len(response) > 80 else ''}")
        print(f"  Retrieved:  {len(retrieved)} memories | Total: {self.memory.collection.count()} | Directives: {directive_count}")
        print(f"  Reflection: {refl_desc}")
        print(f"  Verify:     {verify_status}")
        print(f"  Time:       {elapsed:.2f}s")
        print("\u2500" * 48)

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

        # Tag last-retrieved memories for passive retrieval precision tracking
        if self._last_retrieved:
            tag = "confirmed_relevant" if rating >= 4 else ("flagged_irrelevant" if rating <= 2 else None)
            if tag:
                for entry in self._last_retrieved:
                    self._append_tag(entry["id"], tag)

        # Store correction as high-salience semantic memory
        killed_directives = []
        superseded_count = 0
        if correction:
            correction_mem = MemoryUnit(
                content=f"CORRECTION: {correction}",
                memory_type=MemoryType.SEMANTIC,
                salience_score=0.9,
                confidence=1.0,
                tags=["user_correction", "high_priority"],
                origin="human",
            )
            self.memory.store(correction_mem)

            # Kill contradicting directives
            if self.reflection_engine:
                killed_directives = self.reflection_engine.check_correction_conflicts(correction)

            # Supersede contradicting memories
            superseded = self.memory.supersede_by_correction(
                correction_text=correction,
                correction_id=correction_mem.id,
                similarity_threshold=0.45,
            )
            superseded_count = len(superseded) if superseded else 0

            # Auto-supersede system directives that conflict with this correction
            if self.engine is not None:
                correction_emb = self.engine.get_embedding(correction_mem.content)
                procedural = self.memory.retrieve_by_type(MemoryType.PROCEDURAL)
                system_directives = [m for m in procedural if m.origin == "system"]
                for sd in system_directives:
                    sd_emb = self.engine.get_embedding(sd.content)
                    similarity = float(np.dot(correction_emb, sd_emb))
                    if similarity > 0.50:
                        self.memory.supersede(sd.id, correction_mem.id)
                        killed_directives.append(sd.content)
                        superseded_count += 1

        # Update JSON log for last interaction
        if self._last_log_path:
            self._update_log_rating(rating, correction)

        # Console summary
        correction_trunc = None
        if correction:
            correction_trunc = correction[:50] + ("..." if len(correction) > 50 else "")
        print(f"\u2500\u2500 Feedback " + "\u2500" * 39)
        print(f"  Rating:      {rating} ({delta:+.2f} to {len(recent)} memories)")
        print(f"  Correction:  {correction_trunc or 'None'}")
        print(f"  Superseded:  {superseded_count} system directives")
        for kd in killed_directives:
            print(f'    killed: "{kd[:60]}..."')
        print("\u2500" * 48)

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

    def _append_tag(self, memory_id, tag):
        """Append a tag to an existing memory's tag list in ChromaDB."""
        results = self.memory.collection.get(ids=[memory_id])
        if not results["ids"]:
            return
        metadata = results["metadatas"][0]
        tags = json.loads(metadata.get("tags", "[]"))
        if tag not in tags:
            tags.append(tag)
            metadata["tags"] = json.dumps(tags)
            self.memory.collection.update(ids=[memory_id], metadatas=[metadata])

    def _update_log_rating(self, rating, correction=None):
        """Update the most recent JSON log file with feedback data."""
        import os
        if not self._last_log_path or not os.path.exists(self._last_log_path):
            return
        with open(self._last_log_path, "r") as f:
            entry = json.load(f)
        entry["rating"] = rating
        if correction:
            entry.setdefault("corrections", []).append({
                "type": "user_correction",
                "content": correction,
            })
        with open(self._last_log_path, "w") as f:
            json.dump(entry, f, indent=2)

    def get_retrieval_stats(self):
        """Count memories tagged as relevant vs irrelevant through passive feedback.

        Returns dict with total_tagged, confirmed_relevant, flagged_irrelevant,
        and precision ratio (relevant / total_tagged, or None if no data).
        """
        all_data = self.memory.collection.get(include=["metadatas"])
        relevant = 0
        irrelevant = 0
        for metadata in all_data["metadatas"]:
            tags = json.loads(metadata.get("tags", "[]"))
            if "confirmed_relevant" in tags:
                relevant += 1
            if "flagged_irrelevant" in tags:
                irrelevant += 1
        total = relevant + irrelevant
        return {
            "total_tagged": total,
            "confirmed_relevant": relevant,
            "flagged_irrelevant": irrelevant,
            "precision": relevant / total if total > 0 else None,
        }

    @staticmethod
    def _extract_claims(text):
        """Extract verifiable claims from text using simple heuristics.

        Targets: multi-word capitalized sequences (proper nouns / named entities),
        numbers that are specific enough to matter (3+ digits, decimals, percentages),
        and quoted statements (5+ chars).  Returns a set of lowercase strings.
        """
        claims = set()
        # Multi-word capitalized sequences (e.g. "New York", "Albert Einstein")
        for m in re.finditer(r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+', text):
            claims.add(m.group().lower())
        # Specific numbers: 3+ digits, decimals, or percentages
        for m in re.finditer(r'\b\d{3,}(?:[.,]\d+)*\b|\b\d+\.\d+\b|\b\d+%', text):
            claims.add(m.group())
        # Quoted statements (5+ chars inside quotes)
        for m in re.finditer(r'"([^"]{5,})"', text):
            claims.add(m.group(1).lower())
        return claims

    def _log(self, interaction_id, user_message, response, prompt, elapsed,
             directives=None):
        import os
        os.makedirs(self.log_dir, exist_ok=True)

        # Cumulative session metrics
        procedural = self.memory.retrieve_by_type(MemoryType.PROCEDURAL)
        human_procedural = sum(1 for m in procedural if m.origin == "human")
        system_procedural = sum(1 for m in procedural if m.origin == "system")

        stats = self.memory.get_stats()
        avg_salience = stats.get("avg_salience", 0.0)

        directive_snapshot = []
        if directives:
            directive_snapshot = [
                {"content": d.content, "origin": d.origin} for d in directives
            ]

        log_entry = {
            "id": interaction_id,
            "timestamp": time.time(),
            "user_message": user_message,
            "response": response,
            "prompt_tokens": self.engine.count_tokens(prompt),
            "response_tokens": self.engine.count_tokens(response),
            "elapsed_seconds": round(elapsed, 2),
            "memory_count": self.memory.collection.count(),
            "procedural_human": human_procedural,
            "procedural_system": system_procedural,
            "avg_salience": round(avg_salience, 4),
            "directives": directive_snapshot,
            "rating": None,
            "corrections": [],
            "segment_ratings": [],
        }
        path = os.path.join(self.log_dir, f"{interaction_id}.json")
        with open(path, "w") as f:
            json.dump(log_entry, f, indent=2)
        self._last_log_path = path
