import time
import json
import uuid
from core.memory_store import MemoryUnit, MemoryType


class InteractionLoop:
    """Main loop: assembles context, generates response, stores memories."""

    def __init__(self, engine, memory_store, assembler, log_dir=None):
        self.engine = engine
        self.memory = memory_store
        self.assembler = assembler
        self.log_dir = log_dir
        self.history = []
        self.interaction_count = 0

    def chat(self, user_message):
        self.interaction_count += 1
        interaction_id = str(uuid.uuid4())
        start_time = time.time()

        # Assemble context
        prompt = self.assembler.assemble(
            user_message=user_message,
            conversation_history=self.history,
        )

        # Generate response
        response = self.engine.generate(prompt)

        elapsed = time.time() - start_time

        # Update conversation history
        self.history.append({"role": "user", "content": user_message})
        self.history.append({"role": "assistant", "content": response})

        # Keep history manageable (last 10 turns = 20 messages)
        if len(self.history) > 20:
            self.history = self.history[-20:]

        # Store episodic memory
        episodic = MemoryUnit(
            content=f"User: {user_message}\nAssistant: {response[:200]}",
            memory_type=MemoryType.EPISODIC,
            source_interaction_id=interaction_id,
            salience_score=0.5,
            tags=["interaction"],
        )
        self.memory.store(episodic)

        # Log interaction
        if self.log_dir:
            self._log(interaction_id, user_message, response, prompt, elapsed)

        return response

    def feedback(self, rating, correction=None):
        """Process explicit feedback on the last interaction."""
        if len(self.history) < 2:
            print("No interaction to rate.")
            return

        last_user = self.history[-2]["content"]
        last_assistant = self.history[-1]["content"]

        # Adjust salience of recent episodic memories
        recent = self.memory.retrieve(last_user, top_k=3)
        delta = (rating - 3) * 0.1  # rating 1-5 maps to -0.2 to +0.2
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

        print(f"Feedback recorded: rating={rating}, adjusted {len(recent)} memories by {delta:+.2f}")

    def status(self):
        stats = self.memory.get_stats()
        return {
            "interactions": self.interaction_count,
            "history_length": len(self.history),
            "memory": stats,
        }

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
        }
        path = os.path.join(self.log_dir, f"{interaction_id}.json")
        with open(path, "w") as f:
            json.dump(log_entry, f, indent=2)
