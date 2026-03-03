import os


_FALLBACK_SYSTEM_PROMPT = "You are a helpful assistant with persistent memory."


class ContextAssembler:
    """Packs memory, history, directives, and user message into a prompt.

    v4 changes:
    - Sable identity system prompt replaces generic assistant prompt
    - ~600 tokens fixed cost — justified at 16K context (3.9% of budget)
    - System prompt loaded from external file (sable_system_prompt.txt)
    """

    def __init__(self, engine, memory_store, max_context=16384, max_response=1024,
                 system_prompt_path=None):
        self.engine = engine
        self.memory = memory_store
        self.max_context = max_context
        self.max_response = max_response
        self.available_tokens = max_context - max_response

        if system_prompt_path and os.path.exists(system_prompt_path):
            with open(system_prompt_path, "r") as f:
                self.system_prompt = f.read().strip()
            print(f"Loaded system prompt from {system_prompt_path}")
        else:
            self.system_prompt = _FALLBACK_SYSTEM_PROMPT
            if system_prompt_path:
                print(f"Warning: system prompt file not found at {system_prompt_path}, using fallback")

    def assemble(self, user_message, conversation_history=None, directives=None):
        token_budget = self.available_tokens

        # 1. System prompt (fixed cost)
        system_section = f"[INST] {self.system_prompt}"
        system_tokens = self.engine.count_tokens(system_section)
        token_budget -= system_tokens

        # 2. User message (fixed cost, reserve first)
        user_section = f"\nUser: {user_message} [/INST]"
        user_tokens = self.engine.count_tokens(user_section)
        token_budget -= user_tokens

        # 3. Directives — split by origin
        #    Human-origin: guaranteed 6% of budget, never truncated
        #    System-origin: up to 4% of budget, truncated if over
        directive_section = ""
        if directives:
            human_budget = token_budget * 0.06
            system_budget = token_budget * 0.04

            human_dirs = [d for d in directives if d.origin == "human"]
            system_dirs = [d for d in directives if d.origin != "human"]

            # Human directives — always included (guaranteed budget)
            human_lines = [f"- {d.content}" for d in human_dirs]
            human_text = ""
            human_tokens = 0
            if human_lines:
                human_text = "\n\nUser directives:\n" + "\n".join(human_lines)
                human_tokens = self.engine.count_tokens(human_text)

            # System directives — truncate if over budget
            system_lines = []
            system_tokens = 0
            for d in system_dirs:
                line = f"- {d.content}"
                lt = self.engine.count_tokens(line)
                if system_tokens + lt > system_budget:
                    break
                system_lines.append(line)
                system_tokens += lt

            system_text = ""
            if system_lines:
                system_text = "\n\nSystem directives:\n" + "\n".join(system_lines)
                system_tokens = self.engine.count_tokens(system_text)

            directive_section = human_text + system_text
            token_budget -= (human_tokens + system_tokens)

        # 4. Retrieved memories (up to 30% of remaining budget)
        memory_budget = int(token_budget * 0.3)
        memory_section = ""
        if self.memory and self.memory.collection.count() > 0:
            memories = self.memory.retrieve(user_message, top_k=15)
            if memories:
                memory_lines = []
                used_tokens = 0
                for mem in memories:
                    line = f"[{mem.memory_type.value}] {mem.content}"
                    line_tokens = self.engine.count_tokens(line)
                    if used_tokens + line_tokens > memory_budget:
                        break
                    memory_lines.append(line)
                    used_tokens += line_tokens
                if memory_lines:
                    memory_section = "\n\nRelevant memories:\n" + "\n".join(memory_lines)
                    token_budget -= used_tokens

        # 5. Conversation history (fill remaining space, most recent first)
        history_section = ""
        if conversation_history:
            history_lines = []
            used_tokens = 0
            for turn in reversed(conversation_history):
                line = f"{turn['role'].capitalize()}: {turn['content']}"
                line_tokens = self.engine.count_tokens(line)
                if used_tokens + line_tokens > token_budget:
                    break
                history_lines.insert(0, line)
                used_tokens += line_tokens
            if history_lines:
                history_section = "\n\nRecent conversation:\n" + "\n".join(history_lines)

        # Assemble final prompt
        prompt = system_section + directive_section + memory_section + history_section + user_section

        return prompt

    def get_token_report(self, prompt):
        total = self.engine.count_tokens(prompt)
        return {
            "prompt_tokens": total,
            "max_context": self.max_context,
            "max_response": self.max_response,
            "remaining_for_response": self.max_context - total,
            "utilization": f"{total / self.available_tokens * 100:.1f}%",
        }
