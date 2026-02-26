class ContextAssembler:
    """Packs memory, history, directives, and user message into a prompt."""

    def __init__(self, engine, memory_store, max_context=4096, max_response=512):
        self.engine = engine
        self.memory = memory_store
        self.max_context = max_context
        self.max_response = max_response
        self.available_tokens = max_context - max_response

        self.system_prompt = (
            "You are a helpful AI assistant engaged in an ongoing relationship "
            "with your user. You have access to memories from previous interactions. "
            "Use these memories naturally to provide personalized, contextual responses. "
            "Be concise and direct."
        )

    def assemble(self, user_message, conversation_history=None, directives=None):
        sections = []
        token_budget = self.available_tokens

        # 1. System prompt (fixed cost)
        system_section = f"[INST] {self.system_prompt}"
        system_tokens = self.engine.count_tokens(system_section)
        token_budget -= system_tokens

        # 2. User message (fixed cost, reserve first)
        user_section = f"\nUser: {user_message} [/INST]"
        user_tokens = self.engine.count_tokens(user_section)
        token_budget -= user_tokens

        # 3. Directives (high priority, usually small)
        directive_section = ""
        if directives:
            directive_text = "\n".join(f"- {d}" for d in directives)
            directive_section = f"\n\nActive directives:\n{directive_text}"
            directive_tokens = self.engine.count_tokens(directive_section)
            if directive_tokens < token_budget * 0.1:
                token_budget -= directive_tokens
            else:
                directive_section = ""

        # 4. Retrieved memories (up to 30% of remaining budget)
        memory_budget = int(token_budget * 0.4)
        memory_section = ""
        if self.memory and self.memory.collection.count() > 0:
            memories = self.memory.retrieve(user_message, top_k=10)
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
            "utilization": f"{total / self.available_tokens * 100:.1f}%"
        }
