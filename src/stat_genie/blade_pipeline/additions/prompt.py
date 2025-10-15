from stat_genie.blade_pipeline.baselines.lm.gen_analysis import \
    SYSTEM_PROMPT, INSTRUCTION_PROMPT, EXAMPLE, POST_FIX

class PromptGenerator:
    """
    Custom class that allows for customizable prompts for the LLM.
    Follows the BLADE style of system prompt, instruction prompt, post-fix,
    and example prompt.
    """

    def __init__(self, system_prompt: str = SYSTEM_PROMPT,
                 instruction_prompt: str = INSTRUCTION_PROMPT,
                 post_fix: str = POST_FIX, example: str = EXAMPLE):
        self.system_prompt = system_prompt
        self.instruction_prompt = instruction_prompt
        self.post_fix = post_fix
        self.example = example

    def get_prompts(self) -> dict:
        return {
            "system": self.system_prompt,
            "instruction": self.instruction_prompt,
            "post_fix": self.post_fix,
            "example": self.example
        }

    