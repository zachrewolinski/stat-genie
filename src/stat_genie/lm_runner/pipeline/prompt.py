
class PromptGenerator:

    def __init__(self, system_prompt: str, instruction_prompt: str,
                 post_fix: str, example: str):
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

    