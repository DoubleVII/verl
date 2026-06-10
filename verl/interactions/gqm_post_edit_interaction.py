from typing import Any

from verl.interactions.base import BaseInteraction


GQM_POST_EDIT_PROMPT = (
    "Using the source text, candidate translations, and evaluation above, provide the final improved translation "
    "in the target language. Include a concise step-by-step analysis, then output only the final translation in a "
    "single Markdown code block."
)


class GQMPostEditInteraction(BaseInteraction):
    """Interaction that asks for post-editing after a first-turn GQM response."""

    async def generate_response(
        self, instance_id: str, messages: list[dict[str, Any]], **kwargs
    ) -> tuple[bool, str, float, dict[str, Any]]:
        return False, self.config.get("prompt", GQM_POST_EDIT_PROMPT), 0.0, {}
