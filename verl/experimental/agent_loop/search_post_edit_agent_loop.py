"""Fail-fast Hermes rollout supporting INST or fitted thinking-tag responses."""

import asyncio
import copy
import json
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopOutput,
    register,
)
from verl.tools.search_post_edit_protocol import (
    TOOL_NAME,
    ProtocolError,
    loads,
    parse_turn,
    resolve_format,
    validate_budget,
    validate_queries,
)
from verl.tools.translation_search_tool import SearchServiceError, TranslationSearchTool
from verl.tools.utils.tool_registry import initialize_tools_from_config
from verl.utils.profiler import simple_timer


@register("search_post_edit_agent")
class SearchPostEditAgentLoop(AgentLoopBase):
    @classmethod
    def init_class(cls, config, tokenizer, processor, **kwargs):
        if cls.__dict__.get("_class_initialized", False):
            return
        if processor is not None:
            raise ValueError("SearchPostEditAgentLoop supports text-only models")
        rollout = config.actor_rollout_ref.rollout
        if rollout.multi_turn.get("response_mask_mode", "all_assistant") != "all_assistant":
            raise ValueError(
                "Search post-edit learns all assistant turns; use response_mask_mode=all_assistant"
            )
        if rollout.multi_turn.format != "hermes":
            raise ValueError("SearchPostEditAgentLoop requires the Hermes format")
        path = rollout.multi_turn.tool_config_path
        tools = initialize_tools_from_config(path) if path else []
        if len(tools) != 1 or not isinstance(tools[0], TranslationSearchTool) or tools[0].name != TOOL_NAME:
            raise ValueError("Configure exactly one TranslationSearchTool")
        cls.search_tool = tools[0]
        cls._class_initialized = True

    async def _template(self, messages, **options):
        kwargs = dict(self.config.data.get("apply_chat_template_kwargs", {}))
        kwargs.update(options)
        if getattr(self, '_prompt_type', 'inst') == 'fit':
            kwargs['enable_thinking'] = True
        return await asyncio.to_thread(self.tokenizer.apply_chat_template, messages, tokenize=True, **kwargs)

    async def run(self, sampling_params, **kwargs):
        extra = kwargs["extra_info"]
        max_rounds = extra["max_search_rounds"]
        validate_budget(max_rounds)
        spans = loads(extra["spans"]) if isinstance(extra["spans"], str) else list(extra["spans"])
        if len(spans) != extra["span_count"]:
            raise ValueError("span_count metadata mismatch")
        rollout = self.config.actor_rollout_ref.rollout
        prompt_type, think_tag = resolve_format(extra)
        self._prompt_type = prompt_type
        response_limit = rollout.response_length
        schema = copy.deepcopy(self.search_tool.tool_schema.model_dump(exclude_none=True))
        schema["function"]["parameters"]["properties"]["queries"]["maxItems"] = max(1, len(spans))
        if prompt_type == 'fit':
            schema['function'].pop('strict', None)
            schema = json.loads(json.dumps(schema, sort_keys=True))
        prompt_ids = await self._template(
            list(kwargs["raw_prompt"]),
            tools=[schema] if spans and max_rounds else [],
            add_generation_prompt=True,
        )
        if len(prompt_ids) > rollout.prompt_length:
            raise ValueError("Prompt including tool schema exceeds rollout.prompt_length")
        model_limit = rollout.get("max_model_len") or (rollout.prompt_length + response_limit)
        response_limit = min(response_limit, model_limit - len(prompt_ids))
        if response_limit <= 0:
            raise ValueError("Initial prompt leaves no model context for a response")
        initial_prompt = list(prompt_ids)
        response_ids, mask, logprobs = [], [], []
        logprobs_available = True
        metrics, trace = {}, []
        request_id = uuid4().hex
        rounds = calls = turns = 0
        reason, error, translation = "invalid_output", "", None
        instance_id, _ = await self.search_tool.create(create_kwargs={"span_count": len(spans)})
        try:
            while True:
                remaining = response_limit - len(response_ids)
                if remaining <= 0:
                    reason = "length_limit"
                    break
                with simple_timer("generate_sequences", metrics):
                    generated = await self.server_manager.generate(
                        request_id=request_id,
                        prompt_ids=list(prompt_ids),
                        sampling_params=copy.deepcopy(sampling_params),
                        image_data=None,
                    )
                turns += 1
                ids = generated.token_ids[:remaining]
                response_ids.extend(ids)
                prompt_ids.extend(ids)
                mask.extend([1] * len(ids))
                if generated.log_probs is not None:
                    if len(generated.log_probs) != len(generated.token_ids):
                        raise ValueError("Generated token/logprob length mismatch")
                    logprobs.extend(generated.log_probs[:remaining])
                else:
                    logprobs_available = False
                    logprobs.extend([0.0] * len(ids))
                if prompt_type == 'fit':
                    # Preserve thinking delimiters even if registered as special tokens.
                    text = self.tokenizer.decode(ids, skip_special_tokens=False)
                    eos = getattr(self.tokenizer, 'eos_token', None)
                    if eos and text.endswith(eos):
                        text = text[:-len(eos)]
                else:
                    text = self.tokenizer.decode(ids, skip_special_tokens=True)
                trace.append({"role": "assistant", "content": text})
                if not ids or len(generated.token_ids) >= remaining:
                    reason = "length_limit"
                    break
                try:
                    action, value = parse_turn(text, prompt_type=prompt_type, think_tag=think_tag)
                    if action == "finish":
                        translation, reason = value, "finished"
                        break
                    if rounds >= max_rounds:
                        raise ProtocolError("search_round_budget_exhausted")
                    body = validate_queries(value, len(spans))
                except ProtocolError as exc:
                    reason, error = "invalid_output", str(exc)
                    break
                # All parameters are validated before any external request.
                rounds += 1
                calls += len(body["queries"])
                try:
                    with simple_timer("tool_calls", metrics):
                        observation, _, metadata = await self.search_tool.execute(instance_id, body)
                except SearchServiceError as exc:
                    reason, error = "search_failed", str(exc)
                    break
                evidence = loads(observation.text)
                evidence["search_rounds_remaining"] = max_rounds - rounds
                observation.text = json.dumps(evidence, ensure_ascii=False, allow_nan=False)
                trace.append({"role": "tool", "content": observation.text, "metadata": metadata})
                # Use the existing Hermes incremental framing; preserve original generated IDs.
                tool_ids = await self._template(
                    [{"role": "tool", "content": observation.text}],
                    add_generation_prompt=True,
                )
                prefix = await self._template([{}], add_generation_prompt=False)
                if prefix:
                    if tool_ids[: len(prefix)] != prefix:
                        raise ValueError("Chat template does not support Hermes incremental tool framing")
                    tool_ids = tool_ids[len(prefix) :]
                # Qwen generation normally stops on im_end before its template newline.
                # Supply that observation-side separator without changing sampled tokens.
                eos = getattr(self.tokenizer, "eos_token", None)
                if eos and self.tokenizer.decode(ids, skip_special_tokens=False).endswith(eos):
                    tool_ids = self.tokenizer.encode("\n", add_special_tokens=False) + tool_ids
                if len(response_ids) + len(tool_ids) >= response_limit:
                    reason = "context_limit"
                    break
                prompt_ids.extend(tool_ids)
                response_ids.extend(tool_ids)
                mask.extend([0] * len(tool_ids))
                logprobs.extend([0.0] * len(tool_ids))
                # The next generation remains available after the final search round.
        finally:
            await self.search_tool.release(instance_id)
        if rollout.calculate_log_probs and not logprobs_available:
            raise ValueError("Rollout log probabilities were requested but not returned")
        return AgentLoopOutput(
            prompt_ids=initial_prompt,
            response_ids=response_ids,
            response_mask=mask,
            response_logprobs=logprobs if logprobs_available else None,
            num_turns=1 + turns + rounds,
            metrics=metrics,
            extra_fields={
                "post_edit_translation": translation or "",
                "post_edit_termination": reason,
                "post_edit_error": error,
                "post_edit_search_rounds": rounds,
                "post_edit_search_calls": calls,
                "post_edit_trace": json.dumps(trace, ensure_ascii=False, allow_nan=False),
                "post_edit_prompt_type": prompt_type,
                "post_edit_think_tag": think_tag,
            },
        )
