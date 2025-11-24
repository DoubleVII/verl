## 目标

* 为 `VLLMRewardModelWorker` 引入与 rollout 等价的显存释放（sleep/wake）机制，以在每次奖励推理后主动释放 KV cache/权重占用，降低显存占用。

* 保持现有调用方式不变（`rm_wg.compute_rm_score(...)`），在 Worker 内部自动处理生命周期；同时暴露可选的 `resume/release` 接口以便上层显式控制。

## 参考实现

* Rollout 同步模式采用 `LLM(..., enable_sleep_mode=free_cache_engine)` 并在生成后调用 `reset_prefix_cache()` + `sleep(level)`；推理前根据 `tags` 调用 `wake_up(tags=...)`。

  * 唤醒/释放：`verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py:438–460`

  * 构造引擎：`LLM(..., enable_sleep_mode=config.free_cache_engine, ...)` `vllm_rollout_spmd.py:219–241`

  * `sleep_level` 选择：`layered_summon` 或 `expert_parallel_size>1` → `1`，否则 `VLLM_SLEEP_LEVEL` `vllm_rollout_spmd.py:523–526`

## 具体改动

1. 引擎初始化：

   * 在 `verl/workers/reward_model/vllm_worker.py` 的 `__init__` 中：

     * 读取 `config.rollout.free_cache_engine` 并传入 `LLM(..., enable_sleep_mode=config.rollout.free_cache_engine, ...)`。

     * 依据 `config.rollout.layered_summon` 和 `config.rollout.expert_parallel_size` 设置 `self.sleep_level = 1` 或 `VLLM_SLEEP_LEVEL`。

     * 引入 `from verl.third_party.vllm import VLLM_SLEEP_LEVEL` 以统一 sleep 深度。

2. 生命周期接口：

   * 为 `VLLMRewardModelWorker` 新增：

     * `async def resume(self, tags: list[str])`: 当 `free_cache_engine=True` 时调用 `self.llm.wake_up(tags=tags)`；否则直接返回。

     * `async def release(self)`: 先调用 `self.llm.reset_prefix_cache()`，若 `free_cache_engine=True` 则 `self.llm.sleep(level=self.sleep_level)`。

   * 两个方法使用 `@register(dispatch_mode=Dispatch.ONE_TO_ALL)` 暴露给 `RayWorkerGroup`，与 rollout 的调用语义一致。

3. 生成流程包裹：

   * 在 `compute_rm_score` 开始前后，按需包裹：

     * 前置：`await self.resume(tags=["weights"])`（奖励模型权重通常静态，无需多阶段，但允许 tags）。

     * 调用 `self.llm.generate(prompts, self.sampling_params)`。

     * 后置：`await self.release()`（包含 `reset_prefix_cache()`）。

   * 当 `config.rollout.free_cache_engine=False` 时，以上调用被自动短路，无额外开销。

4. 兼容 Reward Server 模式：

   * 若使用 `RewardModelManager` 启动 vLLM Server（AgentLoop/独立资源池），现有 `wake_up()/sleep()` 已在每次请求前后调用，无需重复；Worker 内部包裹仅在 Worker 直连模式生效。

## 验证与观测

* 在本地单卡运行一次 `compute_rm_score` 前后，通过 `torch.cuda.memory_allocated()` 观测显存释放（或使用现有 `GPUMemoryLogger`）。

* 压测批量场景，确认生成后 `reset_prefix_cache()` 有效并无残留占用；性能回归可通过 `free_cache_engine=False` 比较。

## 兼容性与风险

* vLLM 版本需支持 `LLM.sleep/wake_up/reset_prefix_cache`（与当前 rollout 相同接口）。

* `custom_processor` 仍为必需；本改动不干扰其输入/输出逻辑，仅包裹生命周期。

* 若启用 `layered_summon`/`expert_parallel_size>1`，使用浅层 `sleep_level=1` 保留必要状态，避免频繁重建。

