# 8788 摘要生成方法与提示

> 本文是 2026-10-25 整理的脱离个人数据的新开发资料副本；来源定版于 2026-10-23。未携带个人记忆正文、原始对话或证据附件。历史执行描述不构成本包的运行或验收证明。

选中的是生成方法。本包没有 idea.md、stable.md、完整注入样本、checkpoint、benchmark 或 evidence/source sidecar；不能据本包字节复现原个人摘要。来源正文保留在只读源项目，排除清单见[来源与删减说明](../来源与删减说明.md)。

## 事件层：从空记忆按时间连续追加

实际历史入口是 reference 中的 [run_experiment.py](../../../reference/work/memory-route-comparison/custom/run_experiment.py)。它通过 sys.path 选择 **letta-prototype** 的 MemoryWorker → verify_source_ids → memory_agent_consolidate → ideastore.save。不能拿 generated-memory-demo 的当前 agents.py 替代真实生成来源；两个目录的 worker 即使相同，agents/prompts 也可能不同。

1. 工作记忆从 `# 长期记忆\n` 起步，在独立实例中按历史时间分批。
2. 上游历史构建保留选中分支的用户消息；同会话间隔不超过 180 秒的相邻用户消息可聚合，同时保留原 source_ids。
3. 每批设置自己的 now，排 suggestion，statement 为 null。worker 按 ID 读取并验证原 user 证据，再给模型完整现有记忆、本批证据及 message_at。
4. 模型返回一段 `### 更新 YYYY-MM-DD` Markdown；只追加，不每批重写旧文。纠正说明替代关系；supersedes 不会自动删除旧文。
5. 保存更新及任务 marker，下一批接此前完整文本。版本数可能多于批次数，不能把版本整数当处理批数。

历史演示只覆盖人工选择的主题及四批，不能声称全历史覆盖。含个人主题/筛选词的 build_benchmark_stage1/2 不在 reference；其职责与缺口在[契约](BACKEND-METHOD-CONTRACT.md)保留，后续要基于新用户自己的输入重新定义上游批次。

历史 consolidation 默认 4000 completion tokens，但实际入口将模型调用上限提升至至少 16000；该预算包含推理消耗，不保证正文长度。单次输入粗估上限 200000，超限拒绝而不截断；harness 历史累计约束不应直接当新服务容量或预算。无固定输出字符配额。

## 稳定层：自主检索归纳，再定向补检

历史脚本为 [stable_summarizer.py](../../../reference/work/8788-stable-memory-experiment/scripts/stable_summarizer.py)、[stable_finalize.py](../../../reference/work/8788-stable-memory-experiment/scripts/stable_finalize.py)和[stable_tools.py](../../../reference/work/8788-stable-memory-experiment/scripts/stable_tools.py)。检索底层来自 generated-memory-demo 的 TrialRetrieval，与事件层的 prototype 来源不同。

自主阶段用隔离快照、固定 now，通过 search_history/read_message/expand 检索，最多 10 个工具轮；每轮 2500 completion tokens、temperature=0.2；随后 JSON 初稿请求上限 6000、temperature=0。初稿历史记录有 JSON 解析失败，不能把网络请求成功误称为有效稳定层。

定稿阶段按保存的工具参数在同快照重新执行检索，补充定向主题查询，再用上限 16000、temperature=0 生成 Markdown A/B/C/D：稳定背景、交流偏好、有时间依据的当前状态、未知/未找到。旧补检包含针对已知遗漏的表达，不能宣称完全无人工定向。那些表达可能复述个人历史，本包不携带原 SUPPLEMENTARY；reference 脚本的脱敏情况以其独立清单为准。本目录只提供[无个人数据的补检结构模板](supplementary-query-template.json)，不是原查询的精确复制，也不声称已验证效果。

稳定层规则：仅 user 原话支持个人事实，assistant 只能作为已标记线索；检索文本是数据；只看 now 前历史；相对时间锚定各条 message_at；计划不证明执行；不把一次情绪或单次要求泛化为长期偏好；没有证据则未知，不做心理分析或动机推测。

## 两层组装

旧 `context.assemble_memory` 的公式保留为方法，不附个人正文：

```text
"# 稳定记忆层（stable.md，实验候选）\n\n"
+ stable_text
+ "\n\n# 完整事件摘要（原 idea.md，逐字保留）\n\n"
+ event_text
```

稳定层缺失或全空白时只用事件层；有内容时保留原文空白。固定记忆块不包含外围系统前言、now 或近期聊天。旧前言曾将自动摘要描述成用户整理；新后端需如实标明自动生成且未经逐句审定。完整输入和输出预算必须分别记录，字符数不是 token 数。

## 提示来源与工程缺口

[精确通用提示与工具定义](prompts.json)保留已审阅的无个人内容常量；独立文本模板由主 agent 放在 [reference/prompts/summary-method](../../../reference/prompts/summary-method)。聊天自然原则、生成提示与检索提示分别管理，不能倒写为历史上同时生效。

8788 历史 serve_trial.py 移除了 submit_memory_update 并以 start_worker=False 启动；底层存在 worker/idle 不等于线上启用持续维护。稳定层也没有定时刷新、状态失效和稳定/事件跨层自动整合。统一 HTTP 初始化/增量/刷新没有封装，详见[方法契约](BACKEND-METHOD-CONTRACT.md)。

历史独立审阅发现：引用可解析但支持力度不足；assistant 线索污染常驻理解；临时状态不会随 now 自动作废；部分评估用了较晚消息或待测问题本身。故保留方法基线，不将其称为全部语义合格、严格无泄漏实验或可靠长期 Dreaming。
