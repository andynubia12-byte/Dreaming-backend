# Responses统一-实现与验收-20261022（工程整理版）

> 脱离个人数据的新开发资料副本（2026-10-25）。本文从原报告整理工程结论，非逐字全文副本；删去个人原话、案例过程、模型回答、证据摘录、截图、运行数据及旧校验清单。所有“原报告记录”的结果仅描述历史，不构成本包复验或验收通过证明。

来源：`outputs/Responses统一-实现与验收-20261022.md`（只读原项目相对路径，非本包附件）。

旧运行工作台主聊、检索/综合和摘要传输统一为 Responses，正常聊天与批测复用 context.build_context → chat_agentic → Profile.body/response_input → client。类名保留不代表仍发 Chat Completions。

完整摘要、now、近期在 system，当前问题是原生 user；工具定义统一 function，输出 items 保留在内存，function_call_output 按 call_id 续传。三角色独立 profile/effort；none 明确传，省略不是 none，不支持则报错而非自动降级。

原合成探针记录过供应商拒绝强制 tool_choice 的失败，不能隐藏错误或自动回退；同名 effort 不能称不同厂商等算力。订阅额外编码约束也不表示私有 backend 永久兼容。

旧摘要引擎只换 transport 并未换算法，仍是 v2.6，不是 8788 writer。历史部署/真实调用/测试成功只作为原报告叙述，本包不复制原验收证明或重跑矩阵。

当前路线及完整资料导航见[资料索引](../资料索引.md)。
