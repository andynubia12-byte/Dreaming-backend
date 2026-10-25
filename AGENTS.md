# Dreaming-backend 开发约定

先读 README.md、PROGRESS.md、docs/v1/README.md、docs/运行与前端接入.md。

- 产品 v1 = 8788 事件摘要方法 + 稳定层原则 + 当前独立检索 Agent。v65/v66 不再作为后续默认路线。
- v1 写入已接入 `summary.engine="v1"`；缺省旧请求仍连接 v2.6。文档见 docs/v1/interfaces/摘要写入API.md，不允许两种引擎在同一 workspace 混写。
- `work/continuity-trial-v2` 是现有可运行代码；`reference/` 是历史方法参考。不要因目录名带 Letta/v2 而误认产品架构。
- 所有新推理调用统一 Responses。主聊天、检索、摘要角色分别配置，支持关闭思考；不擅自加 Fast。
- 全量摘要、近期会话、检索证据的注入与消息角色以实际请求为准。评测和聊天复用同一路径。
- 本包没有用户历史数据、个人摘要和密钥。不要自动回读原项目或个人导出；真实数据导入须由用户明确指定，原件始终只读。
- iCloud 目标仅存储/同步，Agent 在客户端/本地执行。此处尚未实现原生 Mac/iPhone 客户端及 iCloud。
- 修改范围要小，接口职责清楚。不引入无关框架或复杂抽象。只做必要核心验证，不自动跑大量收费推理或边界矩阵。
- 实施若使用子 Agent，遵循用户偏好 gpt-6.1-sol、高思考、不开 Fast；主 Agent 验收最终结果，不频繁检查中途。
- 完成工作后更新 PROGRESS.md；改变方法、提示或契约时说明差异，不覆盖原始出处。
