# 旧维护引擎源码依赖

本目录仅为 continuity-trial-v2 保留现有功能：repair-resolve-v26 包含 v2.6 引擎、三阶段提示词及 CLI；review-v2 包含 citation-alias 与实时观察适配；vendor/lettaproto 包含必要 HTTP 客户端源码；subscription-sol/driver_guards.py 只保留原始 input_metrics 纯函数，避免导入历史个人 runner。

这不是 v1 选定摘要方法的实现，也没有接通 8788 写入。没有历史状态、摘要、账本、真实评测与凭据。CLI 的 --dry-run 只验证批次并写指定输出，不创建维护状态、不调用模型；正常运行需要显式提供凭据。

使用说明见 [运行与前端接入](../../docs/运行与前端接入.md)。
