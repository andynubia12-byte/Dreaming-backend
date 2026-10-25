# 历史方法与来源代码

这些文件用于理解和移植已选方法，不是默认启动路径。当前可运行的聊天服务在上级 `work/continuity-trial-v2`。不要批量运行本目录的实验脚本；它们原本依赖未随包提供的实验数据或第三方运行时。

| 路径 | 用途 | 当前地位 |
|---|---|---|
| `work/letta-prototype/src/lettaproto` | 自写的来源验证、后台 worker、事件摘要追加、版本保存与早期聊天 | 8788 事件层实际生成依赖；名称带 Letta 不表示主流程使用 Letta |
| `work/memory-route-comparison/custom/run_experiment.py` | 空记忆、逐批时间推进、worker 调度与预算 | 8788 事件摘要生成实验入口；个人 benchmark 数据不在本包 |
| `work/generated-memory-demo` | 旧 8788 聊天、事件/稳定层全文拼装 | 历史部署参考；不是当前启动入口 |
| `work/8788-stable-memory-experiment/scripts` | 稳定层检索、整理与最终格式化 | 采用的方法来源；不是已经接线的日常稳定层刷新服务 |
| `prompts/summary-method` | 源码中的整理与证据约束提示 | 独立于主聊风格提示，保持方法职责清楚 |
| `work/history-bootstrap-v1` | 原始导出读取、分支/角色筛选与检索库构建 | 导入工具源码；没有附带历史数据 |
| `work/memory-retrieval-v2` | 独立检索的早期组件 | 历史参考；新接入复用当前 continuity 内的检索 |
| `work/memory-backtest`、`work/memory-system-backtest` | 早期对照与时间隔离设计 | 部分来源绑定的案例已移除，不能声称原实验可复跑 |
| `work/letta-native-demo` | 8789 原生 Letta 实验接线 | 对照资料，不是 v1 必选依赖；不附带 vendor、安装包或个人 MemFS |
| `llmtest` | 独立批测项目的服务端与界面源代码 | 复用来源；新开发使用当前后端中的集成批测，不另维护第二套服务 |

`source-manifest.json` 列每份文件的来源、当前哈希和整理改动。历史摘要正文、证据摘录、真实问答、原始导出、数据库、安装依赖与缓存均未复制。

历史生成代码保留其算法流程，但早期 API 传输包含 Chat Completions；将方法接入新工具时必须使用当前统一 Responses 适配器。仅供参考的旧脚本也可能需要本机凭据工具/Letta 运行时，并不属于本包启动依赖。

为避免旧个人数据成为默认输入，参考代码中硬编码的个人数据路径已改到本目录下未提供的 `private-input` 或对应本地目录；原始工程文件不变。定向偏好检索保留五类主题，但移除了来自个人交流的具体措辞，不能据此声称逐字复现旧实验。
