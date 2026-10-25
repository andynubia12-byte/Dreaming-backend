# v1 保留的独立记忆检索 Agent

> 本文是 2026-10-25 整理的脱离个人数据的新开发资料副本；来源定版于 2026-10-23。未携带个人记忆正文、原始对话或证据附件。历史执行描述不构成本包的运行或验收证明。



状态：2026-10-23 按用户决定原样存档，逻辑标识 `memory-retrieval-agent/2.0`。这是检索组件本身的旧工程编号，不是被弃用的 v65/v66 摘要版本；产品基线名称统一为 v1。

## 入口与职责

应用调用 `agents.memory_agent_search(config, conn, published, query_text, *, instance, client, max_rounds, now_epoch, delegation, exclude_chat_ids, conversation_id)`。主聊天通过原生 `delegate_memory` 工具委派，不把XML/DSML伪调用当正文。`search_memory`保留为旧别名。主模型与检索模型使用独立profile/上下文，等待实际返回再继续回答。默认产品检索角色为DeepSeek Flash，协议沿用当前统一Responses客户端；提供商/model id可配置。

委派包含当前问题、具体缺口gaps、keywords、entities、time_hint、known；系统补同一请求now、当前会话近期片段和与查询相关的记忆摘录。检索Agent不直接给用户写答案，不直接写长期记忆。

## 检索链路

1. `chat.py` 提供delegate_memory工具定义并调agents入口。
2. `agents.py` 组装独立委派，默认最多4个检索工具轮。
3. `memory_agent.run` 自主选择search_history、read_message、expand；读取导入的选中分支历史与本地chat_log。初始搜索采用词法/分词排序与时间过滤，不是向量库。
4. 严格只读本轮now之前的历史，排除当前新写user行。区分user_fact、assistant_context、inference、state；相对时间按原消息发送时间，计划过期不等于完成。
5. 预留独立综合轮，输出found/partial/not_found、findings、证据source_ids、gaps、可继续展开项。引用ID由本地验证并物化原文，不能靠模型给出的ID就认定事实受支持。
6. `chat._retrieval_payload` 将最终同一份证据交回主模型；UI事件/trace保存委派、工具参数/结果和handoff。检索状态不能提前当作成功。

精确提示词、工具schema、最终JSON模板在 `prompts-and-tools.json`，源函数和预算以目标根 `work/continuity-trial-v2/src/continuity` 的当前工程副本为准；历史对照位于 `reference/work/memory-retrieval-v2`，详见[源码映射](../summary-method/源码映射.md)。源码保留现有截断标记、证据数量/文本预算和终态防护；不会因改名v1自动放宽。

## 返回与来源约定

- `msg:<conversation_id>#<message_id>`：导入历史来源；`chat#<id>`：当前数据域消息行，ID只在绑定的数据域内有意义。
- found/partial必须有经解析的证据；not_found不允许夹带没有来源的肯定结论。
- 原文、用户事实、旧助手内容和Agent推断分开；出处可解析不等于措辞一定被原文支持。
- 失败、截断、协议伪调用不会当作有效工具执行或完整回答。

## 接入必须保留的条件

新聊天工具必须把检索工具实际传给主模型，工具开关应可独立配置。测试只注入摘要不等同测试了检索：历史固定摘要对照批次没有开放检索工具，不能据此比较模型主动检索意愿。当前检索组件另有历史端到端验证，不能把这96条计入检索验证。

需要同时保存：历史数据库schema和选中分支/角色/时间；来源ID的数据域映射；当前会话隔离；该请求唯一真实now；实际发出的工具定义与返回；后台维护和检索各自的用量。不要把执行时的模型内部思考明文作为可展示产物，存档脱敏需标记无法精确回放。

## 副本边界

本目录只带通用提示与方法说明，不带 source 源码树、历史库、检索结果、wire 或个人来源 sidecar。主 agent 提供目标根 `work/continuity-trial-v2` 与 `reference/work/memory-retrieval-v2`；二者不应被混为同一时间点的逐字源码。

Published 是全文/版本适配对象，不决定摘要算法；新工具应接自己经 8788 方法形成的完整文本。不得因为读取器或数据库包含旧维护字段，就自动开启 v2.6。具体 payload 数量/裁剪上限、Python 入口和 HTTP 行为见[接口说明](../interfaces/接口说明.md)。
