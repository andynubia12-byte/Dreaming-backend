# 历史导入与本地存储

> 本文是 2026-10-25 整理的脱离个人数据的新开发资料副本；来源定版于 2026-10-23。未携带个人记忆正文、原始对话或证据附件。历史执行描述不构成本包的运行或验收证明。



本页记录现有可复用的数据入口，供新聊天工具复建数据层；本轮没有执行导入、读取原始导出或复制真实数据库。

## 导入链路

`work/history-bootstrap-v1/bootstrap.py build` 只读ChatGPT导出conversations分片，生成独立retrieval_import.sqlite、manifest.json及按时间分组的证据。必须给带时区的as-of，只保留此前消息。沿current_node的父链选中分支，去除system/tool和推理内容，保留用户/助手原文、节点/消息ID、角色、消息时间、分支顺序和附件缺口；缺时间的内容单列，不编造时间。原导出不写入。CLI与文本归一化规则见目标根 `reference/work/history-bootstrap-v1` 的 bootstrap.py/hb_core.py；本目录不复制源码树。

`work/continuity-trial-v2/import_history.py` → `continuity.history.import_history` 将独立库只读导入目标聊天库的messages/conversations。必须提供可信外部预期source SHA（直接参数或独立清单）；还会核对manifest规模、重复ID和字段冲突。成功的数据与导入元数据在同一事务；相同源哈希幂等，冲突回滚；失败元数据另存。原项目已有历史；新包不携带该历史。不要因阅读文档去重导原项目真实库，新前端应导入新用户自己授权的数据。

不执行的调用模板：

```sh
# 在既有应用目录执行；路径仅为占位，目标必须是新工具自有数据库
python3 import_history.py --source-db /path/to/retrieval_import.sqlite \
  --manifest /path/to/manifest.json \
  --expected-source-sha256 TRUSTED_SOURCE_SHA256 \
  --db /path/to/new-tool/chat.sqlite
```

源history.bootstrap-v1文件名与本产品v1版本号无直接关系，不表示采用被弃的v65/v66摘要。bootstrap生成供旧结构化维护使用的batch格式只是历史产物；新v1摘要应按summary-method定义另行适配，不能自动调用v2.6引擎。

## 数据边界

- messages/conversations：导入历史，保留角色、选中分支、原文及时间，供检索只读使用。
- chat_log：新工具自己的交流，按instance/conversation_id隔离，不将历史重新伪装成刚发生的聊天。
- history_imports：源绑定、范围、数量、导入状态；原文导入范围不等于摘要已处理范围。旧coverage.py中与v26发布物关联的摘要覆盖判断不是v1新发布协议，移植时应独立接8788摘要版本，不据旧marker声称全部记住。
- maint_queue：当前demo的旧维护队列表，schema为追溯保留；不因为复用chat数据库就默认启用旧维护。

[current-schema.sql](current-schema.sql)从当前db.py的SCHEMA静态提取；history.py仍含增量列兼容逻辑，迁移不能只执行DDL就忽略该模块。DDL 早期注释中历史表为空的说法已在本副本改正：它不能描述原服务后续状态，也不能表示本包携带历史数据。

原8788的idea版本/任务记录采用prototype ideastore/worker的数据结构，见summary-method方法契约；新工具应保存用户/工作区、两层文本、版本、源ID、前版差异与任务状态，而不是把旧v26 claims数据库当作v1必需格式。

当前工程与reference的源码映射说明现有字段，本目录仅保存文档与DDL，不宣称新服务已经启动。新用户数据放新用户自有库，不复制本人的真实记忆成模板。


## v1 写入存储更新（2026-10-25）

新引擎另用 workspace 内的 `memory-v1.sqlite`，不写旧维护 manifest。`owner` 绑定用户/instance；`versions` 保存事件层、稳定层、组合全文和来源；`current` 指向当前版本；`operations` 保存操作输入指纹、成功/noop结果及已处理来源。

版本发布和当前指针更新在一个 SQLite 事务中完成，提交时再次比较 expected_version，网络生成期间不持有写事务。来源数据库由工作台绑定/快照提供，memory_v1 引擎只读它；不因同名 source_id 接受任意用户传来的文本。版本0用于空状态检查。

run、请求、事件和脱敏wire沿用工作台持久记录；它们与记忆事务不是一个跨文件事务。重启恢复根据已保存的操作/版本判断，未提交任务不自动付费重发。HTTP语义见[摘要写入API](../interfaces/摘要写入API.md)。
