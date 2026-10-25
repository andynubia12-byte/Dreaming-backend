# v1 摘要写入与聊天接入

本接口采用 8788 的事件层追加与稳定层独立整理方法。运行代码位于 `continuity.memory_v1` 和工作台接线模块，使用现有 Responses 客户端、模型池与 run/SSE。参考目录保留历史实现，不是运行依赖。

## 启用与数据范围

新建会话时显式选择：

```js
const boot = await fetch('/workbench/bootstrap').then(r => r.json());
const created = await fetch('/workbench/sessions', {
  method: 'POST', headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({
    model: boot.profiles.find(p => p.id === boot.defaults.model_id),
    memory: {mode: 'none', enabled: true},
    retrieval: {enabled: true},
    summary: {engine: 'v1', enabled: false}
  })
}).then(r => r.json());
const sid = created.session.id;
```


`model` 使用模型池返回的完整 profile 对象。`summary.profile` 可以另选摘要模型及思考强度；不填时沿用工作台摘要默认模型。主聊天、检索、摘要各自配置，不随更换主模型而暗中切换。

- `memory.mode=none` 表示新 workspace 从空记忆开始，不导入旧 v65/v66 或任意未分层文本；不代表以后永远不注入生成的记忆。
- `memory.enabled` 控制聊天是否注入记忆全文。
- `summary.engine=v1` 决定使用本方法；旧引擎的显式值为 `v2.6`。缺省引擎的旧客户端保留旧行为，不可假定自动变成 v1。
- `summary.enabled` 只控制成功聊天后是否自动追加本轮用户消息。关闭后，已生成记忆仍可读取/注入，也可显式提交整理任务。
- workspace 创建时冻结服务绑定的原文视图，后续本 workspace 的聊天会加入自己的来源库。不能在请求里提交数据库路径或让模型凭空接受“事实正文”。本项目没有附带个人历史；要初始化自己的历史，先明确导入到绑定来源，再新建 workspace。
- `continue_from` 开新会话时共用同一 workspace 的长期版本，近期消息按新会话隔离；普通新建 workspace 则完全独立。不能在共享 workspace 中途混换 v1/旧引擎。

## 读取记忆和版本

| 方法与路径 | 用途 |
|---|---|
| `GET /workbench/sessions/{sid}/memory/v1` | 完整事件层、稳定层、组合全文与当前版本 |
| `GET /workbench/sessions/{sid}/memory/v1/versions` | 完整版本列表（包括未发布的版本 0） |
| `GET /workbench/sessions/{sid}/memory/v1/versions/{number}` | 某个历史版本全文与元数据 |
| `GET /workbench/sessions/{sid}/memory` | 与既有聊天一致的组合全文视图 |

版本 0 代表尚未生成；发布版本中 `event_text`、`stable_text`、`text` 是三种明确视图，`text` 是完整注入内容，不是再次压缩的第三层。读取旧版本不会自动回滚当前版本。

## 提交任务

`POST /workbench/sessions/{sid}/memory/v1/jobs`

共同字段：`request_id`（UUID）、`operation`、`expected_version`（读取到的当前版本）。`now` 可选，必须是带时区且不在未来的时间；省略时由服务在受理时固定一次。所有来源必须早于相应截止时间，且属于这个 workspace 可用的用户消息。

### 按时间初始化事件层

```json
{
  "request_id": "00000000-0000-4000-8000-000000000001",
  "operation": "initialize",
  "expected_version": 0,
  "now": "2026-10-25T12:00:00+08:00",
  "batches": [
    {"now": "2026-10-24T12:00:00+08:00", "source_ids": ["msg:your-conversation#your-node"]},
    {"now": "2026-10-25T12:00:00+08:00", "source_ids": ["chat#1"]}
  ]
}
```

日期和来源 ID 都是结构示例，必须替换为自己的真实输入。仅未发布的 workspace 可以初始化；批次按时间推进，每批使用前批累积的完整事件层。所有批次成功后一次提交，后面某批失败时不会留下半份“初始化成功”记忆。初始化事件层与生成稳定层是两个明确操作，之后提交 `refresh_stable`。

### 追加新增用户来源

```json
{
  "request_id": "00000000-0000-4000-8000-000000000002",
  "operation": "append",
  "expected_version": 1,
  "source_ids": ["chat#3"]
}
```

只增加有来源的新段落，不让模型重写整个事件层。纠正作为带时间的新记录表达；历史经历不无声删除。已经处理的来源不会重复追加。稳定层保持原文。

### 刷新稳定层

```json
{
  "request_id": "00000000-0000-4000-8000-000000000003",
  "operation": "refresh_stable",
  "expected_version": 2
}
```

独立摘要模型使用本地 `search_history`、`read_message`、`expand` 检索当前截止时间之前的原文，再整理稳定背景、明确交流偏好、有日期的状态和未知。事件层保持原文。来源存在与角色检查不等于自动证明每一句自然语言结论正确，仍需真实使用评估；没有额外引入一套 v2.6 提议/核验/分类引擎。

## 实时过程与结果

提交返回 `202` 和 `run_id`。订阅现有 `GET /workbench/runs/{run_id}/events`，用 `Last-Event-ID` 或 `?after=` 重连。任务过程包含实际工具输入、返回和整理阶段；不公开内部思考正文或凭据。

- `202` 仅表示任务受理，不代表已经发布。
- 成功发布、无新增内容（`noop`）、失败和进程中断分别记录；以终态及版本号为准。
- `GET /workbench/runs/{run_id}` 查询状态和结果；`/wire` 查看经过脱敏的实际 Responses 请求/返回记录。
- 同一请求 UUID 与同一输入复用原 run；同 UUID 换内容冲突。重复发送不是失败重试。
- 服务重启不自动发送收费推理。检查 run 和当前版本后，如仍需重试，用新 UUID 和最新的 `expected_version` 显式提交；保留旧 run 记录。

### 具体事件名称

| phase | 展示内容 |
|---|---|
| `event_generation` | 当前批次、截止时间、使用的用户来源 ID |
| `stable_retrieval` | 当前稳定层检索轮次 |
| `stable_tool_call` | 工具名与实际参数 |
| `stable_tool_result` | 工具实际返回数据 |
| `stable_synthesis` | 正在综合稳定层 |
| `publication_started` / `publication_complete` | 版本提交开始及真实结果 |
| `provider_call` | 脱敏后的 Responses 请求、返回与用量 |
| `maintenance_published` / `maintenance_noop` / `maintenance_failed` | 后维护结果，分别展示 |
| `run_complete` | 任务结束，读取 status/result/error |

`run.kind="memory_v1"` 区分显式记忆任务；`run.result` 保存结果，`run.error/error_type` 保存安全错误。自动聊天后追加的结果在聊天 run 的 `summary` 中。提交显式任务时原文复制到该 run 的只读快照；输入/now保存，避免重连重新取一个时间。

## 发布和失败边界

提交后的产物或事件记录异常不会将已发布版本改报失败；附加记录问题见 `run.summary_warnings`，以实际版本与结果为准。

事件层、稳定层、组合全文、来源和当前版本在一个 SQLite 事务中发布。生成期间仍读旧版本。`expected_version` 在提交时再次检查，避免慢任务覆盖较新记忆；冲突不能靠自动覆盖解决。

用户来源校验失败、超预算、输出截断、解析/结构错误或模型请求失败均不覆盖旧版。预算按完整输入检查，不能静默削掉历史。初始化大批任务失败后需要显式重试，当前不承诺保存各批模型输出作自动断点续跑。

聊天回答保存和记忆更新是不同结果：回答已完成后即使摘要失败，也保留已保存的回答，并显示维护失败。事件自动追加可打开，稳定层刷新仍由显式任务触发，不暗中每轮全部重写。

本轮范围是本地工作台中的 v1 写入；不扩展新前端、跨设备同步、多用户鉴权或自动任务取消。旧批量后维护不得将 `engine=v1` 默默转为旧算法；具体支持范围以当前接口说明为准。


## 与历史脚本的具体差别

通用事件提示、稳定层系统/定稿提示、检索工具定义保留原文。没有移植个人主题筛选、个人表达补检或旧数据。新实现默认最多 10 轮稳定层工具调用，再独立综合；显式 read/expand 返回完整原文，超过预算即失败，不静默裁去证据。

新增严格来源、角色、时间、输出结构与引用检查，替代旧脚本可能将空/截断结果写出的行为。新客户端走 Responses，保留工具原生续传项；输出预算至少 16k，低于该值的 profile 会明确失败。没有固定摘要字数目标。

初始化核心可按批次时间排序，HTTP 要求提交的批次已经按时间递增以便审计；全部成功才形成一个版本。事件请求默认 temperature=0，检索阶段0.2、稳定层定稿0；profile 中显式配置的参数仍由统一适配器优先处理。这是接入实现的参数选择，不能称作历史实验字节复现或相同采样条件。

事件的明确 noop 会记录已处理来源而不创建空记忆版本；相同操作 ID 的成功结果可重放。失败任务不会自动增加预算或换模型。来源引用有效仅是机械门槛，事实支持、遗漏与跨用户效果需要后续真实使用确认。
