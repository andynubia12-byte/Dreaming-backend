# 新后端重实现参考：8788 摘要方法契约（工程资料副本）

> 2026-10-25 更新：本文继续保留历史方法契约和旧缺陷，新实现已经接入统一写入接口。当前行为见[摘要写入 API](../interfaces/摘要写入API.md)，不得将本文旧实现的缺口覆盖新接口状态。


> 本文是 2026-10-25 整理的脱离个人数据的新开发资料副本；来源定版于 2026-10-23。未携带个人记忆正文、原始对话或证据附件。历史执行描述不构成本包的运行或验收证明。



本文件说明历史方法的真实接口与缺口，不是已实现的新 API、部署声明或完整后端设计。新聊天工具可据此复建摘要模块；当前 retrieval Agent 的版本与接口由主控另行归档。下述“方法接口”是语言/传输无关的文档表示，不指定 HTTP 路由。历史脚本会写绝对旧路径并可能调用模型，不可为阅读存档而执行。

## 1. 阶段和旧函数映射

源码由主 agent 放在目标根的 `reference/work/<原 work 子目录>/...`。下表来源列省略共同前缀 `reference/work/`；未复制的上游构建脚本会明确说明，不提供失效链接。

| 方法职责 | 实际旧函数/常量 | 来源路径 |
|---|---|---|
| 挑选历史/分批 | stage1 main、WINDOW_LO/HI、BATCHES、THEMES | memory-route-comparison/custom/build_benchmark_stage1.py（含个人筛选，不带入；仅保留职责说明） |
| 历史初始化入口/隔离配置 | main、setup_db、make_config、consolidate | memory-route-comparison/custom/run_experiment.py |
| completion 下限适配 | FIX_MIN_COMPLETION、instrument_client.LoggedClient.chat | 同上 |
| 排队、认领、处理、重试/恢复 | MemoryWorker.enqueue / _claim_next / process_task / _process_task / run_once / recover | letta-prototype/src/lettaproto/worker.py |
| 来源核对/输入/追加 | verify_source_ids / build_consolidate_input / memory_agent_consolidate、DATA_GUARD、CONSOLIDATE_SYSTEM | letta-prototype/src/lettaproto/agents.py |
| 文本与版本保存 | load / save / _write_atomic / current_version | letta-prototype/src/lettaproto/ideastore.py |
| JSON/预算/请求失败 | DeepSeekClient.ask_json / chat / _check_budget / effective_max_requests、extract_json | letta-prototype/src/lettaproto/deepseek.py |
| 估算输入量 | approx_tokens | letta-prototype/src/lettaproto/util.py |
| 稳定层自主检索归纳 | main / dispatch / _resolve_ids、SYSTEM / TOOLS / FINAL / MAX_TOOL_TURNS | 8788-stable-memory-experiment/scripts/stable_summarizer.py |
| 稳定层回放补检/定稿 | main / to_epoch、FINAL / SUPPLEMENTARY / CIT_RE | 8788-stable-memory-experiment/scripts/stable_finalize.py |
| 定稿共用规则/工具调度 | SYSTEM / TOOLS / dispatch | 8788-stable-memory-experiment/scripts/stable_tools.py |
| 稳定层工具实际后端 | TrialRetrieval._load / search / read / expand / _resolve | generated-memory-demo/src/lettaproto/retrieval.py |
| 两层原文组装 | load_stable_layer / assemble_memory、STABLE_LAYER_HEADER / EVENT_SUMMARY_HEADER | generated-memory-demo/src/lettaproto/context.py |
| 注入外围系统消息 | build_context / ContextBundle.system_prompt | 同上 |
| 旧在线关闭证据 | 顶层移除工具；serve(...start_worker=False) | generated-memory-demo/serve_trial.py |

`run_experiment.py` 是 idea 的真实导入证据；stable 脚本则显式把 generated 的 src 加到 sys.path。不互换两个目录的 agents/prompts。源码是冻结时当前副本，不具备历史代码字节的独立证明。

## 2. 文档接口：initialize_event_history

输入：

```text
instance: 独立工作记忆实例（历史值 run）
seed_text: "# 长期记忆\n"
batches: 按历史时间排序的 [{batch_id, now_iso, message_ids}]
message_index: message_id -> {source_ids, message_at, role}
source_reader: 可按 source_id 读取原始记录的隔离只读历史视图
budget: input_estimate_ceiling=200000, completion_limit=16000
```

真实输入需要原始 user 文本，档案不包含原始聊天库与 benchmark 的整份消息正文。原包的批次元数据与 source sidecar 未带入；本包不提供真实来源 ID、角色、时刻或 checkpoint 哈希。未来获授权重建时自行提供 source_reader；本包不提供 sidecar，不能据本资料重生成相同个人摘要。模型输出也不保证确定性；原样本已排除，本包不承诺其字节复现。

历史顺序：

1. 旧 setup_db 将 prototype canonical.sqlite 用 SQLite backup 复制到 custom/run 隔离库（本次未做），配置独立 memory/usage 路径。
2. cold 检查该 instance 无既有版本，save 空种子；历史 cold 入口还会清理隔离实例状态。reference 中只保留经主 agent 整理的参考脚本，个人路径/输入可能已脱敏，不保证与原运行逐字相同；本轮未执行。
3. b1→b4：展开各条 message_ids 对应的全部 source_ids；设置 worker.now_shanghai 返回批末 now；enqueue(kind=suggestion, spec={statement:None}, dedup_key="bench:<batch_id>:fixed")。
4. worker 查原记录并验证 role=user，输入既有完整记忆和本批 evidence；调用 consolidation，追加一段，再追加 `<!-- task:<id> -->` marker；保存版本。
5. 保存当批 idea 为 checkpoint；下一批读取此前累积完整文本，非重新从空开始。
6. 最终 b4 原文作为 8788 idea/seed。四批的规模和历史 now 见 README；这只是人工选择的三个主题，不是全历史覆盖声明。

输出：每批 `{batch_id, now_iso, checkpoint_text, checkpoint_sha256, chars}`；最后完整 event 文本。版本号不等于批次数：种子、内容更新、marker 都可分别 save。不带入历史 checkpoint 或其哈希；这些输出须由新用户自己的证据生成。

## 3. 文档接口：append_event_batch

输入/输出结构对应旧 Python 函数，不表示新增运行中的 endpoint：

```text
input:
  instance, task_id, source_ids[], statement|null, trigger, now
  current_memory_text (full)
verified evidence[]:
  {id, role:"user", time:message_at|null, text, conversation_id}
model JSON:
  {section_markdown, summary, sources[], supersedes[], confidence}
output on save:
  {instance, version, sha256, chars, path, snapshot,
   diff_stat:{added,removed}, summary, supersedes[], input_token_estimate}
```

`statement` 是聊天模型未验证措辞，只能当建议；事实来自验证过的 user evidence。历史实验 statement=None。输入构造把证据 `time` 写成 `message_at`；now 是段落日期锚，证据自身日期是相对时间锚。输出以 `current.rstrip() + "\n\n" + section.strip() + "\n"` 拼接，这个历史行为不同于冻结快照时的零改写 byte copy。

CONSOLIDATE_SYSTEM 规定 append-only、`### 更新 YYYY-MM-DD`、source id 引用、纠正说明、计划≠完成、过期/覆盖状态、无持久信息可返回空段。不支持自动重写旧段、删除旧事实或跨层 merge；supersedes 只是返回/记录字段。旧实现未做 JSON schema、confidence 或 sources 支持关系的完整校验。

来源验证：`chat#id` 需要相同 instance 且 user；`msg:conversation#node` 查导入 messages，另支持 message_id/node_id 回退。未验证项被排除，若仍有一条有效 user 可继续；全部无效才失败。该函数本身没有 `<now` 或 selected_branch 限制，历史时间范围依赖上游冻结分批。新实现若要强制这些不变量，应视为明确修复项，不可声称旧代码已有。

预算：旧函数 ask_json 默认请求 4000，历史 harness 将 chat 的 max_tokens 提升至至少 16000。16k 是包括推理消耗的 completion 上限，不是正文长度承诺。温度沿客户端默认 0.2。估算输入使用 approx_tokens（CJK 约1 token/字符，其他约4字符/token），核对的是构造后的 user prompt，未精确计入系统提示；超过 200000 则 BudgetExceeded、拒绝截断。历史 harness 总请求上限 400（max_requests 与 trial_max_requests 取较大值）、累计输入上限 20000000；这些是工程账本限制，不是模型最大上下文证明。无固定输出字符目标。

### 失败与提交语义：旧行为应如实保留在文档

| 条件 | 旧行为 | 不能误称为 |
|---|---|---|
| 无可验证 user 来源 | 返回 failed，队列落 failed | 自动补全证据 |
| input/累计 budget 超限 | BudgetExceeded；worker 直接 failed | 静默截断或自动加预算 |
| 空模型正文 | ask_json 抛 RuntimeError，附 finish/completion 诊断 | “没有值得记忆” |
| JSON 不合法 | extract_json 尝试围栏/截断修复，仍失败抛错 | 完整 schema/语义验收 |
| section_markdown 为空/缺失 | consolidate 返回 noop；worker 队列记 saved、nothing durable | 必然得到新摘要 |
| HTTP、超时、解析/其他异常 | worker 增 attempts；小于 worker_max_attempts 时重置 submitted，否则 failed；默认上限3 | 无限重试 |
| 历史 harness 每批轮询 | 最多4次 run_once，saved/failed/None 即停止；会保存当时加载到的 checkpoint | 每个 checkpoint 都由入口严格断言成功 |
| 重复 dedup_key | 返回已有 task/status，不新建任务 | 跨实例安全的内容去重 |
| 重复 task marker 已存在 | saved/already applied；idle 可推进 cursor | 崩溃场景的严格 exactly-once |
| 进程恢复 | processing→submitted，追加 recovered 错误标记 | 完整事务回滚 |

`ideastore.save` 先写版本快照、再临时文件 replace 当前文本、后提交 SQLite 版本行；这不是文件+数据库的单一原子事务。marker 是 consolidation 后第二次 save，若内容已存但 marker 未存时崩溃，恢复重试仍可能重复追加。worker 锁是进程内锁；`_claim_next` 不按 instance 过滤，`run_once(instance)` 的参数不限定认领实例。隔离实例/单 worker 是历史前提，不能夸大为跨进程并发安全。

历史入口没有真正的 `--resume` 参数（错误提示虽然提过该词）；不带 --cold 也会重新遍历全部批次并依赖既有队列/dedup 状态。不能把它包装为经过验证的断点续跑协议。

## 4. 文档接口：build_stable_candidate

输入：隔离快照 reader、instance=prod、固定 now、通用 SYSTEM/TOOLS/FINAL、独立调用账本。旧 now 为快照全体 chat_log 的 MAX(created_at)，不是调用时真实日期，也不是按单题生成的 as-of 记忆。旧快照及元数据未带入，不能在本包重新核对。

顺序：

1. TrialRetrieval(conn,prod,now) 形成冻结可检索视图；导出只取 selected branch、非 assistant reasoning、非空且 create_time<now；chat 取相同 instance 的 user/assistant、created_at<now，过滤工具泄漏正文。
2. 系统提示为通用摘要角色；起始 user 提示带 SIMULATED NOW 并要求覆盖背景/明确偏好/兴趣/关系状态。
3. 最多10轮 chat(max_tokens=2500,temperature=0.2,tools=TOOLS)。模型无工具调用即退出工具循环；有调用则解析参数，执行 dispatch，并把完整工具结果加入下一轮 messages。参数 JSON 解析失败或非字典被替换为空字典，未知工具返回 error，不是独立失败终态。
4. 加 FINAL 后 JSON 请求（max_tokens=6000,temperature=0,response_json=True）；解析得到 stable_layer_markdown、sources（claim/section/source_ids/message_at/status）、unknowns、notes。解析失败记录 _parse_error/_raw；旧代码仍可能写空 stable 文本和空来源，未可靠拒绝无效产物。
5. 保存工具步骤元数据用于重放。原归档曾保存小型轨迹及解析诊断；本资料副本不携带轨迹、模型结果或来源附件。

工具接口与输出：

| 工具 | 输入 | 输出要点/默认与限制 |
|---|---|---|
| search_history | query,keywords[],entities[],time_start?,time_end?,kind?,limit? | results[{source_id,source,role,message_at,score,text_len,snippet,conversation_id}]；matched_total/returned/has_more；默认8，clamp 1..20，BM25+短语增强、时间只破同分 |
| read_message | source_id,max_chars? | found,item{source_id,source,role,message_at,text,truncated,total_chars,…}；默认4000，可显式12000；未命中 found=false |
| expand | source_id,before?,after? | found,items[],n_items,window,has_more_before/after；默认前后2，各最多12，同来源会话，目标一定包含，每条最多4000字符 |

工具会返回 assistant 甚至导出中其他可见角色；“仅 user 支持事实”主要由提示词要求，不是检索硬过滤。搜索是 snippet；需要完整证据时应读取正文。has_more 不代表自主模型一定继续翻查，预算会限制召回。轨迹中的 read returned=0 是旧统计只找 returned/n_items、不识别 read.item 所致，不等于未读到；同条记录 found=true。

## 5. 文档接口：finalize_stable_candidate

输入：同一快照和 now；自主阶段保存的 `steps[{turn,tool,args,…}]`；stable_tools.SYSTEM/dispatch；5 条 SUPPLEMENTARY；定稿 FINAL。不是拿已有 stable 正文润色。

顺序：按 turn 重建 assistant 工具调用，**重新本地执行**检索/read 得到证据（依赖同快照+同检索代码，未冻结所有返回字节）；追加5条定向补检工具结果；加 FINAL；单次函数执行中发一次 chat(max_tokens=16000,temperature=0)。最终要求 Markdown A 稳定背景 / B 交流偏好 / C 带时间状态 / D 未找到或不确定，不要求固定字数。

原 SUPPLEMENTARY 含定向个人表达，未复制；仅保留补检机制与无个人数据的结构模板。工具调用声明 limit=6，但实际本地 dispatch 的原字典没带 limit，因此默认8；这里记录真实行为，未修复源码。

输出：

```text
stable.md = content.strip() + LF
sources.json = {now,model,finish_reason,claims:[{claim:整行,source_ids:去重排序}],n_claims}
finalize_call.json = {finish_reason,messages_chars,markdown_chars,usage}
```

claim sidecar 来自 CIT_RE 对非标题行内引用的扫描，不验证该引用是否出现在本轮工具结果、角色是否 user 或原文是否支持断言。expanded/缩写引用也未在此完整核实；另存的 evidence_resolved 才是来源解析侧证，解析仍不是语义证明。原侧证存在 assistant 来源，不能据来源可解析就宣称全部为已核实的 user 事实；侧证文件不在本包。

失败语义：客户端网络异常会向外抛出，定稿脚本没有队列重试/提交状态机；即使 finish=length 或正文为空也没有主动阻止写出。stable 和 sources 使用分别 write_text，非一组原子发布，不具备稳定版本事务或自动 rollback。原始元数据的 status=experimental-candidate 必须保留。最终文件 finish=stop 是当前证据，不是旧脚本实现了这些成功门槛。

原历史账本中的请求数和成功终态不等于每次生成必然如此；其中也存在 JSON 解析失败。旧 load_config 与环境影响累计预算；本包未复制账本、调用结果、配置或凭据。

## 6. 文档接口：assemble_frozen_memory

输入为 event 原文与可选 stable 原文；输出对应 `{text,event,stable,stable_present,stable_path,assembly}`。旧 load_stable_layer 对缺失或纯空白返回 None；有内容时保留所有原始空白。存在 stable 时按 README 精确公式拼接；无 stable 则 text=event。两层不存在任意 budget 截断，build_context stats 标明 full/truncated=false；字符/token统计不证明模型真实容量。

原文曾核对个人样本的拼接；本资料副本未带入样本或其验证文件，也未执行组装代码。系统外围仍需 PREAMBLE、now 与可变近期对话。

## 7. 在线状态与未来实现边界

旧 worker 有 idle 方法：检查 chat_log 中晚于 update_cursor:<instance> 的 user 记录，排队 idle；处理时读全部新 user 后 consolidate，成功或 noop 推进 cursor。该路径并非本次四批 suggestion 的执行方法；idle evidence 的 time=None，历史方法没有每条 message_at 的同等精度，也不是可靠跨层 Dreaming。

8788 serve_trial.py 移除 submit_memory_update、追加 fixed-memory 告知、start_worker=False，故当前试用方案后台整理关闭。原 worker/config 中保留的 idle 默认 True 与队列函数不代表在线已运行，底层保留写路径也不能当永久禁写保证。本轮不访问运行服务，以上为代码/历史报告证据，不是新现场部署验收。

新后端可以复建本文方法，但如要上线持续更新，必须另行定义事实核实、状态有效期、跨层维护、事务/恢复、预算策略、实例隔离及验收；这些是未完成项。本档案不提供新 HTTP API、不启用更新、不重生成或修正用户真实摘要，也不把后续重实现提前宣称已部署。

## 本副本的出处与排除

本文件保留方法输入/输出、预算、失败、事务和并发缺陷；没有执行历史脚本。原报告、样本和来源附件的排除原因见[来源与删减说明](../来源与删减说明.md)，实际源码/提示路径见[源码映射](源码映射.md)。历史脚本中绝对旧路径仍可能存在，阅读资料不应执行这些入口。
