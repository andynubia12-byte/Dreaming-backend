/* Local, buildless workbench. Server owns credentials, isolation and execution. */
(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const STORE = 'continuity.workbench.v1';
  const state = {boot:null, profiles:[], memories:[], sessions:[], batches:[], tab:'chat', session:null,
    sessionConfig:null, sessionConfigIncomplete:false, sourceSignature:null, dirty:false, sourceDirty:false, creating:false, sending:false,
    run:null, stream:null, streamTimer:null, streamAttempt:0, draft:null, batch:null, pollTimer:null,
    batchGeneration:0, sessionGeneration:0, selectedModels:new Set(), contexts:[], questionIndex:0,
    contextIndex:0, editing:null, credentials:[], poolBusy:false, connectionBusy:false, connectionTests:new Map(), sessionBusy:new Set(), discoveryGeneration:0, discoveryBusy:false, editorRevision:0, memoryFingerprint:'', batchBusy:false, results:[], resultSignature:'', fileGeneration:0};
  const welcomeTemplate=$('welcome').cloneNode(true);
  let saved = {};
  try { saved = JSON.parse(localStorage.getItem(STORE) || '{}'); } catch { /* Private browser storage may be unavailable. */ }
  if(saved.profileEdits||saved.editedProfiles){delete saved.profileEdits;delete saved.editedProfiles;try{localStorage.setItem(STORE,JSON.stringify(saved));}catch{/* Storage is optional. */}}
  const node = (tag, className, text) => { const el=document.createElement(tag); if(className)el.className=className; if(text!==undefined)el.textContent=String(text); return el; };
  const pretty = value => typeof value==='string' ? value : JSON.stringify(value,null,2);
  const encode = encodeURIComponent;
  const list = value => Array.isArray(value) ? value : [];
  function errorMessage(error) { return error instanceof Error ? error.message : pretty(error); }
  function showError(error) { $('global-error').textContent=errorMessage(error); $('global-error').hidden=false; }
  function clearError() { $('global-error').hidden=true; }
  function guarded(fn) { return async (...args) => { try { clearError(); await fn(...args); } catch(error) { showError(error); } }; }
  async function api(path, body, options={}) {
    const response=await fetch(path,{...options,headers:{Accept:'application/json',...(body===undefined?{}:{'Content-Type':'application/json'}),...options.headers},...(body===undefined?{}:{method:'POST',body:JSON.stringify(body)})});
    let data; try { data=await response.json(); } catch { throw new Error(`本地服务未返回 JSON（HTTP ${response.status}），请检查 workbench 路由。`); }
    if(!response.ok) { const error=new Error(typeof data.error==='string'?data.error:pretty(data.error || data.detail || `请求失败（HTTP ${response.status}）`)); error.status=response.status; throw error; }
    return data;
  }
  function persist() {
    // Allowlist nonsecret settings. Never persist uploaded/pasted text, wire events, keys or messages.
    const data={contextSettings:state.contexts.filter(c=>c.kind!=='text').map(c=>({id:c.id,kind:c.kind,label:c.label,memoryId:c.memoryId,checked:c.checked})),fixedNow:$('fixed-now').value,tab:state.tab,sessionId:state.session?.id || null,batchId:state.batch?.id || null,
      modelId:$('main-model').value,memoryId:$('memory-source').value==='text'?'none':$('memory-source').value,
      retrievalId:$('retrieval-profile').value,inject:$('inject-memory').checked,retrieval:$('retrieval-enabled').checked,
      summary:$('summary-enabled').checked,summaryModel:$('summary-model').value,
      summaryRequests:$('summary-requests').value,
      modelIds:[...state.selectedModels],repeats:$('repeats').value,concurrency:$('concurrency').value,
      maxRequests:$('max-requests').value,trialMaxRequests:$('trial-max-requests').value,maxInputTokens:$('max-input-tokens').value,
      previousId:$('previous-batch').value,run:state.run?{id:state.run.id,requestId:state.run.requestId,sessionId:state.run.sessionId,answerDone:state.run.answerDone}:null};
    try { localStorage.setItem(STORE,JSON.stringify(data)); } catch { /* No persistence required for execution. */ }
  }
  function option(select,value,label) { const el=node('option','',label);el.value=value;select.append(el);return el; }
  function titleOf(profile) { return profile?.label || profile?.friendly_label || profile?.name || profile?.model || profile?.id || '未配置'; }
  function memoryOf(id) { return state.memories.find(m=>m.id===id); }
  function profileOf(id) { return state.profiles.find(p=>p.id===id); }
  function safeProfile(profile, options={}) {
    if(!profile)throw new Error('请选择有效的模型配置。');
    const allowed=['id','model','base_url','api_format','auth','api_key_env','parameters','timeout_seconds','token_limit_field'];
    const result={}; for(const key of allowed)if(profile[key]!==undefined)result[key]=structuredClone(profile[key]);
    result.parameters ||= {};
    validateProfile(result,options);
    return result;
  }
  function validateProfile(p,{allowTemporaryKey=false}={}) {
    if(!/^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$/.test(p.id || ''))throw new Error('模型配置 ID 只能使用 1–80 位字母、数字、点、横线或下划线。');
    if(!p.model?.trim())throw new Error('请填写模型 ID。');
    let url;try{url=new URL(p.base_url);}catch{throw new Error('模型服务地址必须是完整的 HTTP 或 HTTPS 地址。');}
    if(!['http:','https:'].includes(url.protocol)||url.username||url.password||url.search||url.hash)throw new Error('模型服务地址不能包含账号、密码、查询参数或片段；认证请使用环境变量引用。');
    if(p.api_format!=='responses')throw new Error('仅支持 Responses API；新配置不能使用 Chat Completions。');
    const allowedParams=new Set(['temperature','top_p','service_tier','parallel_tool_calls','max_output_tokens','reasoning','text','max_tool_calls','token_limit_field']);
    if(!p.parameters||typeof p.parameters!=='object'||Array.isArray(p.parameters)||Object.keys(p.parameters).some(key=>!allowedParams.has(key)))throw new Error('模型参数包含不支持的字段，请使用服务端提供的合法配置。');
    if(p.auth==='codex_subscription'&&p.api_format!=='responses')throw new Error('GPT 订阅认证需要使用 Responses 格式。');
    if(p.auth!=='codex_subscription'&&!allowTemporaryKey&&!/^[A-Za-z_][A-Za-z0-9_]*$/.test(p.api_key_env||''))throw new Error('请填写有效的服务端环境变量名，例如 DEEPSEEK_API_KEY，或在 API Key 字段填写密钥。');
    if(/fast/i.test(String(p.parameters?.service_tier||'')))throw new Error('本实验不使用 Fast 服务层，请选择常规模型配置。');
  }
  function number(id,label,min=1,max=Number.MAX_SAFE_INTEGER) {
    const text=$(id).value.trim(), value=Number(text);
    if(!text||!Number.isInteger(value)||value<min||value>max)throw new Error(`${label}必须是 ${min}–${max===Number.MAX_SAFE_INTEGER?'安全整数上限':max} 范围内的整数。`);
    return value;
  }
  function summaryConfig() {
    const profile=safeProfile(profileOf($('summary-model').value));
    return {enabled:$('summary-enabled').checked,model:profile.model,profile,
      max_requests:number('summary-requests','摘要额外请求上限',0,1000),max_output_tokens:64000};
  }
  function sourceConfig() {
    const id=$('memory-source').value, enabled=$('inject-memory').checked;
    if(id==='none')return {mode:'none',enabled};
    if(id==='text') { const text=$('memory-text').value;if(!text.trim())throw new Error('请选择“无初始记忆”，或粘贴 / 导入非空记忆文本。');return {mode:'context',text,enabled}; }
    const memory=memoryOf(id); if(!memory)throw new Error('所选记忆来源已不可用，请重新选择。');
    return {mode:isSnapshot(memory)?'snapshot':'context',id,enabled};
  }
  function sourceKey() { return JSON.stringify({id:$('memory-source').value,text:$('memory-source').value==='text'?$('memory-text').value:'',summaryProfile:profileOf($('summary-model').value)?safeProfile(profileOf($('summary-model').value)):null,summaryMaxRequests:Number($('summary-requests').value)}); }
  function isSnapshot(memory) { return !!memory?._snapshot || /snapshot|current|published/.test(memory?.source_kind || '') || state.boot?.snapshots?.some(m=>m.id===memory?.id); }
  function sessionPayload(continuing=false) {
    const payload={model:safeProfile(profileOf($('main-model').value)),memory:sourceConfig(),
      retrieval:{enabled:$('retrieval-enabled').checked,profile:safeProfile(profileOf($('retrieval-profile').value))},summary:summaryConfig()};
    if(continuing&&state.session?.id)payload.continue_from=state.session.id;
    return payload;
  }
  function updateSource() {
    const id=$('memory-source').value, memory=memoryOf(id);
    $('text-source').hidden=id!=='text';
    $('source-meta').textContent=id==='none'?'无初始记忆 · 可从本次聊天逐步积累摘要':id==='text'?`${$('memory-text').value.length.toLocaleString()} 字符 · 全文保留`:memory?`${memory.version!=null?'v'+memory.version+' · ':''}${Number(memory.chars||memory.text?.length||0).toLocaleString()} 字符 · ${isSnapshot(memory)?'隔离快照':'静态文本'}`:'来源不可用';
    updateModelSummary(); updateMatrix();
  }
  function updateModelSummary() {
    const p=profileOf($('main-model').value);$('model-summary').replaceChildren();
    if(p){$('model-summary').append(node('strong','',titleOf(p)),node('div','',`Responses · ${p.auth==='codex_subscription'?'订阅登录':p.api_key_env||'环境变量未配置'}`));
      const cap=p.parameters?.max_output_tokens||p.parameters?.max_completion_tokens||p.parameters?.max_tokens;
      const effort=p.parameters?.reasoning?.effort||p.parameters?.reasoning_effort;
      $('model-summary').append(node('div','',[cap?`输出上限 ${Number(cap).toLocaleString()}`:'',effort?`推理 ${effort}`:''].filter(Boolean).join(' · ')));}
    if(!state.session)$('session-model').textContent=titleOf(p);
  }
  function settingsChanged() {
    if(state.session){state.dirty=true;state.sourceDirty=sourceKey()!==state.sourceSignature;}
    updateSource(); updateScope(); persist();
  }
  function updateScope() {
    const active=!!state.session, dirty=active&&state.dirty;
    $('config-scope').textContent=dirty?(state.sessionConfigIncomplete&&!state.sourceDirty?'服务端未返回完整会话配置；右侧设置将在下次新建会话时应用。':state.sourceDirty?'材料或摘要维护配置改变，下次新建实验使用所选初始材料。':'配置已改变。下次发送将延续当前实验记忆并新建会话。'):active?'当前会话配置。调整后需新建会话应用；材料或摘要模型 / 预算改变将新建实验。':'首次发送创建独立实验；“无初始记忆”也可逐步积累摘要。';
    $('config-scope').classList.toggle('dirty-note',dirty);
    $('apply-settings').hidden=!dirty||state.tab!=='chat';
    $('apply-settings').textContent=state.sourceDirty?'应用配置并新建实验':'应用配置并新建会话';
    $('apply-settings').disabled=state.sending||state.creating||!!state.run&&!state.run.answerDone;
    $('compose-hint').textContent=dirty?(state.sourceDirty?'新实验将清空当前消息并使用所选初始材料':'新会话将清空近期消息，保留本实验已生长记忆'):'⌘ / Ctrl + Enter 发送 · 回答完成后仍可后台整理摘要';
    $('send-message').textContent=dirty?(state.sourceDirty?'新建实验并发送 ↗':'新建会话并发送 ↗'):'发送 ↑';
    $('session-mode').textContent=dirty?(state.sourceDirty?'配置待应用 · 下一条消息使用新实验':'配置待应用 · 下一条消息使用新会话'):active?`会话 ${String(state.session.id).slice(0,8)} · 延续实验记忆`:'新实验 · 配置可调整';
    updateLocks();
  }
  function updateLocks() {
    const busy=state.sending||state.creating||state.uncertain||!!state.run&&!state.run.answerDone;
    $('send-message').disabled=!state.boot||busy;
    $('setup-fields').disabled=state.creating||state.sending||(state.tab==='chat'&&busy);
    $('new-session').disabled=state.creating||state.sending;
    $('new-experiment').disabled=state.creating||state.sending;
    $('plan-batch').disabled=!state.boot||state.batchBusy;
    $('start-batch').disabled=!state.boot||state.batchBusy;
    $('chat-input').disabled=state.creating;
  }
  function selectTab(tab) {
    state.tab=tab;
    for(const name of ['chat','batch']){const selected=name===tab;$(name+'-tab').setAttribute('aria-selected',String(selected));$(name+'-tab').tabIndex=selected?0:-1;$(name+'-panel').hidden=!selected;}
    $('page-label').textContent=tab==='chat'?'聊天试用':'批量对比';updateScope();persist();
  }
  function populateProfiles() {
    for(const id of ['main-model','retrieval-profile','summary-model']){const selected=$(id).value;$(id).replaceChildren();for(const p of state.profiles)option($(id),p.id,titleOf(p));if(profileOf(selected))$(id).value=selected;}
    renderModels();updateModelSummary();
  }
  function renderModels() {
    $('batch-models').replaceChildren();
    for(const p of state.profiles){const card=node('label','model-card'),checkbox=node('input');checkbox.type='checkbox';checkbox.value=p.id;checkbox.checked=state.selectedModels.has(p.id);checkbox.addEventListener('change',()=>{checkbox.checked?state.selectedModels.add(p.id):state.selectedModels.delete(p.id);updateMatrix();persist();});
      const body=node('span');body.append(node('strong','',titleOf(p)),node('small','',`${p.model} · ${p.auth==='codex_subscription'?'订阅':'Responses'}`));
      const edit=node('button','text-button','配置');edit.type='button';edit.addEventListener('click',e=>{e.preventDefault();openModel(p.id,'batch');});card.append(checkbox,body,edit);$('batch-models').append(card);}
  }
  function credentialLabel(credential,tested=false) { const stored=['keychain','persistent'].includes(credential?.kind);return `${credential?.available===true?stored?'Key 已保存到本机钥匙串 / 重启可用':'Key 已配置':credential?.available===false?'Key 未配置':tested?'订阅认证':'订阅未检查'} · ${tested?'已运行连接测试':'连接未测试'}`; }
  function credentialFor(profile) { return profile.credential_status||state.credentials.find(c=>c.id===profile.id)||state.credentials.find(c=>c.id===profile.api_key_env)||(profile.auth==='codex_subscription'?state.credentials.find(c=>c.kind==='codex_subscription'):undefined); }
  function renderMigrations(migrations) {
    const records=list(migrations);
    $('migration-note').hidden=!records.length;
    $('migration-note').textContent=records.length?`已显式迁移 ${records.length} 项旧协议配置为 Responses 运行副本（${records.map(p=>p.role||p.id).join('、')}）。原始历史请求和结果保留原样；新提交的 CC 配置会被拒绝。`:'';
  }
  function renderCredentials() {
    $('credential-status').replaceChildren();
    for(const c of state.credentials)$('credential-status').append(node('div','',`${c.id} · ${credentialLabel(c)}${c.kind?' · '+c.kind:''}`));
    if(!state.credentials.length)$('credential-status').textContent='服务端未提供凭据状态';
  }
  function builtin(profile) { return profile.builtin===true||profile.built_in===true||profile.is_builtin===true; }
  function renderPool() {
    $('pool-list').replaceChildren();$('pool-count').textContent=`${state.profiles.length} 个共享模型`;
    for(const p of state.profiles){
      const card=node('article','pool-card');card.dataset.profileId=p.id;const tested=state.connectionTests.get(p.id),stale=tested&&(tested.stale||tested.signature!==JSON.stringify(p));
      const head=node('div','pool-card-head');head.append(node('h3','',titleOf(p)),node('span','small-badge',builtin(p)?'内置':'自定义'));
      let host;try{host=new URL(p.base_url).host;}catch{host=p.base_url;}
      card.append(head,node('div','pool-model-id',p.model),node('p','help',`${host} · Responses`),node('p','credential-line',`${credentialLabel(credentialFor(p),!!tested?.outcome&&!stale)} · ${p.auth==='codex_subscription'?'订阅登录':p.api_key_env||'未设置环境变量引用'}`));
      const actions=node('div','pool-card-actions');
      const test=node('button','text-button','测试连接');test.type='button';test.dataset.action='test';test.disabled=state.poolBusy||state.connectionBusy;test.onclick=()=>testSavedModel(p.id);actions.append(test);
      if(!builtin(p)){const edit=node('button','text-button','编辑');edit.type='button';edit.dataset.action='edit';edit.disabled=state.poolBusy;edit.onclick=()=>openModel(p.id,'pool');actions.append(edit);}
      const copy=node('button','text-button','复制另存');copy.type='button';copy.dataset.action='copy';copy.disabled=state.poolBusy;copy.onclick=()=>openModel(p.id,'pool',true);actions.append(copy);
      if(!builtin(p)){const remove=node('button','text-button danger','删除');remove.type='button';remove.dataset.action='delete';remove.disabled=state.poolBusy;remove.onclick=()=>deleteModel(p.id);actions.append(remove);}
      card.append(actions);
      if(tested){const status=node('p','notice connection-result');status.dataset.testStatus=stale?'stale':tested.running?'running':tested.outcome?.ok?'success':'error';
        status.textContent=stale?'配置已修改，连接测试已失效。':tested.running?'正在测试连接…':connectionText(tested.outcome);if(status.dataset.testStatus==='error')status.classList.add('error');card.append(status);}
      $('pool-list').append(card);
    }
  }
  function applyPool(data) {
    const main=$('main-model').value,retrieval=$('retrieval-profile').value,summary=$('summary-model').value;
    const previous=JSON.stringify([profileOf(main),profileOf(retrieval),profileOf(summary)]);
    state.profiles=list(data.profiles).map(p=>structuredClone(p));state.credentials=list(data.credential_status);
    for(const [id,tested] of state.connectionTests)if(tested.signature!==JSON.stringify(profileOf(id)))tested.stale=true;
    state.selectedModels=new Set([...state.selectedModels].filter(id=>profileOf(id)));
    populateProfiles();renderCredentials();renderPool();renderMigrations(data.protocol_migrations);updateMatrix();persist();
    if(state.session&&previous!==JSON.stringify([profileOf($('main-model').value),profileOf($('retrieval-profile').value),profileOf($('summary-model').value)]))settingsChanged();
  }
  async function refreshPool() { const data=await api('/workbench/models');applyPool(data); }
  async function openPool() {
    $('pool-status').hidden=true;$('pool-dialog').showModal();
    try{await refreshPool();}catch(error){$('pool-status').textContent=errorMessage(error);$('pool-status').className='notice error';$('pool-status').hidden=false;}
  }
  async function deleteModel(id) {
    if(state.poolBusy)return;state.poolBusy=true;renderPool();$('pool-status').hidden=true;
    try{await api('/workbench/models/'+encode(id),undefined,{method:'DELETE'});await refreshPool();$('pool-status').textContent='模型已从共享池删除。已有实验记录保留。';$('pool-status').className='notice';$('pool-status').hidden=false;}
    catch(error){$('pool-status').textContent=errorMessage(error);$('pool-status').className='notice error';$('pool-status').hidden=false;}
    finally{state.poolBusy=false;renderPool();}
  }
  function uniqueProfileId(base) { let id=base,index=2;while(profileOf(id))id=base+'-'+index++;return id; }
  function syncModelAuth() {
    const subscription=$('profile-auth').value==='codex_subscription';
    $('profile-key-field').hidden=subscription;$('profile-key-help').hidden=subscription;$('profile-env-field').hidden=subscription;
    $('profile-key').disabled=subscription;$('profile-env').disabled=subscription;
    if(subscription)$('profile-key').value='';
  }
  function connectionText(outcome) {
    if(!outcome)return '连接测试未完成。';
    return `${outcome.ok?'基础流式对话通过':outcome.category==='incomplete'?'响应不完整':'连接测试失败'} · HTTP ${outcome.http_status??'未收到'} · ${outcome.protocol_outcome||'未完成'} · ${outcome.category||'local'} · ${outcome.elapsed_seconds??0}s\n${outcome.model||''}${outcome.url?' · '+outcome.url:''}\n${outcome.message||''}${outcome.error?.code?'\ncode: '+outcome.error.code:''}${outcome.error?.param?'\nparam: '+outcome.error.param:''}${outcome.incomplete_details?'\n'+pretty(outcome.incomplete_details):''}${outcome.response_sample?'\n响应样本：'+outcome.response_sample:''}`;
  }
  function invalidateModelTest() {
    ++state.editorRevision;$('model-error').hidden=true;delete $('model-test-result').dataset.fingerprint;
    if(!$('model-test-result').hidden){$('model-test-result').className='notice connection-result';$('model-test-result').dataset.testStatus='stale';$('model-test-result').textContent='配置已修改，连接测试已失效，请重新测试。';}
  }
  function modelFormProfile(temporaryKey) {
    const p=safeProfile({...state.editing.profile,id:$('profile-id').value.trim(),model:$('profile-model').value.trim(),base_url:$('profile-url').value.trim(),api_format:$('profile-format').value,auth:$('profile-auth').value,api_key_env:$('profile-env').value.trim(),parameters:{}},{allowTemporaryKey:!!temporaryKey});
    const old=state.editing.profile;
    p.parameters=structuredClone(old.parameters||{});
    delete p.token_limit_field;for(const key of ['max_tokens','max_completion_tokens','max_output_tokens','token_limit_field','reasoning_effort','reasoning'])delete p.parameters[key];
    p.parameters.max_output_tokens=number('profile-tokens','模型输出 token 上限');
    const temp=$('profile-temperature').value;if(temp!==''){const t=Number(temp);if(!Number.isFinite(t)||t<0||t>2)throw new Error('Temperature 必须在 0–2 之间，或留空使用模型默认值。');p.parameters.temperature=t;}else p.parameters.temperature=null;
    const effort=$('profile-effort').value;const reasoning=structuredClone(old.parameters?.reasoning||{});delete reasoning.effort;if(effort)reasoning.effort=effort;if(Object.keys(reasoning).length)p.parameters.reasoning=reasoning;
    if(p.auth==='codex_subscription')delete p.api_key_env;
    validateProfile(p,{allowTemporaryKey:!!temporaryKey});return p;
  }
  async function testSavedModel(id) {
    if(state.connectionBusy||state.poolBusy)return;
    const tested={signature:JSON.stringify(profileOf(id)),running:true};state.connectionTests.set(id,tested);state.connectionBusy=true;renderPool();
    try{tested.outcome=await api('/workbench/models/test',{model_id:id});}
    catch(error){tested.outcome={ok:false,category:'local',message:errorMessage(error)};}
    finally{tested.running=false;state.connectionBusy=false;renderPool();}
  }
  async function testEditorModel() {
    if(state.connectionBusy||state.poolBusy)return;
    let temporaryKey=$('profile-auth').value==='api_key'?$('profile-key').value.trim():'';
    const revision=state.editorRevision,target=$('model-test-result');
    try{
      const payload={profile:modelFormProfile(temporaryKey)};if(temporaryKey)payload.api_key=temporaryKey;
      state.connectionBusy=true;$('test-model').disabled=true;$('save-model').disabled=true;$('model-error').hidden=true;
      target.hidden=false;target.className='notice connection-result';target.dataset.testStatus='running';target.textContent='正在测试连接…仅发送一次合成文本请求。';
      const outcome=await api('/workbench/models/test',payload);
      if(revision===state.editorRevision){target.textContent=connectionText(outcome);target.dataset.testStatus=outcome.ok?'success':'error';target.classList.toggle('error',!outcome.ok);target.dataset.fingerprint=outcome.profile_fingerprint||'';}
    }catch(error){if(revision===state.editorRevision){target.hidden=false;target.className='notice connection-result error';target.dataset.testStatus='error';target.textContent=temporaryKey?'连接测试未完成；请核对本地服务与模型配置。':errorMessage(error);}}
    finally{temporaryKey='';state.connectionBusy=false;$('test-model').disabled=false;$('save-model').disabled=false;renderPool();}
  }
  function clearDiscovery() {
    ++state.discoveryGeneration;$('discovered-models').replaceChildren();$('discovered-model-select').replaceChildren();option($('discovered-model-select'),'','先获取模型列表');$('discovered-model-select').disabled=true;$('model-discovery-status').textContent='';
  }
  async function discoverModels() {
    if(state.discoveryBusy)return;
    const generation=state.discoveryGeneration;let key=$('profile-auth').value==='api_key'?$('profile-key').value.trim():'';
    try {
      const base_url=$('profile-url').value.trim(),auth=$('profile-auth').value,api_key_env=$('profile-env').value.trim();
      let url;try{url=new URL(base_url);}catch{throw new Error('请先填写完整的服务地址。');}
      if(!['http:','https:'].includes(url.protocol)||url.username||url.password||url.search||url.hash)throw new Error('请填写不含账号、密码、查询参数的 HTTP / HTTPS 服务地址。');
      const original=profileOf(state.editing.id),unchanged=original&&original.base_url===base_url&&original.auth===auth&&(original.api_key_env||'')===api_key_env&&!key;
      const payload=unchanged?{model_id:original.id}:{base_url,auth,...(auth==='api_key'?{...(api_key_env?{api_key_env}:{}),...(key?{api_key:key}:{})}:{})};
      state.discoveryBusy=true;$('discover-models').disabled=true;$('model-discovery-status').textContent='正在获取模型列表…';
      const outcome=await api('/workbench/models/discover',payload);
      if(generation!==state.discoveryGeneration||!$('model-dialog').open)return;
      const models=list(outcome.models).filter(m=>typeof m.id==='string'&&m.id);
      $('discovered-models').replaceChildren();$('discovered-model-select').replaceChildren();option($('discovered-model-select'),'','选择后填入模型 ID');
      for(const model of models){option($('discovered-models'),model.id,model.id);option($('discovered-model-select'),model.id,model.id);}
      $('discovered-model-select').disabled=!outcome.ok||!models.length;
      $('model-discovery-status').textContent=`${outcome.ok?models.length?`获取到 ${models.length} 个模型；请选择，当前模型 ID 保持不变。`:'服务未返回可选模型，可手动填写 ID。':'获取失败：'+(outcome.message||pretty(outcome.error)||'请检查服务配置。')} · HTTP ${outcome.http_status??'未收到'} · ${outcome.elapsed_seconds??0}s`;
    }catch(error){if(generation===state.discoveryGeneration)$('model-discovery-status').textContent=key?'获取失败，请检查服务地址与认证。':errorMessage(error);}
    finally{key='';state.discoveryBusy=false;$('discover-models').disabled=false;}
  }
  function openModel(id,target,copy=false) {
    const source=profileOf(id),duplicate=!!source&&(copy||builtin(source));
    const p=source?structuredClone(source):{id:uniqueProfileId('custom-model'),model:'',base_url:'',api_format:'responses',auth:'api_key',api_key_env:'',parameters:{max_output_tokens:64000,reasoning:{effort:'high'}}};
    if(duplicate){p.id=uniqueProfileId(p.id+'-copy');p.label=titleOf(source)+' · 副本';}
    clearDiscovery();++state.editorRevision;$('model-test-result').hidden=true;delete $('model-test-result').dataset.fingerprint;$('test-model').disabled=state.connectionBusy;$('save-model').disabled=state.connectionBusy;state.editing={id:source&&!duplicate?source.id:null,target,profile:p};
    $('profile-id').value=p.id;$('profile-id').readOnly=!!state.editing.id;$('profile-label').value=p.label||p.friendly_label||'';$('profile-model').value=p.model;$('profile-url').value=p.base_url;
    $('profile-format').value=p.api_format;$('profile-auth').value=p.auth||'api_key';$('profile-env').value=p.api_key_env||'';$('profile-key').value='';syncModelAuth();
    $('profile-tokens').value=p.parameters?.max_output_tokens||p.parameters?.max_completion_tokens||p.parameters?.max_tokens||64000;
    $('profile-temperature').value=p.parameters?.temperature??'';$('profile-effort').value=p.parameters?.reasoning?.effort||p.parameters?.reasoning_effort||'';
    $('model-error').hidden=true;$('model-dialog-title').textContent=duplicate?'复制模型到共享池':source?'编辑共享模型':'新增共享模型';$('model-dialog').showModal();
  }
  async function saveModel(event) {
    event.preventDefault();if(state.poolBusy||state.connectionBusy)return;
    let temporaryKey=$('profile-auth').value==='api_key'?$('profile-key').value.trim():'';let submitted=false;
    try {
      const p=modelFormProfile(temporaryKey);
      if(state.profiles.some(item=>item.id===p.id&&item.id!==state.editing.id))throw new Error('该配置 ID 已存在，请使用不同的 ID。');
      validateProfile(p,{allowTemporaryKey:!!temporaryKey});
      const payload={profile:p,label:$('profile-label').value.trim()||p.model};if(temporaryKey)payload.api_key=temporaryKey;
      state.poolBusy=true;$('save-model').disabled=true;$('profile-key').value='';
      submitted=true;const data=await api('/workbench/models',payload);if(!data.profile?.id)throw new Error('服务端未返回保存后的模型配置。');
      await refreshPool();const id=data.profile.id,target=state.editing.target;
      if(target==='batch')state.selectedModels.add(id);
      if(target==='main'){$('main-model').value=id;settingsChanged();}
      else if(target==='retrieval'){$('retrieval-profile').value=id;settingsChanged();}
      else if(target==='summary'){$('summary-model').value=id;settingsChanged();}
      renderModels();persist();$('model-dialog').close();
      $('pool-status').textContent='已保存到服务端共享模型池，聊天、检索、摘要和批测均可选择。';$('pool-status').className='notice';$('pool-status').hidden=false;
    }catch(error){$('model-error').textContent=temporaryKey&&submitted?'模型保存未完成。请检查端点、接口格式与认证配置；本次填写的 Key 已清空，重新提交时需再次填写。':errorMessage(error);$('model-error').hidden=false;}
    finally{$('profile-key').value='';temporaryKey='';state.poolBusy=false;$('save-model').disabled=false;renderPool();}
  }
  async function readTextFile(file,target,status) {
    if(!file)return;
    if(!/\.(txt|md)$/i.test(file.name))throw new Error('仅支持 .txt 或 .md 文本文件，请重新选择。');
    const generation=++state.fileGeneration;const text=await file.text();if(generation!==state.fileGeneration)return;
    $(target).value=text;$(status).textContent=`${file.name} · 已读取完整 ${text.length.toLocaleString()} 字符`;
    if(target==='memory-text')settingsChanged();
  }
  function renderHistory() {
    for(const [kind,items] of [['session',state.sessions],['batch',state.batches]]){const container=$(kind+'-list');container.replaceChildren();if(!items.length)container.append(node('p','empty-small',kind==='session'?'还没有聊天会话':'还没有批量实验'));
      for(const item of items){const button=node('button','history-item');button.type='button';button.classList.toggle('active',kind==='session'?state.session?.id===item.id:state.batch?.id===item.id);
        button.append(node('strong','',item.title||item.label||item.id),node('small','',[item.status,dateLabel(item.updated_at||item.created_at)].filter(Boolean).join(' · ')));
        button.addEventListener('click',guarded(async()=>{if(kind==='session'){selectTab('chat');await loadSession(item.id);}else{selectTab('batch');await openBatch(item.id);}}));if(kind==='session'){const row=node('div','history-row');row.dataset.sessionId=item.id;row.append(button);sessionActions(row,item);container.append(row);}else container.append(button);}
    }
    const previous=$('previous-batch').value;$('previous-batch').replaceChildren();option($('previous-batch'),'','不选择历史批次');for(const b of state.batches)option($('previous-batch'),b.id,b.title||b.id);if(state.batches.some(b=>b.id===previous))$('previous-batch').value=previous;
  }
  function sessionActions(row,item) {
    const actions=node('div','history-actions');
    const rename=node('button','','重命名'),remove=node('button','danger','删除');rename.type=remove.type='button';rename.dataset.action='rename';remove.dataset.action='delete';
    rename.setAttribute('aria-label',`重命名 ${item.title||item.id}`);remove.setAttribute('aria-label',`删除 ${item.title||item.id}`);
    rename.disabled=remove.disabled=state.sessionBusy.has(item.id);actions.append(rename,remove);row.append(actions);
    const cancel=()=>{row.querySelector('.history-edit,.delete-confirm')?.remove();actions.hidden=false;};
    rename.onclick=()=>{
      actions.hidden=true;const form=node('form','history-edit'),input=node('input');input.value=item.title||item.id;input.required=true;input.setAttribute('aria-label','会话名称');
      const save=node('button','primary','保存'),back=node('button','','取消');save.type='submit';back.type='button';back.onclick=cancel;
      form.append(input,save,back);row.append(form);input.focus();input.select();input.onkeydown=event=>{if(event.key==='Escape'){event.preventDefault();cancel();}};
      form.onsubmit=guarded(async event=>{event.preventDefault();const title=input.value.trim();if(!title||state.sessionBusy.has(item.id))return;
        state.sessionBusy.add(item.id);save.disabled=back.disabled=input.disabled=true;
        try{const data=await api('/workbench/sessions/'+encode(item.id)+'/rename',{title});if(!data.session?.id)throw new Error('服务端未返回更新后的会话。');
          state.sessions=state.sessions.map(s=>s.id===item.id?{...s,...data.session}:s);
          if(state.session?.id===item.id){state.session={...state.session,...data.session};$('chat-title').textContent=state.session.title;}persist();
        }finally{state.sessionBusy.delete(item.id);renderHistory();}
      });
    };
    remove.onclick=()=>{
      actions.hidden=true;const confirm=node('div','delete-confirm'),yes=node('button','primary','确认删除'),back=node('button','','取消');yes.type=back.type='button';yes.dataset.action='confirm-delete';back.onclick=cancel;
      confirm.append(node('p','',`删除“${item.title||item.id}”？共享工作区和记忆仍保留。`),yes,back);row.append(confirm);
      yes.onclick=guarded(async()=>{if(state.sessionBusy.has(item.id))return;state.sessionBusy.add(item.id);yes.disabled=back.disabled=true;
        try{const data=await api('/workbench/sessions/'+encode(item.id),undefined,{method:'DELETE'});if(!data.ok||!data.deleted||data.id!==item.id)throw new Error('服务端尚未确认删除会话。');
          state.sessions=state.sessions.filter(s=>s.id!==item.id);
          if(state.session?.id===item.id){$('chat-input').value='';resetExperiment();}persist();
        }finally{state.sessionBusy.delete(item.id);renderHistory();}
      });
    };
  }
  function dateLabel(value) { if(!value)return '';const date=new Date(typeof value==='number'?(value<1e12?value*1000:value):value);return Number.isNaN(date.getTime())?'':date.toLocaleString('zh-CN',{month:'2-digit',day:'2-digit',hour:'2-digit',minute:'2-digit'}); }
  function rememberMetadata(kind,item) { const key=kind==='session'?'sessions':'batches';state[key]=[item,...state[key].filter(x=>x.id!==item.id)];renderHistory(); }
  async function refreshBootstrap(initial=false) {
    try {
      const boot=await api('/workbench/bootstrap');state.boot=boot;state.sessions=list(boot.sessions);state.batches=list(boot.batches);
      if(initial){state.profiles=list(boot.profiles).map(p=>structuredClone(p));
        const map=new Map();for(const m of list(boot.memories))map.set(m.id,{...m});for(const m of list(boot.snapshots))map.set(m.id,{...map.get(m.id),...m,_snapshot:true});state.memories=[...map.values()];
        const defaults=boot.defaults||{};
        const preferred=saved.modelId||defaults.model_id||state.profiles.find(p=>/pro/i.test(p.model))?.id;
        state.selectedModels=new Set(list(saved.modelIds).filter(id=>profileOf(id)));if(!state.selectedModels.size&&preferred)state.selectedModels.add(preferred);
        populateProfiles();if(profileOf(preferred))$('main-model').value=preferred;
        const retrievalId=saved.retrievalId||(typeof defaults.retrieval_profile==='string'?defaults.retrieval_profile:defaults.retrieval_profile?.id);
        if(profileOf(retrievalId))$('retrieval-profile').value=retrievalId;
        $('memory-source').replaceChildren();option($('memory-source'),'none','无初始记忆 · 从空白开始');
        for(const m of state.memories)option($('memory-source'),m.id,`${m.label||m.id}${m.version!=null?' · v'+m.version:''}`);
        option($('memory-source'),'text','粘贴 / 上传完整文本');
        const memoryId=saved.memoryId||defaults.memory_id;if(memoryId==='none'||memoryOf(memoryId))$('memory-source').value=memoryId;
        for(const [key,id] of [['inject','inject-memory'],['retrieval','retrieval-enabled'],['summary','summary-enabled']])if(typeof saved[key]==='boolean')$(id).checked=saved[key];
        for(const [key,id] of [['summaryRequests','summary-requests'],['repeats','repeats'],['concurrency','concurrency'],['maxRequests','max-requests'],['trialMaxRequests','trial-max-requests'],['maxInputTokens','max-input-tokens']])if(/^\d+$/.test(saved[key]||''))$(id).value=saved[key];
        const summaryId=saved.summaryModel||defaults.summary_profile?.profile?.id||state.profiles.find(p=>p.model==='deepseek-flash')?.id;
        if(profileOf(summaryId))$('summary-model').value=summaryId;
        applyCapabilities(boot.capabilities||{});
        state.contexts=list(saved.contextSettings).filter(c=>['none','selected','snapshot','pipeline','context'].includes(c.kind)&&/^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$/.test(c.id)&&(!c.memoryId||memoryOf(c.memoryId)));
        if(!state.contexts.length)state.contexts=[{id:'no-memory',kind:'none',label:'无记忆基线',checked:true},{id:'selected-memory',kind:'selected',label:'所选记忆 · 跟随右侧配置',checked:true}];
        state.contextIndex=Math.max(0,...state.contexts.map(c=>Number(c.id.match(/-(\d+)$/)?.[1])||0));renderContexts();
      }
      renderHistory();if(initial&&saved.previousId&&state.batches.some(b=>b.id===saved.previousId))$('previous-batch').value=saved.previousId;
      if(initial){state.credentials=list(boot.credential_status);renderCredentials();renderPool();renderMigrations(boot.protocol_migrations);}else applyPool(boot);
      $('server-status').textContent='本地服务已连接';$('server-dot').className='status-dot connected';$('server-clock').textContent=boot.server_time?dateLabel(boot.server_time):'';
      updateSource();updateScope();
    }catch(error){$('server-status').textContent='本地服务未连接';$('server-dot').className='status-dot failed';throw error;}
  }
  function applyCapabilities(caps) {
    const notes=[];
    if(caps.summary===false||caps.summary_maintenance===false){$('summary-enabled').checked=false;$('summary-enabled').disabled=true;notes.push('服务端当前未启用摘要维护。');}
    if(caps.batch_summary===false)notes.push('批量摘要维护暂不可用；启用时计划校验会明确报错。');
    const max=caps.max_concurrency||caps.concurrency_max;if(Number.isInteger(max))$('concurrency').max=String(max);
    $('capability-note').textContent=notes.join(' ')||'摘要由真实 v2.6 引擎维护；批量能力与预算以服务端计划校验为准。';
  }
  function emptyThread() { $('thread').replaceChildren(welcomeTemplate.cloneNode(true));state.draft=null; }
  function addMessage(content,role='assistant') {
    $('welcome')?.remove();const wrap=node('article','message '+(role==='user'?'user':'assistant'));
    const label=node('div','message-label');if(role!=='user')label.append(node('span','avatar','↳'));label.append(node('span','',role==='user'?'你':'模型'));
    const body=node('div','message-body',typeof content==='string'?content:pretty(content??''));wrap.append(label,body);$('thread').append(wrap);return {wrap,body};
  }
  function followThread(force=false) { const thread=$('thread');if(force||thread.scrollHeight-thread.scrollTop-thread.clientHeight<160)thread.scrollTop=thread.scrollHeight; }
  function setPhase(text,kind='') { $('run-phase').textContent=text;$('run-dot').className='status-dot '+kind; }
  function detachStream() { state.stream?.abort();state.stream=null;clearTimeout(state.streamTimer);state.streamTimer=null; }
  function detachRun() { state.uncertain=false;detachStream();state.run=null;state.draft=null;$('resume-stream').hidden=true;$('reconcile-session').hidden=true; }
  function clearRunLog() { $('event-log').replaceChildren();$('event-count').textContent='0'; }
  function resetExperiment() {
    state.sending=false;
    ++state.sessionGeneration;detachRun();state.memoryFingerprint='';state.session=null;state.sessionConfig=null;state.sessionConfigIncomplete=false;state.dirty=false;state.sourceDirty=false;state.sourceSignature=null;
    emptyThread();clearRunLog();$('chat-title').textContent='聊天测试';$('session-caption').textContent='新实验使用所选初始材料。';
    $('memory-preview').textContent='等待新实验产生会话记忆';$('memory-version').textContent='未开始';$('maintenance-status').textContent='新实验将在首次发送时创建。';
    $('view-session-memory').disabled=true;$('refresh-memory').disabled=true;setPhase('新实验就绪 · 首次发送时创建空间');selectTab('chat');updateModelSummary();updateScope();renderHistory();persist();$('chat-input').focus();
  }
  function hydrateSessionConfig(session) {
    const config=session.config||session.configuration||session;
    if(config.model?.id&&profileOf(config.model.id))$('main-model').value=config.model.id;
    const memory=config.memory;
    if(memory&&typeof memory==='object'&&('mode' in memory)){
      const id=memory.id||memory.memory_id;
      if(memory.mode==='none')$('memory-source').value='none';else if(id&&memoryOf(id))$('memory-source').value=id;else if(typeof memory.text==='string'){$('memory-source').value='text';$('memory-text').value=memory.text;}
      if(typeof memory.enabled==='boolean')$('inject-memory').checked=memory.enabled;
    }
    if(config.retrieval){if(typeof config.retrieval.enabled==='boolean')$('retrieval-enabled').checked=config.retrieval.enabled;const p=config.retrieval.profile;if(p?.id&&profileOf(p.id))$('retrieval-profile').value=p.id;}
    if(config.summary){if(typeof config.summary.enabled==='boolean')$('summary-enabled').checked=config.summary.enabled;const summaryId=config.summary.profile?.id||state.profiles.find(p=>p.model===config.summary.model)?.id;if(profileOf(summaryId))$('summary-model').value=summaryId;if(config.summary.max_requests!=null)$('summary-requests').value=config.summary.max_requests;}
    updateSource();
  }
  async function loadSession(id,{resume=true}={}) {
    const generation=++state.sessionGeneration;detachRun();setPhase('读取会话中…','busy');
    const data=await api('/workbench/sessions/'+encode(id));if(generation!==state.sessionGeneration)return;
    state.session=data.session;renderMigrations(data.session.protocol_migrations);state.sessionConfig=structuredClone(data.session.config||data.session);state.dirty=false;state.sourceDirty=false;hydrateSessionConfig(data.session);state.sourceSignature=sourceKey();
    const config=data.session.config||data.session.configuration||data.session;state.sessionConfigIncomplete=!(config.model&&config.memory&&config.retrieval&&config.summary&&profileOf(config.model.id)&&profileOf(config.retrieval.profile?.id));state.dirty=state.sessionConfigIncomplete;
    const currentModel=profileOf(config.model?.id);
    // Corrected model IDs and credentials apply to the next conversation;
    // historical requests keep their original configuration.
    if(currentModel&&(config.model.model!==currentModel.model||config.model.api_key_env!==currentModel.api_key_env))state.dirty=true;
    emptyThread();clearRunLog();for(const m of list(data.messages)){const message=addMessage(m.content??m.text??'',m.role);message.wrap.dataset.messageId=m.id||'';}
    $('chat-title').textContent=state.session.title||'聊天测试';$('session-caption').textContent='新会话延续本实验记忆。';
    $('session-model').textContent=titleOf(state.session.config?.model||state.session.model||profileOf(state.session.model_id));
    $('view-session-memory').disabled=false;$('refresh-memory').disabled=false;if(data.memory)renderMemory(data.memory);else await refreshSessionMemory();
    setPhase('已恢复保存的会话');updateScope();renderHistory();persist();followThread(true);
    const pending=resume&&saved.run?.sessionId===id?saved.run:null;
    if(pending&&!pending.id){state.uncertain=true;setPhase('上次请求未收到运行 ID，请核对服务端会话；不会自动重发','failed');$('reconcile-session').hidden=false;updateLocks();}
    if(pending?.id){state.run={id:pending.id,sessionId:id,seq:0,answerDone:!!pending.answerDone,complete:false};
      if(!pending.answerDone){const messages=list(data.messages);const last=messages[messages.length-1];if(last?.role==='assistant'){const wrap=$('thread').lastElementChild;state.draft={wrap,body:wrap.querySelector('.message-body')};state.draft.body.textContent='';}}
      state.streamAttempt=0;connectRun();updateLocks();}
  }
  async function createSession({continueMemory=false}={}) {
    if(state.creating)throw new Error('会话正在创建，请稍候。');
    const payload=sessionPayload(continueMemory);state.creating=true;updateLocks();
    try {
      const data=await api('/workbench/sessions',payload);if(!data.session?.id)throw new Error('服务端没有返回新会话 ID。');
      detachRun();++state.sessionGeneration;state.session=data.session;state.sessionConfig=structuredClone(payload);state.sessionConfigIncomplete=false;state.dirty=false;state.sourceDirty=false;state.sourceSignature=sourceKey();
      emptyThread();clearRunLog();$('chat-title').textContent=data.session.title||'聊天测试';$('session-caption').textContent=continueMemory?'新会话已创建，延续本实验记忆。':'新实验已创建，使用所选初始材料。';
      $('session-model').textContent=titleOf(profileOf($('main-model').value));$('view-session-memory').disabled=false;$('refresh-memory').disabled=false;
      $('memory-preview').textContent='正在读取本实验记忆…';$('memory-version').textContent='读取中';
      rememberMetadata('session',data.session);setPhase(continueMemory?'已延续实验记忆，开始新会话':'新实验就绪');persist();
      await refreshSessionMemory().catch(error=>{$('maintenance-status').textContent='记忆预览暂不可用：'+errorMessage(error);});
      return data.session;
    }finally{state.creating=false;updateScope();}
  }
  async function newConversation() {
    selectTab('chat');if(!state.session){setPhase('空白会话已就绪 · 首次发送时创建实验');$('chat-input').focus();return;}
    await createSession({continueMemory:!state.sourceDirty});$('chat-input').focus();
  }
  function renderMemory(memory) {
    const text=memory.text??memory.memory?.text??'';const version=memory.version??memory.memory?.version;
    state.memoryFingerprint=JSON.stringify([state.session?.id,version,text]);
    $('memory-preview').textContent=text||'尚无已发布摘要。可让摘要 Agent 从本次聊天逐步积累。';$('memory-version').textContent=version!=null?'v'+version:'空白';
    $('memory-version').title=`${String(text).length.toLocaleString()} 字符`;
    $('maintenance-status').textContent=memory.present===false?'尚无已发布记忆':`${version!=null?'已发布 v'+version:'当前记忆'} · ${String(text).length.toLocaleString()} 字符${(state.sessionConfig?.memory?.enabled??$('inject-memory').checked)===false?' · 当前不注入对话':''}`;
  }
  async function refreshSessionMemory(quiet=false) { const id=state.session?.id;if(!id)return;const response=await api('/workbench/sessions/'+encode(id)+'/memory');const memory=response.memory||response;if(state.session?.id===id&&(!quiet||state.memoryFingerprint!==JSON.stringify([id,memory.version,memory.text||''])))renderMemory(memory); }
  async function previewMemory(session=false) {
    $('memory-dialog-title').textContent=session?'当前实验 · 完整记忆':'初始材料 · 完整预览';$('memory-dialog-meta').textContent='';$('memory-dialog-text').textContent='读取中…';$('memory-dialog').showModal();
    try {let memory;
      if(session){if(!state.session)throw new Error('请先创建会话。');memory=await api('/workbench/sessions/'+encode(state.session.id)+'/memory');memory=memory.memory||memory;}
      else {const id=$('memory-source').value;if(id==='none')memory={text:''};else if(id==='text')memory={text:$('memory-text').value};else {memory=memoryOf(id);if(memory?.text===undefined){$('memory-dialog-meta').textContent=`${memory?.label||id} · 服务端仅返回来源元数据`;$('memory-dialog-text').textContent='当前 bootstrap 未提供该来源正文。所选材料仍可由服务端在实验中加载；创建实验后可查看当前实验的完整记忆。';return;}}}
      $('memory-dialog-meta').textContent=`${memory.version!=null?'v'+memory.version+' · ':''}${String(memory.text||'').length.toLocaleString()} 字符 · 完整正文`;$('memory-dialog-text').textContent=memory.text||'（空文本）';
    }catch(error){$('memory-dialog-text').textContent=errorMessage(error);}
  }
  const phases={request_accepted:'请求已接收',context_loading:'正在装载上下文',context_loaded:'上下文已装载',model_started:'模型请求已发出，等待响应',model_thinking:'模型处理中（仅显示状态与计数）',draft_reset:'草稿已撤回，等待有效回答',tool_call:'主模型调用工具',retrieval_model_started:'检索 Agent 正在规划',retrieval_tool_call:'检索工具调用',retrieval_tool_result:'检索工具返回',retrieval_synthesis_started:'检索 Agent 正在整理证据',retrieval_handoff:'检索 Agent 已交回证据',pro_resuming:'主模型继续回答',model_finished:'本轮模型输出结束',answer_complete:'回答已完成',answer_committed:'回答已保存',request_failed:'回答失败',maintenance_queued:'摘要维护已排队',maintenance_started:'摘要维护已开始',maintenance_processing:'摘要维护进行中',maintenance_stage:'摘要维护阶段更新',maintenance_published:'新摘要已发布',maintenance_done:'摘要维护完成',maintenance_failed:'摘要维护失败',run_complete:'本次执行已结束',replay_gap:'部分事件已过期'};
  function eventDetail(event,target=$('event-log')) {
    const item=node('details','event-item'),payload=event.data||event;
    const kind=/handoff/.test(event.phase)?'handoff':/tool_result/.test(event.phase)?'result':/tool_call/.test(event.phase)?'call':'status';item.dataset.kind=kind;
    const name=payload.name||payload.tool_name||payload.tool?.name||payload.call?.name;
    const summary=node('summary','',(phases[event.phase]||event.phase||'执行事件')+(name?' · '+name:''));
    if(event.seq!==undefined)summary.append(node('span','event-time','#'+event.seq));item.append(summary);
    const content=kind==='call'?payload.arguments??payload.input??payload.tool?.arguments??payload.call?.arguments:kind==='result'?payload.result??payload.output:kind==='handoff'?payload.content??payload.evidence:undefined;
    if(kind!=='status'){
      item.open=true;let readable=content??payload;if(typeof readable==='string'){try{readable=JSON.parse(readable);}catch{/* Plain tool text is valid. */}}
      item.append(node('p','event-data-label',kind==='call'?'实际工具参数':kind==='result'?'实际工具返回':'交接给主模型的证据'),node('pre','event-payload',pretty(readable)));
      const raw=node('details');raw.append(node('summary','','完整事件 JSON'),node('pre','',pretty(event)));item.append(raw);
    }else item.append(node('pre','',pretty(event)));target.append(item);
    if(target===$('event-log'))$('event-count').textContent=String(target.children.length);
    return item;
  }
  function ensureDraft() { if(!state.draft)state.draft=addMessage('','assistant');return state.draft; }
  function handleRunEvent(event,run) {
    if(state.run!==run||state.session?.id!==run.sessionId)return;
    const seq=Number(event.seq);if(Number.isFinite(seq)&&seq>0){if(seq<=run.seq)return;run.seq=seq;}
    const phase=event.phase||event.type, data=event.data&&typeof event.data==='object'?event.data:event;
    if(['heartbeat','connected'].includes(phase))return;
    if(phase==='answer_delta'||phase==='model_delta'){
      if(!run.answerDone){const near=$('thread').scrollHeight-$('thread').scrollTop-$('thread').clientHeight<160;ensureDraft().body.textContent+=String(data.text||'');if(near)followThread(true);setPhase('正在生成回答…','busy');}return;
    }
    if(phase==='model_thinking'){const counts=['tokens','reasoning_tokens','count','chars','thinking_chars','reasoning_chars'].filter(k=>typeof data[k]==='number').map(k=>`${k}: ${data[k]}`).join(' · ');setPhase(phases[phase]+(counts?' · '+counts:''),'busy');return;}
    if(phase==='draft_reset'){if(!run.answerDone&&state.draft)state.draft.body.textContent='';}
    const maintenance=phase?.startsWith('maintenance_');
    if(maintenance){$('maintenance-status').textContent=(phases[phase]||phase)+(data.stage?' · '+data.stage:'');
      if(phase==='maintenance_published'||phase==='maintenance_done'||phase==='maintenance_publish_result'){
        const memory=data.memory||data.published;if(memory&&typeof memory==='object'&&typeof memory.text==='string')renderMemory(memory);
        else refreshSessionMemory().catch(error=>{$('maintenance-status').textContent='摘要状态已变化，读取预览失败：'+errorMessage(error);});
      }
      if(phase==='maintenance_failed')$('maintenance-status').textContent='摘要维护失败：'+pretty(data.error||data.message||'请展开执行过程查看');
    }else if(!run.answerDone)setPhase(phases[phase]||phase||'执行中','busy');
    eventDetail({...event,phase});
    if(phase==='answer_complete'){
      const result=data.result||data;const answer=result.answer??result.content??result.text;
      if(typeof answer==='string')ensureDraft().body.textContent=answer;
      if(state.draft){let meta=state.draft.wrap.querySelector('.message-meta');if(!meta){meta=node('div','message-meta');state.draft.wrap.append(meta);}meta.textContent=metrics(result);}
      run.answerDone=true;setPhase('回答已完成 · 后台维护状态单独显示','connected');updateLocks();persist();followThread();
    }
    if(phase==='request_failed'){
      run.answerDone=true;run.failed=true;setPhase('回答失败：'+pretty(data.error||'请查看执行过程'),'failed');
      if(state.draft&&!state.draft.body.textContent)state.draft.wrap.remove();updateLocks();persist();
    }
    if(phase==='replay_gap'){setPhase('事件回放存在缺口；请读取服务端会话核对，不会自动重发','failed');$('reconcile-session').hidden=false;}
    if(phase==='run_complete'){
      run.complete=true;if(!run.answerDone){setPhase('执行已结束，但未收到完整回答事件；请核对会话','failed');$('reconcile-session').hidden=false;}
      else setPhase(run.failed?'本次请求失败 · 执行结束':'回答已完成 · 本次后台执行结束',run.failed?'failed':'connected');
      detachStream();state.run=null;state.draft=null;saved.run=null;updateLocks();persist();
    }
  }
  async function consumeEvents(response,receive,signal) {
    if(!response.ok){let error;try{error=await response.json();}catch{}throw new Error(error?.error||`事件连接失败（HTTP ${response.status}）`);}
    if(!response.body)throw new Error('浏览器不支持读取事件流。');
    const reader=response.body.getReader(),decoder=new TextDecoder();let buffer='';
    try{while(!signal.aborted){const {value,done}=await reader.read();buffer=(buffer+decoder.decode(value||new Uint8Array(),{stream:!done})).replace(/\r\n/g,'\n');
      let cut;while((cut=buffer.indexOf('\n\n'))>=0){const packet=buffer.slice(0,cut);buffer=buffer.slice(cut+2);const lines=packet.split('\n');const raw=lines.filter(line=>line.startsWith('data:')).map(line=>line.slice(5).replace(/^ /,'')).join('\n');
        if(raw){if(raw==='[DONE]')continue;const event=JSON.parse(raw);const type=lines.find(line=>line.startsWith('event:'))?.slice(6).trim();if(!event.phase&&type)event.phase=type;receive(event);}}
      if(done)break;}
    }finally{reader.releaseLock();}
  }
  async function connectRun() {
    const run=state.run;if(!run?.id)return;detachStream();const controller=new AbortController();state.stream=controller;$('resume-stream').hidden=true;
    if(!run.answerDone)setPhase(`正在连接执行事件${run.seq?' · 从 #'+run.seq+' 继续':''}`,'busy');
    try {
      const response=await fetch('/workbench/runs/'+encode(run.id)+'/events?after='+run.seq,{headers:{Accept:'text/event-stream'},signal:controller.signal});
      await consumeEvents(response,event=>{state.streamAttempt=0;handleRunEvent(event,run);},controller.signal);
      if(!controller.signal.aborted&&state.run===run&&!run.complete)streamDisconnected(run,'事件连接已关闭');
    }catch(error){if(error.name!=='AbortError'&&state.run===run)streamDisconnected(run,errorMessage(error));}
  }
  function streamDisconnected(run,reason) {
    if(state.run!==run)return;state.streamAttempt++;
    setPhase(`${run.failed?'请求已失败，执行事件连接断开':run.answerDone?'回答已完成，后台事件连接断开':'事件连接断开，回答状态待确认'} · ${reason}。仅恢复事件，不重发消息。`,'failed');
    $('resume-stream').hidden=false;$('reconcile-session').hidden=false;
    if(state.streamAttempt<=4)state.streamTimer=setTimeout(()=>{if(state.run===run)connectRun();},Math.min(1000*state.streamAttempt,5000));
  }
  async function sendMessage(event) {
    event.preventDefault();if(state.sending||state.creating||state.uncertain||state.run&&!state.run.answerDone)return;
    const message=$('chat-input').value;if(!message.trim())return;
    state.sending=true;updateLocks();clearError();
    let posted=false;
    try {
      if(!state.session||state.dirty)await createSession({continueMemory:!!state.session&&!state.sourceDirty});
      detachRun();clearRunLog();const sessionId=state.session.id,requestId=crypto.randomUUID();addMessage(message,'user');followThread(true);$('chat-input').value='';
      state.run={id:null,sessionId,requestId,seq:0,answerDone:false,complete:false};persist();setPhase('消息已提交，等待运行 ID…','busy');posted=true;
      const data=await api('/workbench/sessions/'+encode(sessionId)+'/chat',{request_id:requestId,message});
      const id=data.run_id||data.run?.id;if(!id)throw new Error('服务端未返回 run_id；请核对会话，不能自动重发。');
      state.run.id=id;state.streamAttempt=0;persist();connectRun();
    }catch(error){showError(error);
      if(posted){state.uncertain=true;setPhase('请求状态待确认 · 不会自动重发，请读取服务端会话','failed');$('reconcile-session').hidden=false;}
      else setPhase('配置或会话创建失败 · 消息尚未发送','failed');
    }finally{state.sending=false;updateLocks();}
  }
  async function reconcileSession() {
    if(!state.session)return;
    const id=state.session.id;const data=await api('/workbench/sessions/'+encode(id));if(state.session?.id!==id)return;
    const runId=data.session.active_run_id||data.session.run_id||data.active_run?.id||data.run_id||state.run?.id;
    if(state.uncertain&&!runId){emptyThread();for(const m of list(data.messages))addMessage(m.content??m.text??'',m.role);
      setPhase('已读取保存消息。服务端未报告运行 ID；请新建会话继续，避免重复提交','failed');if(data.memory)renderMemory(data.memory);return;}
    if(runId){state.uncertain=false;state.run={id:runId,sessionId:id,seq:0,answerDone:state.run?.answerDone||false,complete:false};
      emptyThread();for(const m of list(data.messages))addMessage(m.content??m.text??'',m.role);
      if(!state.run.answerDone&&list(data.messages).at(-1)?.role==='assistant'){const wrap=$('thread').lastElementChild;state.draft={wrap,body:wrap.querySelector('.message-body')};state.draft.body.textContent='';}
      clearRunLog();state.streamAttempt=0;persist();connectRun();}
    else{state.uncertain=false;await loadSession(id,{resume:false});}updateLocks();
  }
  function addQuestion(text='') {
    const id='q'+(++state.questionIndex),row=node('div','question-row');row.dataset.id=id;
    const head=node('div','question-head'),label=node('label','',`问题 ${state.questionIndex}`);label.htmlFor='question-'+id;
    const remove=node('button','text-button','移除');remove.type='button';remove.setAttribute('aria-label',`移除问题 ${state.questionIndex}`);remove.onclick=()=>{row.remove();updateMatrix();};head.append(label,remove);
    const textarea=node('textarea');textarea.id='question-'+id;textarea.rows=3;textarea.placeholder='输入一道完整问题，支持多行文本…';textarea.value=text;textarea.dataset.question='true';textarea.addEventListener('input',updateMatrix);
    const details=node('details'),summary=node('summary','','检查条件（可选）'),grid=node('div','field-grid');
    for(const [key,title] of [['contains','必须包含（每行一项）'],['not_contains','不得包含（每行一项）']]){const l=node('label','',title),input=node('textarea');input.rows=2;input.dataset.check=key;l.append(input);grid.append(l);}
    details.append(summary,grid);row.append(head,textarea,details);$('questions').append(row);updateMatrix();
  }
  function questions(validate=true) {
    const rows=[...$('questions').children];if(validate&&!rows.length)throw new Error('请至少添加一道实验问题。');
    return rows.map((row,index)=>{const text=row.querySelector('[data-question]').value;if(validate&&!text.trim())throw new Error(`第 ${index+1} 道问题为空，请填写完整问题或移除该卡片。`);
      const result={id:row.dataset.id,text};const checks={};for(const input of row.querySelectorAll('[data-check]')){const values=input.value.split('\n').map(s=>s.trim()).filter(Boolean);if(values.length)checks[input.dataset.check]=values;}if(Object.keys(checks).length)result.checks=checks;return result;});
  }
  function renderContexts() {
    $('batch-contexts').replaceChildren();
    for(const context of state.contexts){const row=node('div','context-row'),label=node('label'),checkbox=node('input');checkbox.type='checkbox';checkbox.checked=context.checked;checkbox.addEventListener('change',()=>{context.checked=checkbox.checked;updateMatrix();persist();});
      const text=node('span','',context.label);text.append(node('small','',contextDescription(context)));label.append(checkbox,text);
      const preview=node('button','text-button','预览');preview.type='button';preview.onclick=guarded(()=>previewContext(context));
      const remove=node('button','remove-button','×');remove.type='button';remove.setAttribute('aria-label','移除上下文 '+context.label);remove.onclick=()=>{state.contexts=state.contexts.filter(c=>c!==context);renderContexts();};row.append(label,preview,remove);$('batch-contexts').append(row);}
    updateMatrix();persist();
  }
  function contextDescription(c) {
    if(c.kind==='none')return 'none · 不带入初始材料';if(c.kind==='selected')return '随注入 / 检索开关 · 创建计划时固定';
    if(c.kind==='text')return `context · ${c.text.length.toLocaleString()} 字符全文`;
    return `${c.kind} · ${memoryOf(c.memoryId)?.label||c.memoryId}${c.kind==='pipeline'?' · 固定检索流水线':''}`;
  }
  async function previewContext(c) {
    if(c.kind==='selected')return previewMemory();
    $('memory-dialog-title').textContent=c.label;$('memory-dialog-meta').textContent=contextDescription(c);
    $('memory-dialog-text').textContent=c.kind==='none'?'无初始材料。后台摘要与注入开关仍独立配置。':c.kind==='text'?c.text:memoryOf(c.memoryId)?.text??'服务端只返回该来源的元数据。创建实验后可查看已加载的会话记忆。';$('memory-dialog').showModal();
  }
  function addContext() {
    const kind=$('context-kind').value;
    if(kind==='text'){$('context-dialog').showModal();return;}
    const id='ctx-'+(++state.contextIndex),source=$('memory-source').value,memory=memoryOf(source);
    if(kind==='none'){state.contexts.push({id,kind,label:'无记忆基线 '+state.contextIndex,checked:true});renderContexts();return;}
    if(kind==='selected'){state.contexts.push({id,kind,label:'所选记忆 · 跟随右侧配置 '+state.contextIndex,checked:true});renderContexts();return;}
    if(source==='none')throw new Error('请先在右侧选择一个记忆来源，再添加上下文变体。');
    if(source==='text'){if(!['static'].includes(kind))throw new Error('文本来源没有历史数据库，只能添加静态注入变体。');if(!$('memory-text').value.trim())throw new Error('请先粘贴或导入非空记忆文本。');state.contexts.push({id,kind:'text',text:$('memory-text').value,label:'全文静态注入 '+state.contextIndex,checked:true});}
    else if(kind==='static'&&!isSnapshot(memory))state.contexts.push({id,kind:'context',memoryId:source,label:(memory.label||source)+' · 静态注入',checked:true});
    else{if(!isSnapshot(memory))throw new Error('所选来源是纯文本；snapshot / pipeline 变体需要服务端提供的历史快照。');state.contexts.push({id,kind:kind==='static'?'snapshot':kind,memoryId:source,label:(memory.label||source)+' · '+(kind==='pipeline'?'检索流水线':'静态快照'),checked:true});}
    renderContexts();
  }
  function buildContexts() {
    const selected=state.contexts.filter(c=>c.checked);if(!selected.length)throw new Error('请至少选择一个对比上下文。');
    const retrieval=$('retrieval-enabled').checked;
    return selected.map(c=>{
      if(c.kind==='none')return {id:c.id,mode:'none'};
      if(c.kind==='selected'){
        if(!$('inject-memory').checked&&!retrieval)return {id:c.id,mode:'none'};
        const m=sourceConfig();if(m.mode==='none')return {id:c.id,mode:'none'};
        if(m.mode==='snapshot')return {id:c.id,mode:retrieval?'pipeline':'snapshot',memory_id:m.id,...(!m.enabled?{inject_memory:false}:{})};
        if(!m.enabled)return {id:c.id,mode:'none'};
        if(m.mode==='context')return m.text!==undefined?{id:c.id,mode:'context',text:m.text}:{id:c.id,mode:'context',memory_id:m.id};
      }
      if(c.kind==='text')return {id:c.id,mode:'context',text:c.text};
      return {id:c.id,mode:c.kind,memory_id:c.memoryId};
    });
  }
  function buildBatchPayload() {
    if(!state.selectedModels.size)throw new Error('请至少选择一个参与批量实验的模型。');
    if($('summary-enabled').checked&&state.boot?.capabilities?.batch_summary===false)throw new Error('本地服务尚不支持每条实验独立摘要维护，请关闭后台摘要 Agent 后运行，或升级服务。');
    const suite={version:1,models:[...state.selectedModels].map(id=>safeProfile(profileOf(id))),questions:questions(),contexts:buildContexts(),
      repeats:number('repeats','重复次数',1,100),concurrency:number('concurrency','并发任务',1,Number($('concurrency').max)||32),
      now_epoch:new Date($('fixed-now').value).getTime()/1000,
      budgets:{max_requests:number('max-requests','总请求预算'),trial_max_requests:number('trial-max-requests','单条请求上限'),max_input_tokens:number('max-input-tokens','输入 token 预算')}};
    if(!Number.isFinite(suite.now_epoch))throw new Error('请选择有效的固定实验时间。');
    if(suite.contexts.some(c=>c.mode==='pipeline'))suite.retrieval_profile=safeProfile(profileOf($('retrieval-profile').value));
    const payload={suite,summary:summaryConfig()};if($('previous-batch').value)payload.previous_id=$('previous-batch').value;return payload;
  }
  function updateMatrix() {
    const models=state.selectedModels.size,qs=$('questions').children.length,contexts=state.contexts.filter(c=>c.checked).length,repeats=Math.max(0,Number($('repeats').value)||0);
    $('plan-count').textContent=(models*qs*contexts*repeats).toLocaleString()+' 条实验';$('matrix-formula').textContent=`${models} 模型 × ${qs} 问题 × ${contexts} 上下文 × ${repeats} 次`;
    $('plan-feedback').hidden=true;
  }
  async function planBatch() {
    if(state.batchBusy)return;const payload=buildBatchPayload();delete payload.previous_id;state.batchBusy=true;updateLocks();
    try{const plan=await api('/workbench/batches/plan',payload);const count=Array.isArray(plan.cases)?plan.cases.length:plan.cases??plan.case_count??'未知';const missing=list(plan.missing_credentials);
      $('plan-feedback').textContent=`${plan.valid===false?'计划未通过':'计划检查完成'} · ${count} 条实验${missing.length?'\n缺少凭据：'+missing.map(c=>typeof c==='string'?c:c.id||pretty(c)).join('、'):''}${plan.errors?'\n'+pretty(plan.errors):''}${plan.warnings?'\n'+pretty(plan.warnings):''}\n只有点击“开始批量实验”才会执行模型请求。`;
      $('plan-feedback').className='plan-feedback'+(plan.valid===false||missing.length?' error':'');$('plan-feedback').hidden=false;
    }finally{state.batchBusy=false;updateLocks();}
  }
  async function startBatch() {
    if(state.batchBusy)return;const payload=buildBatchPayload();state.batchBusy=true;updateLocks();
    try{const data=await api('/workbench/batches',payload);if(!data.batch?.id)throw new Error('服务端没有返回批次 ID。');state.batch=data.batch;rememberMetadata('batch',data.batch);persist();await openBatch(data.batch.id);}finally{state.batchBusy=false;updateLocks();}
  }
  const terminalStatuses=new Set(['completed','complete','done','failed','error','cancelled','canceled','finished','partial','interrupted']);
  async function openBatch(id) {
    clearTimeout(state.pollTimer);const generation=++state.batchGeneration;state.resultSignature='';state.results=[];state.batch={id};$('batch-builder').hidden=true;$('batch-results').hidden=false;$('result-cards').replaceChildren(node('p','empty-small','正在读取实验结果…'));
    $('report-link').href='/workbench/batches/'+encode(id)+'/report';persist();await pollBatch(id,generation);
  }
  async function pollBatch(id,generation) {
    try {const data=await api('/workbench/batches/'+encode(id));if(generation!==state.batchGeneration)return;
      state.batch=data.batch||{id};renderMigrations(state.batch.protocol_migrations);state.results=list(data.results);renderBatch(data);persist();
      if(!terminalStatuses.has(state.batch.status))state.pollTimer=setTimeout(()=>pollBatch(id,generation),1100);
    }catch(error){if(generation!==state.batchGeneration)return;$('batch-status').textContent='结果连接断开：'+errorMessage(error)+'；将重试读取，不重新提交实验。';state.pollTimer=setTimeout(()=>pollBatch(id,generation),3500);}
  }
  function identityOf(result) { return Array.isArray(result.identity)?{model:result.identity[0],question:result.identity[1],context:result.identity[2]}:result.identity||{}; }
  function resultModel(result) { const id=identityOf(result).model||result.model_id||result.parameters?.id;return titleOf(profileOf(id)||result.parameters||result.model)||id; }
  function resultQuestion(result) { return identityOf(result).question||result.question_id||result.question?.id||''; }
  function resultContext(result) { return identityOf(result).context||result.context_id||result.context?.id||result.context_mode||''; }
  function metrics(result) {
    const usage=result.usage||{},input=usage.prompt_tokens??usage.input_tokens,output=usage.completion_tokens??usage.output_tokens,total=usage.total_tokens;
    const duration=result.duration_seconds??result.elapsed_seconds??result.duration;
    return [input!=null?`输入 ${input} tokens`:null,output!=null?`输出 ${output} tokens`:null,total!=null?`总计 ${total} tokens`:null,input==null&&output==null&&total==null?'token 用量未返回':null,duration!=null?`${Number(duration).toFixed(2)} 秒`:null].filter(Boolean).join(' · ');
  }
  function renderBatch(data) {
    const b=state.batch,manifest=data.manifest||{};const cases=Array.isArray(manifest.cases)?manifest.cases.length:manifest.cases;
    const total=b.total??b.total_cases??cases??b.plan?.cases?.length??state.results.length;const complete=state.results.filter(r=>terminalStatuses.has(r.status)||r.answer!==undefined).length;
    $('batch-title').textContent=b.title||`实验 ${b.id}`;$('batch-progress').max=Math.max(Number(total)||1,1);$('batch-progress').value=complete;
    const statusNames={pending:'等待执行',running:'运行中',completed:'已完成',complete:'已完成',done:'已完成',failed:'执行失败',error:'执行失败',cancelled:'已取消',finished:'已结束',partial:'部分完成',interrupted:'已中断'};
    const activeConcurrency=b.runtime_concurrency??manifest.runtime_concurrency;
    $('batch-status').textContent=`${statusNames[b.status]||b.status||'等待状态'}${activeConcurrency?' · 并发 '+activeConcurrency:''} · ${complete} / ${total} 条结果`;
    $('batch-raw').textContent=pretty({batch:b,manifest:data.manifest,events:data.events});
    const regression=data.regression||manifest.regression||b.regression;$('regression-summary').hidden=!regression;if(regression)$('regression-summary').textContent='回归比较 · '+pretty(regression);
    const questionSelect=$('result-question'),selected=questionSelect.value;const ids=[...new Set(state.results.map(resultQuestion).filter(Boolean))];if(JSON.stringify(ids)!==questionSelect.dataset.ids){questionSelect.replaceChildren();option(questionSelect,'','全部问题');for(const id of ids)option(questionSelect,id,id);questionSelect.value=ids.includes(selected)?selected:'';questionSelect.dataset.ids=JSON.stringify(ids);}
    const signature=JSON.stringify(state.results);if(signature!==state.resultSignature){state.resultSignature=signature;renderResults();}
  }
  function renderResults() {
    const filter=$('result-question').value;const results=state.results.filter(r=>!filter||resultQuestion(r)===filter).slice().sort((a,b)=>String(resultQuestion(a)).localeCompare(String(resultQuestion(b)))||Number(a.repeat||0)-Number(b.repeat||0)||Number(a.index||0)-Number(b.index||0));
    $('result-cards').replaceChildren();if(!results.length)$('result-cards').append(node('p','empty-small','尚无已返回结果；运行中的实验会持续更新。'));
    for(const r of results){const card=node('article','result-card '+(['ok','success','completed','done'].includes(r.status)?'success':['error','failed'].includes(r.status)?'failed':''));
      card.append(node('p','eyebrow',`${resultQuestion(r)||r.case_id} · 第 ${r.repeat??1} 次`),node('h3','',resultModel(r)),node('div','result-context',`${resultContext(r)}${r.context_mode?' / '+r.context_mode:''} · ${r.status||'状态未知'}`));
      if(r.question?.text||r.question_text)card.append(node('div','result-question-text',r.question?.text||r.question_text));
      card.append(node('div','result-answer',r.answer??r.result?.answer??(r.error?'执行失败：'+pretty(r.error):'尚未返回回答')),node('div','result-metrics',metrics(r)));
      const inspect=node('button','text-button','查看完整请求 / 工具交接 ↗');inspect.type='button';inspect.onclick=guarded(()=>inspectCase(r.case_id||r.id));card.append(inspect);$('result-cards').append(card);}
  }
  async function inspectCase(caseId) {
    if(!caseId)throw new Error('该结果没有 case_id，无法读取详情。');const id=state.batch.id;$('result-dialog').showModal();$('result-dialog-title').textContent=caseId;$('result-detail').replaceChildren(node('p','help','正在读取真实执行记录…'));
    try{const data=await api('/workbench/batches/'+encode(id)+'/results/'+encode(caseId));if(state.batch?.id!==id)return;const result=data.result||{};$('result-detail').replaceChildren();
      $('result-detail').append(node('p','help',`${resultModel(result)} · ${resultContext(result)} · ${metrics(result)}`),node('pre','full-text',result.answer??''));
      const metadata=node('details'),summary=node('summary','','完整结果与检查条件');metadata.append(summary,node('pre','',pretty(result)));$('result-detail').append(metadata);
      for(const [index,event] of list(data.wire_events).entries()){const details=node('details'),s=node('summary','',`${index+1}. ${event.phase||event.kind||event.type||event.role||'实际请求 / 响应'}`);details.append(s,node('pre','',pretty(event)));$('result-detail').append(details);}
      if(!list(data.wire_events).length)$('result-detail').append(node('p','help','服务端未返回 wire_events。'));
    }catch(error){$('result-detail').replaceChildren(node('p','notice error',errorMessage(error)));}
  }
  function bind() {
    $('chat-tab').onclick=()=>selectTab('chat');$('batch-tab').onclick=()=>selectTab('batch');
    for(const id of ['chat-tab','batch-tab'])$(id).addEventListener('keydown',event=>{if(['ArrowLeft','ArrowRight','ArrowUp','ArrowDown'].includes(event.key)){event.preventDefault();const other=id==='chat-tab'?'batch':'chat';selectTab(other);$(other+'-tab').focus();}});
    $('chat-form').onsubmit=sendMessage;$('chat-input').onkeydown=event=>{if(event.key==='Enter'&&(event.metaKey||event.ctrlKey)&&!event.isComposing){event.preventDefault();$('chat-form').requestSubmit();}};
    $('new-session').onclick=guarded(newConversation);$('new-experiment').onclick=resetExperiment;$('apply-settings').onclick=guarded(newConversation);
    $('reconnect-server').onclick=guarded(()=>refreshBootstrap(!state.boot));$('refresh-history').onclick=guarded(()=>refreshBootstrap());
    $('resume-stream').onclick=()=>{state.streamAttempt=0;connectRun();};$('reconcile-session').onclick=guarded(reconcileSession);
    $('view-session-memory').onclick=()=>previewMemory(true);$('preview-source').onclick=()=>previewMemory();$('refresh-memory').onclick=guarded(refreshSessionMemory);
    $('memory-file').onchange=guarded(event=>readTextFile(event.target.files[0],'memory-text','file-status'));
    $('context-file').onchange=guarded(event=>readTextFile(event.target.files[0],'context-text','context-file-status'));
    $('memory-text').oninput=settingsChanged;
    for(const id of ['main-model','memory-source','inject-memory','retrieval-enabled','summary-enabled','retrieval-profile','summary-model','summary-requests','summary-tokens'])$(id).addEventListener('change',settingsChanged);
    for(const id of ['repeats','concurrency','max-requests','trial-max-requests','max-input-tokens','fixed-now','previous-batch'])$(id).addEventListener('change',()=>{updateMatrix();persist();});
    $('edit-model').onclick=()=>openModel($('main-model').value,'main');$('edit-retrieval').onclick=()=>openModel($('retrieval-profile').value,'retrieval');$('edit-summary').onclick=()=>openModel($('summary-model').value,'summary');$('add-model').onclick=()=>openModel(null,'batch');$('model-form').onsubmit=saveModel;
    $('open-model-pool').onclick=openPool;$('pool-add-model').onclick=()=>openModel(null,'pool');$('refresh-model-pool').onclick=guarded(refreshPool);$('profile-auth').onchange=syncModelAuth;
    $('test-model').onclick=testEditorModel;$('discover-models').onclick=discoverModels;
    $('discovered-model-select').onchange=()=>{if($('discovered-model-select').value){$('profile-model').value=$('discovered-model-select').value;invalidateModelTest();}};
    for(const id of ['profile-url','profile-auth','profile-env','profile-key'])for(const event of ['input','change'])$(id).addEventListener(event,clearDiscovery);
    $('model-form').addEventListener('input',invalidateModelTest);$('model-form').addEventListener('change',invalidateModelTest);
    $('model-dialog').addEventListener('close',()=>{$('profile-key').value='';clearDiscovery();++state.editorRevision;});
    $('add-question').onclick=()=>{addQuestion();$('questions').lastElementChild.querySelector('textarea').focus();};$('add-context').onclick=guarded(addContext);
    $('context-form').onsubmit=event=>{event.preventDefault();const text=$('context-text').value;if(!text.trim())return;state.contexts.push({id:'text-'+(++state.contextIndex),kind:'text',label:$('context-label').value.trim(),text,checked:true});renderContexts();$('context-dialog').close();$('context-form').reset();};
    $('plan-batch').onclick=guarded(planBatch);$('start-batch').onclick=guarded(startBatch);$('result-question').onchange=renderResults;
    $('refresh-batch').onclick=guarded(async()=>{if(state.batch){clearTimeout(state.pollTimer);await pollBatch(state.batch.id,++state.batchGeneration);}});
    $('new-batch').onclick=()=>{clearTimeout(state.pollTimer);++state.batchGeneration;$('batch-builder').hidden=false;$('batch-results').hidden=true;state.batch=null;persist();};
    $('toggle-history').onclick=()=>{const expanded=$('toggle-history').getAttribute('aria-expanded')!=='true';$('toggle-history').setAttribute('aria-expanded',String(expanded));document.querySelector('.history-scroll').classList.toggle('mobile-open',expanded);};
    for(const button of document.querySelectorAll('[data-close]'))button.onclick=()=>$(button.dataset.close).close();
  }
  async function init() {
    bind();const now=new Date(),local=new Date(now.getTime()-now.getTimezoneOffset()*60000);$('fixed-now').value=saved.fixedNow&&Number.isFinite(new Date(saved.fixedNow).getTime())?saved.fixedNow:local.toISOString().slice(0,19);addQuestion();
    setInterval(()=>{if(state.session&&!state.creating)refreshSessionMemory(true).catch(()=>{});},5000);
    try {await refreshBootstrap(true);
      const params=new URLSearchParams(location.search),sessionId=params.get('session')||saved.sessionId,batchId=params.get('batch')||saved.batchId;
      if(sessionId)await loadSession(sessionId).catch(showError);
      if(batchId)await openBatch(batchId).catch(showError);
      selectTab(params.has('batch')||saved.tab==='batch'?'batch':'chat');
    }catch(error){showError(error);setPhase('请先启动本地 workbench 服务');}
  }
  init();
})();
