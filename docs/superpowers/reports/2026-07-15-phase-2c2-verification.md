# Phase 2C-2 运行时出站安全验证报告

验证日期：2026-07-18（Asia/Shanghai）

分支：`phase-2c-outbound-url-security`

Phase 2C-1 基线：`b52eb91`

代码验证 HEAD：`137e047 test: isolate model router log capture`

## 结论

Phase 2C-2 的聚焦安全、后端全量、插件注册、前端测试、lint、类型检查、生产构建、结构守卫、敏感哨兵和仓库卫生门禁均已从当前代码 HEAD 取得通过证据。

完整检查确认：`ModelRouter` 七个运行时网络入口和 Embedding 的远程批次都委托统一安全运行时客户端；默认、能力覆盖、负载均衡池、环境/构造参数、旧版配置和 UI 刷新来源没有旁路。DNS 批准结果约束实际连接，安全错误不触发服务商切换或假向量降级，响应限制、超时、取消、清理、配置快照、缓存命名空间和脱敏契约均有自动测试覆盖。

验证过程中发现并诚实保留了一次全量后端测试失败。根因是测试日志捕获受收集顺序影响，不是产品网络行为错误；该问题在独立测试提交 `137e047` 中修复并通过独立复审。复跑后的完整门禁为绿色。当前没有未解决的 Critical、Important 或 Minor 发现，Phase 2C 可进入合并考虑，但本任务未合并、推送、发布或上线。

## 验证环境

所有后端命令均从本工作树的 `backend` 目录运行，并使用既有解释器：

```text
E:\my word\https-github-com-pocketfans-clade-tree\.worktrees\phase-2b2-save-path-boundary\backend\.venv\Scripts\python.exe
```

没有创建虚拟环境、安装或升级依赖，也没有运行 `npm install`。前端使用工作树已连接的依赖。

## 全量门禁失败、根因与独立修复

### 首次失败证据

首次当前阶段聚焦安全命令退出 0：

```powershell
& $PYTHON -m pytest app/security/tests app/ai/tests/test_model_router_security.py app/services/system/tests/test_embedding_security.py app/core/tests/test_ai_router_config.py -q
```

结果为 **569 passed、1 skipped、1 warning**，耗时 30.42 秒。

随后首次后端全量命令退出 1：

```powershell
& $PYTHON -m pytest app -q
```

结果为 **1 failed、1058 passed、2 skipped、45 warnings**，耗时 51.74 秒。唯一失败是：

```text
app/ai/tests/test_model_router_security.py::test_legacy_compatibility_methods_do_not_create_or_reset_network_client
```

该测试期望捕获兼容模式切换的 INFO 日志，但 `caplog.text` 为空。失败发生后，Task 9 立即停止后续门禁，没有越界修改产品代码。

### 根因证据

- 失败测试单独运行时通过：1 passed。
- 将它与 `app/api/tests/test_save_path_api.py` 一起收集时可稳定复现；两节点最小复现为 1 failed、1 passed、2 warnings。
- `test_save_path_api.py` 在收集阶段导入 `app.main`；应用日志初始化把命名 logger `app.ai.model_router` 设为 WARNING。
- 原测试只降低 root logger 的捕获级别，没有降低上述命名 logger，因此 INFO 记录在传播前已被过滤。
- 模式值、活动/排队计数、无普通客户端、无安全客户端调用等产品行为断言均正常；故障只属于测试顺序与日志捕获隔离。

### 独立修复与复审

经批准，仅在原测试的 `caplog.at_level` 中明确指定 `app.ai.model_router` 命名 logger，没有改产品代码、放宽断言或改变安全行为。修复单独提交为：

```text
137e047 test: isolate model router log capture
```

独立复审结论为 **APPROVED**，没有 Critical、Important 或 Minor 发现。当前 HEAD 的修复后证据为：

| 验证 | Exit | 新鲜结果 |
| --- | ---: | --- |
| 最小顺序回归组合 | 0 | 27 passed、2 warnings |
| `app/ai/tests/test_model_router_security.py -q` | 0 | 121 passed |
| Phase 2C 聚焦安全命令 | 0 | 569 passed、1 skipped、1 warning，29.39 秒 |
| `python -m pytest app -q` | 0 | 1059 passed、2 skipped、45 warnings，54.25 秒 |
| 修复范围 `git diff --check` | 0 | 无 whitespace error |

两项跳过是既有 Windows symlink 测试。45 条后端警告等于既有上限；聚焦套件的 1 条警告是 Starlette TestClient 与当前 httpx 兼容层的既有弃用提示。没有失败、收集错误或新增资源泄漏警告。

## Fresh-process 插件注册守卫

从 `backend` 目录启动全新 Python 进程，执行计划规定的 `PluginRegistry` / `load_all_plugins` 集合等值断言。命令退出 0，加载和注册集合都精确为以下六项：

```powershell
& $PYTHON -c "from app.services.embedding_plugins import PluginRegistry, load_all_plugins; expected={'ancestry','behavior_strategy','evolution_space','food_web','prompt_optimizer','tile_biome'}; loaded=set(load_all_plugins()); registered=set(PluginRegistry.list_plugins()); assert loaded == expected; assert registered == expected; print(sorted(registered))"
```

```text
ancestry
behavior_strategy
evolution_space
food_web
prompt_optimizer
tile_biome
```

没有缺项或额外插件。

## 前端门禁

以下命令均从 `frontend` 目录运行：

| 命令 | Exit | 新鲜结果 |
| --- | ---: | --- |
| `npm run test:run` | 0 | 12 个 test files 通过；84 tests passed；0 failed |
| `npm run lint` | 0 | 0 errors、162 warnings |
| `npx tsc --noEmit` | 0 | 无输出，类型检查通过 |
| `npm run build` | 0 | TypeScript 与 Vite build 完成；4,399 modules transformed；Vite 14.89 秒 |

lint 的 162 条 warning 精确等于既有 `--max-warnings=162` 上限，其中 1 条被标记为可自动修复；本阶段没有修改这些既有 warning。build 保留一个信息性提示：设置模块同时被动态和静态导入，因此不会被移动到单独 chunk；构建仍正常退出 0。

## 结构、卫生与敏感哨兵

| 命令 | Exit | 新鲜结果 |
| --- | ---: | --- |
| `git diff --check` | 0 | 无 whitespace error；工作副本只提示 Git 下次接触文档时会按配置将 LF 转为 CRLF |
| 禁止直接网络 `rg` 扫描 `model_router.py` 与 `embedding.py` | 1 | 无匹配；exit 1 是 ripgrep 的正常 no-match 结果 |
| 四类 runtime secret sentinel 生产范围 `rg` | 1 | 无匹配；排除 Python/TypeScript 测试文件后，后端、前端和 API 指南均未发现哨兵 |
| `git status --short`（写报告前） | 0 | 只有预期的 connectivity 文档草稿 |

禁止直接网络扫描覆盖 `httpx.post`、普通 `httpx.Client` / `AsyncClient`、`requests`、`aiohttp`、`urllib.request`、`urllib3`、`httpcore` 和 `socket`。测试中的 AST 守卫还覆盖 httpx 导入别名、请求动词以及 urllib 导入/赋值别名。

实际执行的卫生命令为：

```powershell
git diff --check
rg -n "httpx\.(post|Client|AsyncClient)|requests\.|aiohttp|urllib\.request|urllib3|httpcore|(^|[^A-Za-z])socket([^A-Za-z]|$)" backend/app/ai/model_router.py backend/app/services/system/embedding.py
rg -n "runtime-query-sentinel|runtime-header-sentinel|runtime-body-sentinel|runtime-backend-sentinel" backend/app frontend/src docs/api-guides -g '!**/test_*.py' -g '!**/*.test.ts' -g '!**/*.test.tsx'
git status --short
```

敏感哨兵扫描覆盖 runtime query、header、body 和 backend exception 四类哨兵。报告和 API 指南没有记录真实凭据、完整 Google query、完整请求目标、请求/响应正文或底层异常原文。

## 七个 `ModelRouter` 运行时入口

| 入口 | 模式 | 统一安全出口 | 核对结果 |
| --- | --- | --- | --- |
| `invoke` | 同步 JSON | `post_json` | 通过；普通 AI JSON 上限 8 MiB |
| `ainvoke` | 异步 JSON | `apost_json` | 通过；可用性重试每次重新验证同一 provider |
| `astream` | 异步行流 | `astream_lines` | 通过；状态、心跳、文本事件与一次固定错误契约保留 |
| `call_capability` | 旧版同步 JSON | `post_json` | 通过；现有服务商解析与返回形状保留 |
| `acall_capability` | 旧版异步 JSON | `apost_json` | 通过；取消和计数清理保留 |
| `chat` | 异步 JSON | `apost_json` | 通过；正常结果仍返回提取后的纯文本 |
| `astream_capability` | 旧版异步行流 | `astream_lines` | 通过；服务商格式、状态顺序、心跳与清理保留 |

七项结构矩阵确认每个入口只调用预期安全方法，不创建普通网络客户端。`set_keepalive_mode()` 只保留兼容标志和诊断日志，`reset_client()` 是不创建网络连接的异步兼容空操作。

## 服务商、地址与请求目标核对

- 默认 provider、capability override 和已选中的 provider pool 成员都在业务选择后进入同一运行时验证。
- 环境变量/构造参数、UI 持久配置和旧版迁移地址没有来源豁免。
- OpenAI、Anthropic、Google 的同步、异步和流式请求都把无 query/fragment 的 `base_url` 与服务端控制的同源 `request_target` 分开传递。
- Google 模型名按单一路径段编码，key 使用标准 query 编码；测试证明特殊字符不能改变 origin，日志和错误不含完整 target 或 query。
- target 必须以单个 `/` 开头，拒绝绝对 URL、`//`、反斜杠、fragment、空白、控制字符、嵌套 URL 和其他 origin 逃逸形式。
- 拼接保留已验证 base path，不使用会丢弃基础路径的通用 URL join。

## DNS、传输与本地地址策略

- 公网 provider 必须使用 HTTPS；HTTP 公网配置立即以固定错误失败。
- 默认拒绝回环、私网、链路本地、保留、多播、未指定及其他 IANA 特殊地址；mixed DNS answer 整体拒绝。
- 仅当 `allow_local_ai_endpoints=true` 时，允许规范化 `localhost`、字面量 `127.0.0.1` 与字面量 `::1` 的窄回环例外。本地例外可用 HTTP 或 HTTPS；其他 `127/8`、子域 localhost、局域网和转换/映射地址仍拒绝。
- URL 策略完成本次唯一 DNS 验证；同步和异步传输只按批准顺序连接 `resolved_ips`，不执行不受约束的第二次解析，也不加入新 IP。
- TCP 目标使用批准 IP；HTTP Host、TLS SNI 和证书身份继续使用原始规范主机名。origin 的 scheme、host 与 port 均被绑定核对。
- 客户端使用 `trust_env=False`、`follow_redirects=False`，不读取环境代理、不跟随 3xx，也拒绝 Unix socket。

## 响应、超时、重试与清理

### 固定限制

| 类型 | 总大小 | 单行/事件 |
| --- | ---: | ---: |
| 普通 AI JSON | 8 MiB | 不适用 |
| Embedding JSON | 16 MiB | 不适用 |
| AI 行流 | 16 MiB | 1 MiB |

- 请求强制 `Accept-Encoding: identity`；响应声明非 identity 编码时零正文读取并失败。
- 非 2xx 与声明超限的 `Content-Length` 都在正文读取前失败；无长度时最多读到上限加 1 字节。
- 普通响应必须是 JSON 对象，随后再由 provider 解析器验证；行流按原始字节限长并严格 UTF-8 解码。
- connect 10 秒、write 30 秒、pool 10 秒；普通 read 使用当前配置，默认 60 秒；流式 idle 120 秒、total 600 秒。
- URL/HTTPS/地址/DNS、target、status、encoding、size、JSON/schema 错误不重试。只有 `outbound_connect_failed` 和 `outbound_timeout` 能进入入口原本允许的可用性重试。
- 每次网络重试重新执行 URL/DNS 验证，仍针对已选中的同一 provider；安全错误和普通失败都不触发隐式 pool failover。
- 正常、错误、超时、解析失败、调用方提前结束和取消均关闭 response、stream、client 与 pool；`CancelledError` 保持向上传播。

## Embedding 与缓存边界

- 每个远程 chunk 使用 `post_json`，固定 target 为 `/embeddings`，响应上限为 16 MiB。
- 响应 `data` 必须条数准确；每项必须含唯一、完整且范围正确的整数 index；embedding 必须是非空、有限数值列表，布尔值、NaN 和 Infinity 均不接受。
- 上游条目可以乱序完成，但按 index 恢复输入顺序；并发 chunk 也按原输入顺序合并。
- 公开 `embed()` 在开始时捕获一份冻结的运行时配置。provider 身份、base URL、key、model、timeout、本地策略、假向量策略、缓存查询/写入和磁盘元数据在整个调用中使用同一快照。
- 配置写入通过服务级锁一次发布完整新值；运行中的重试不会混用新旧 provider、凭据或 timeout，下一次独立调用才看到新配置。
- 缓存 key 与元数据绑定生成该向量的同一 provider/model 快照，避免运行中刷新后把旧 provider 结果写入新 provider 命名空间。
- 只有连接失败或超时重试耗尽、`require_real=false`，且请求开始快照已启用假向量时才允许降级。
- URL、HTTPS、本地/私网、DNS、状态、编码、大小、JSON 与 schema 错误都立即失败，永不重试成假向量。

## 脱敏契约

公开异常和流式事件只使用固定错误码与固定中文提示。日志仅记录必要的错误码、provider/capability 类别和执行模式，不包含：

- API key、管理员令牌、Authorization 或 provider key；
- 完整 URL、完整 request target、Google query 或查询参数；
- headers、请求 body、prompt 或消息；
- 上游正文、流片段、候选内容或解析失败原文；
- 带敏感值的底层网络、TLS、DNS 或解析异常字符串。

## 完整 Phase 2C-2 范围

在代码验证 HEAD `137e047` 上执行：

```powershell
git log --oneline b52eb91..HEAD
git diff --stat b52eb91..HEAD
git diff --name-only b52eb91..HEAD
```

命令均退出 0。`b52eb91..137e047` 共 **21 个物理提交**，stat 为 **16 files changed、8706 insertions、1011 deletions**。name-only 清单为：

```text
backend/app/ai/model_router.py
backend/app/ai/tests/__init__.py
backend/app/ai/tests/test_model_router_security.py
backend/app/core/ai_router_config.py
backend/app/core/tests/test_ai_router_config.py
backend/app/security/__init__.py
backend/app/security/pinned_transport.py
backend/app/security/runtime_http.py
backend/app/security/safe_http.py
backend/app/security/tests/test_pinned_transport.py
backend/app/security/tests/test_runtime_http.py
backend/app/security/tests/test_safe_http.py
backend/app/services/system/embedding.py
backend/app/services/system/tests/test_embedding_security.py
docs/superpowers/plans/2026-07-15-phase-2c2-runtime-outbound-security.md
docs/superpowers/specs/2026-07-15-phase-2c2-runtime-outbound-security-design.md
```

逐提交 log 已核对，范围从设计、计划、同步/异步 pinned transport、JSON/流式运行时客户端、七入口接入、Embedding 接入及原子配置/cache 修复，到最终测试日志隔离。Task 9 的 connectivity 与本报告是此代码范围之后的纯文档提交。

## 设计核对与交接

新鲜证据没有证明 `docs/superpowers/specs/2026-07-15-phase-2c2-runtime-outbound-security-design.md` 存在错误，因此 Task 9 未修改设计文档。

最终复审检查表全部有实现、测试或结构扫描证据，没有未解决的 Critical、Important 或 Minor 问题。Phase 2C 可以进入合并考虑；本验证未执行 merge、push、publish 或 release。
