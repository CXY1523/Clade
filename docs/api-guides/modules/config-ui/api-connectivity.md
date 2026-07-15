# AI 接入探测 – `/config/test-api` 与 `/config/fetch-models`

> 这里记录设置界面使用的配置探测。Phase 2C-1 只保护这些探测请求，不代表实际 AI、负载均衡、流式或 Embedding 请求已经完成同等加固。

- **测试连接**：`POST /api/config/test-api`
- **获取模型列表**：`POST /api/config/fetch-models`
- **实现**：`backend/app/api/analytics.py#test_api_connection`、`backend/app/api/analytics.py#fetch_models`

## 凭据来源与请求

请求既可以直接提供服务商信息，也可以用 `provider_id` 引用当前已保存的服务商。直接提供的 `base_url`、`api_key`、`provider_type` 优先；缺少的值从已保存服务商补齐。

```json
{
  "provider_id": "default",
  "provider_type": "openai",
  "base_url": "https://api.example.com/v1",
  "api_key": "<API_KEY>"
}
```

无论 Base URL 来自本次请求还是已保存配置，都会经过同一套外部 URL 安全策略。`provider_id` 只负责选择已保存配置，不能绕过安全检查。

测试连接会向受控的 `/models` 端点发出一次 GET；Anthropic 类型改用受控的 `/messages` 端点。获取非 Anthropic 模型列表时，只发出一次 GET：同一个响应先检查状态，只有状态为 200 才限长读取并解析 JSON。非 200 响应不读取上游正文，也不会为了读取 JSON 再发第二个请求。Anthropic 模型列表沿用内置列表，不发模型列表请求。

## 响应

连接成功：

```json
{
  "success": true,
  "message": "连接成功"
}
```

模型列表成功：

```json
{
  "success": true,
  "models": [
    { "id": "example-model", "name": "example-model" }
  ]
}
```

上游返回非 200 时，接口保留兼容的 HTTP 200 业务失败结果，例如：

```json
{
  "success": false,
  "error": "API 返回非成功状态"
}
```

模型列表失败结果还会包含空的 `models` 数组。该结果不包含上游状态正文。URL 策略、DNS、连接、超时、大小或解析失败则使用下表中的 HTTP 状态和结构化 `detail`：

```json
{
  "detail": {
    "code": "outbound_timeout",
    "message": "外部服务请求超时"
  }
}
```

## 状态与错误码

| HTTP | 错误码 | 场景 | 处理建议 |
| ---: | --- | --- | --- |
| 400 | `outbound_url_invalid` | URL 格式、协议、凭据、query、fragment 或端口非法 | 检查 Base URL |
| 400 | `outbound_https_required` | 公网地址使用 HTTP | 改用 HTTPS |
| 400 | `local_ai_disabled` | 回环地址有效，但本地 AI 开关关闭 | 在设置中使用管理员令牌开启 |
| 400 | `private_network_blocked` | 私网、链路本地或其他特殊地址 | 改用允许的公网地址或精确回环地址 |
| 502 | `outbound_dns_failed` | DNS 失败或没有返回地址 | 检查域名与 DNS |
| 502 | `outbound_connect_failed` | 外部服务不可达 | 检查服务状态与地址 |
| 502 | `outbound_response_too_large` | 模型列表正文超过 1 MiB | 缩小服务返回内容 |
| 502 | `outbound_bad_response` | 模型列表响应非预期或无法解析 | 检查服务响应格式 |
| 504 | `outbound_timeout` | DNS、连接、读取或总体操作超时 | 稍后重试或检查服务 |
| 503 | `admin_token_unconfigured` | 修改本地 AI 开关，但服务端未配置管理员令牌 | 启动前设置 `CLADE_ADMIN_TOKEN` |
| 403 | `admin_token_invalid` | 修改本地 AI 开关时令牌缺失或错误 | 重新输入管理员令牌 |

最后两项由 `POST /api/config/ui` 在本地 AI 开关发生变化时返回，列在这里便于设置界面统一处理安全错误。

## 外部 URL 安全边界

- 公网服务必须使用 HTTPS。域名本次解析得到的每一个 IP 都必须是全局可路由地址；只要混入一个私有或特殊地址，整个请求就会被拒绝。
- 默认禁止回环、家庭局域网、链路本地、保留、多播、未指定及其他非全局地址。
- 只有 `allow_local_ai_endpoints=true` 时才例外允许规范化后的 `localhost`、字面量 `127.0.0.1` 和 `::1`。`localhost` 的全部解析结果还必须都是回环地址。`*.localhost`、其他 `127.0.0.0/8` 地址和局域网地址始终不允许。
- 本地例外可以使用 HTTP 或 HTTPS；公网地址仍只能使用 HTTPS。
- 策略批准 DNS 结果后，实际 TCP 连接只使用本次批准的 IP，不再进行第二次不受约束的 DNS 查询；HTTP Host、HTTPS SNI 与证书验证仍使用原始规范主机名。
- 不读取系统代理配置，也不自动跟随 3xx 重定向。探测端点只能是服务端控制的 `/models` 或 `/messages`。
- 连接测试的连接、读取、写入和连接池等待各不超过 5 秒，总截止时间为 10 秒；它只读取响应状态，不读取完整正文。
- 模型列表的连接、写入和连接池等待各不超过 5 秒，单次读取不超过 10 秒，总截止时间为 15 秒。总截止时间包含 DNS、连接、响应头和正文读取。
- 模型列表正文上限精确为 `1_048_576` 字节（1 MiB）。响应声明超限时不读取正文；未声明长度时最多读到上限加 1 字节后拒绝。客户端要求未压缩响应并拒绝响应压缩，避免压缩内容绕过大小边界。

## 敏感信息处理

API Key 和管理员令牌不会出现在公开错误、响应或日志中。错误和日志同样不会包含 Authorization、`x-api-key` 等请求头、带敏感 query 的完整 URL、上游响应正文、解析器或网络库的原始异常文本。

前端通过 `frontend/src/services/api/config.ts` 调用这些接口。API Key 只按请求或服务端已保存配置使用；管理员令牌只用于修改本地 AI 开关，不属于探测请求体。
