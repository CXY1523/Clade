"""通用流式调用助手 - 提供带心跳检测的 LLM 调用封装

所有 LLM 模块都应使用此模块来确保：
1. 流式传输心跳检测（防止前端超时）
2. 智能空闲超时（只有真正卡住才超时）
3. 统一的错误处理和日志
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)


async def stream_call_with_heartbeat(
    router: Any,
    capability: str,
    messages: list[dict[str, str]],
    response_format: dict | None = None,
    task_name: str = "AI处理",
    idle_timeout: float = 60.0,
    heartbeat_interval: float = 1.5,
    event_callback: Callable[[str, str, str], None] | None = None,
) -> str:
    """通用流式调用 - 带智能心跳和空闲超时
    
    智能超时机制：
    - 只要持续收到 chunk，就不会触发超时
    - 只有在 idle_timeout 秒内没有收到任何 chunk 才触发超时
    - 这样即使 AI 思考+输出很长时间，只要在输出就不会超时
    
    Args:
        router: ModelRouter 实例
        capability: AI 能力名称
        messages: 消息列表
        response_format: 响应格式（如 {"type": "json_object"}）
        task_name: 任务名称（用于心跳消息和日志）
        idle_timeout: 空闲超时秒数（两个chunk之间的最大间隔）
        heartbeat_interval: 心跳发送间隔秒数
        event_callback: 事件回调函数 (event_type, message, category)
        
    Returns:
        完整的 AI 响应内容
        
    Raises:
        asyncio.TimeoutError: 空闲超时
        Exception: 其他调用错误
    """
    chunks: list[str] = []
    chunk_count = 0
    last_heartbeat_time = asyncio.get_event_loop().time()
    last_chunk_time = asyncio.get_event_loop().time()
    
    def emit_event(event_type: str, message: str, **kwargs):
        """发送事件（如果有回调）"""
        if event_callback:
            try:
                event_callback(event_type, message, "AI")
            except Exception as e:
                logger.debug(f"[流式心跳] 事件回调失败: {e}")
    
    async def iter_with_idle_timeout():
        """带空闲超时的迭代器包装"""
        nonlocal last_chunk_time
        
        async for item in router.astream_capability(
            capability=capability,
            messages=messages,
            response_format=response_format,
        ):
            last_chunk_time = asyncio.get_event_loop().time()
            yield item
    
    try:
        stream_iter = iter_with_idle_timeout()
        
        while True:
            try:
                # 计算剩余空闲超时时间
                elapsed_idle = asyncio.get_event_loop().time() - last_chunk_time
                remaining_timeout = max(1.0, idle_timeout - elapsed_idle)
                
                # 尝试获取下一个 item，带超时保护
                item = await asyncio.wait_for(
                    stream_iter.__anext__(),
                    timeout=remaining_timeout
                )
                
            except StopAsyncIteration:
                break
            except asyncio.TimeoutError:
                emit_event(
                    "ai_idle_timeout",
                    f"⏰ {task_name} 空闲超时 ({idle_timeout:.0f}s无输出)"
                )
                logger.warning(
                    f"[流式调用] {task_name} 空闲超时 "
                    f"(已收到{chunk_count}个chunks, 空闲{idle_timeout}秒)"
                )
                # 如果已经收到一些内容，尝试返回
                if chunks:
                    logger.info(f"[流式调用] {task_name} 使用已接收的部分内容 ({len(''.join(chunks))} chars)")
                    break
                raise asyncio.TimeoutError(f"空闲超时: {idle_timeout}秒内无输出")
            
            # 处理状态事件
            if isinstance(item, dict):
                state = item.get("state", "")
                if state == "connected":
                    emit_event("ai_stream_start", f"🔗 {task_name} 已连接")
                elif state == "receiving":
                    emit_event("ai_stream_receiving", f"📥 {task_name} 正在接收...")
                elif state == "completed":
                    emit_event("ai_stream_complete", f"✅ {task_name} 接收完成")
                elif item.get("type") == "error":
                    error_msg = item.get("message", "未知错误")
                    emit_event("ai_stream_error", f"❌ {task_name} 错误: {error_msg}")
            else:
                # 这是文本 chunk
                chunks.append(str(item))
                chunk_count += 1
                
                # 发送 chunk 心跳（限制频率）
                current_time = asyncio.get_event_loop().time()
                if current_time - last_heartbeat_time >= heartbeat_interval:
                    emit_event(
                        "ai_chunk_heartbeat",
                        f"💓 {task_name} 输出中 ({chunk_count} chunks)"
                    )
                    last_heartbeat_time = current_time
        
        full_content = "".join(chunks)
        logger.debug(f"[流式调用] {task_name} 完成，共 {chunk_count} chunks, 总长度 {len(full_content)}")
        return full_content
        
    except Exception as e:
        logger.error(f"[流式调用] {task_name} 失败: {e}")
        emit_event("ai_stream_error", f"❌ {task_name} 流式调用失败: {e}")
        raise


async def stream_invoke_with_heartbeat(
    router: Any,
    capability: str,
    payload: dict,
    task_name: str = "AI处理",
    idle_timeout: float = 90.0,
    heartbeat_interval: float = 2.0,
    event_callback: Callable[[str, str, str], None] | None = None,
) -> dict:
    """流式调用的心跳封装 - 使用 astream (推荐)
    
    智能超时机制：
    - 只要持续收到 chunk，就不会触发超时
    - 只有在 idle_timeout 秒内没有收到任何 chunk 才触发超时
    - 这样即使 AI 思考+输出很长时间，只要在输出就不会超时
    
    Args:
        router: ModelRouter 实例
        capability: AI 能力名称
        payload: 请求载荷（会传给 prompt 模板）
        task_name: 任务名称
        idle_timeout: 空闲超时秒数（两个chunk之间的最大间隔）
        heartbeat_interval: 心跳发送间隔秒数
        event_callback: 事件回调函数
        
    Returns:
        AI 响应字典 {"content": parsed_json_or_text}
    """
    import json
    
    chunks: list[str] = []
    chunk_count = 0
    last_heartbeat_time = asyncio.get_event_loop().time()
    last_chunk_time = asyncio.get_event_loop().time()
    
    def emit_event(event_type: str, message: str):
        if event_callback:
            try:
                event_callback(event_type, message, "AI")
            except Exception:
                pass
    
    emit_event("ai_stream_start", f"🚀 {task_name} 开始流式请求")
    
    async def iter_with_idle_timeout():
        """带空闲超时的迭代器包装"""
        nonlocal last_chunk_time
        
        async for item in router.astream(capability, payload):
            last_chunk_time = asyncio.get_event_loop().time()
            yield item
    
    try:
        stream_iter = iter_with_idle_timeout()
        
        while True:
            try:
                # 计算剩余空闲超时时间
                elapsed_idle = asyncio.get_event_loop().time() - last_chunk_time
                remaining_timeout = max(1.0, idle_timeout - elapsed_idle)
                
                item = await asyncio.wait_for(
                    stream_iter.__anext__(),
                    timeout=remaining_timeout
                )
                
            except StopAsyncIteration:
                break
            except asyncio.TimeoutError:
                emit_event("ai_idle_timeout", f"⏰ {task_name} 空闲超时 ({idle_timeout:.0f}s无输出)")
                logger.warning(f"[流式调用] {task_name} 空闲超时 (已收到{chunk_count}个chunks, 空闲{idle_timeout}秒)")
                if chunks:
                    logger.info(f"[流式调用] {task_name} 使用已接收的部分内容 ({len(''.join(chunks))} chars)")
                    break
                raise asyncio.TimeoutError(f"空闲超时: {idle_timeout}秒内无输出")
            
            # 处理状态事件
            if isinstance(item, dict):
                state = item.get("state", "")
                if state == "connected":
                    emit_event("ai_stream_connected", f"🔗 {task_name} 已连接")
                elif state == "receiving":
                    emit_event("ai_stream_receiving", f"📥 {task_name} 正在接收...")
                elif state == "completed":
                    emit_event("ai_stream_complete", f"✅ {task_name} 接收完成")
                elif item.get("type") == "error":
                    error_msg = item.get("message", "未知错误")
                    emit_event("ai_stream_error", f"❌ {task_name} 错误: {error_msg}")
                    raise Exception(f"流式错误: {error_msg}")
            else:
                # 这是文本 chunk
                chunks.append(str(item))
                chunk_count += 1
                
                # 发送 chunk 心跳（限制频率）
                current_time = asyncio.get_event_loop().time()
                if current_time - last_heartbeat_time >= heartbeat_interval:
                    emit_event("ai_chunk_heartbeat", f"💓 {task_name} 输出中 ({chunk_count} chunks)")
                    last_heartbeat_time = current_time
        
        full_content = "".join(chunks)
        logger.debug(f"[流式调用] {task_name} 完成，共 {chunk_count} chunks, 总长度 {len(full_content)}")
        
        # 尝试解析 JSON
        try:
            parsed = json.loads(full_content)
            return {"content": parsed}
        except json.JSONDecodeError:
            # 如果不是 JSON，返回原始文本
            return {"content": full_content}
        
    except Exception as e:
        logger.error(f"[流式调用] {task_name} 失败: {e}")
        emit_event("ai_stream_error", f"❌ {task_name} 流式调用失败: {e}")
        raise


async def invoke_with_heartbeat(
    router: Any,
    capability: str,
    payload: dict,
    task_name: str = "AI处理",
    heartbeat_interval: float = 2.0,
    event_callback: Callable[[str, str, str], None] | None = None,
) -> dict:
    """非流式调用的心跳封装 - 使用 ainvoke (不推荐，容易超时)
    
    ⚠️ 建议使用 stream_invoke_with_heartbeat 替代，它有智能空闲超时机制
    
    对于不支持流式的场景，通过定时心跳来保持连接活跃。
    
    Args:
        router: ModelRouter 实例
        capability: AI 能力名称
        payload: 请求载荷
        task_name: 任务名称
        heartbeat_interval: 心跳发送间隔秒数
        event_callback: 事件回调函数
        
    Returns:
        AI 响应字典
    """
    def emit_event(event_type: str, message: str):
        if event_callback:
            try:
                event_callback(event_type, message, "AI")
            except Exception:
                pass
    
    emit_event("ai_request_start", f"🚀 {task_name} 开始请求")
    
    # 创建心跳任务
    heartbeat_task = None
    heartbeat_count = 0
    
    async def send_heartbeats():
        nonlocal heartbeat_count
        while True:
            await asyncio.sleep(heartbeat_interval)
            heartbeat_count += 1
            emit_event("ai_heartbeat", f"💓 {task_name} 等待中 ({heartbeat_count * heartbeat_interval:.0f}s)")
    
    try:
        # 启动心跳
        heartbeat_task = asyncio.create_task(send_heartbeats())
        
        # 执行请求
        response = await router.ainvoke(capability, payload)
        
        emit_event("ai_request_complete", f"✅ {task_name} 完成")
        return response
        
    except asyncio.CancelledError:
        emit_event("ai_request_cancelled", f"🚫 {task_name} 已取消")
        raise
    except Exception as e:
        emit_event("ai_request_error", f"❌ {task_name} 失败: {e}")
        logger.error(f"[AI请求] {task_name} 失败: {e}")
        raise
    finally:
        if heartbeat_task:
            heartbeat_task.cancel()
            try:
                await heartbeat_task
            except asyncio.CancelledError:
                pass


async def acall_with_heartbeat(
    router: Any,
    capability: str,
    messages: list[dict[str, str]],
    response_format: dict | None = None,
    task_name: str = "AI处理",
    heartbeat_interval: float = 2.0,
    event_callback: Callable[[str, str, str], None] | None = None,
) -> str:
    """非流式 acall_capability 的心跳封装
    
    Args:
        router: ModelRouter 实例
        capability: AI 能力名称
        messages: 消息列表
        response_format: 响应格式
        task_name: 任务名称
        heartbeat_interval: 心跳发送间隔秒数
        event_callback: 事件回调函数
        
    Returns:
        AI 响应内容
    """
    def emit_event(event_type: str, message: str):
        if event_callback:
            try:
                event_callback(event_type, message, "AI")
            except Exception:
                pass
    
    emit_event("ai_request_start", f"🚀 {task_name} 开始请求")
    
    heartbeat_task = None
    heartbeat_count = 0
    
    async def send_heartbeats():
        nonlocal heartbeat_count
        while True:
            await asyncio.sleep(heartbeat_interval)
            heartbeat_count += 1
            emit_event("ai_heartbeat", f"💓 {task_name} 等待中 ({heartbeat_count * heartbeat_interval:.0f}s)")
    
    try:
        heartbeat_task = asyncio.create_task(send_heartbeats())
        
        response = await router.acall_capability(
            capability, messages, response_format
        )
        
        emit_event("ai_request_complete", f"✅ {task_name} 完成")
        return response
        
    except asyncio.CancelledError:
        emit_event("ai_request_cancelled", f"🚫 {task_name} 已取消")
        raise
    except Exception as e:
        emit_event("ai_request_error", f"❌ {task_name} 失败: {e}")
        logger.error(f"[AI请求] {task_name} 失败: {e}")
        raise
    finally:
        if heartbeat_task:
            heartbeat_task.cancel()
            try:
                await heartbeat_task
            except asyncio.CancelledError:
                pass
