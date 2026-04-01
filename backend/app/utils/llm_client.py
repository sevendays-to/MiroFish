"""
LLM客户端封装
统一使用OpenAI格式调用
"""

import json
import re
from typing import Optional, Dict, Any, List
from openai import OpenAI, BadRequestError

from ..config import Config


class LLMClient:
    """LLM客户端"""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None
    ):
        self.api_key = api_key or Config.LLM_API_KEY
        self.base_url = base_url or Config.LLM_BASE_URL
        self.model = model or Config.LLM_MODEL_NAME
        
        if not self.api_key:
            raise ValueError("LLM_API_KEY 未配置")
        
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        response_format: Optional[Dict] = None
    ) -> str:
        """
        发送聊天请求
        
        Args:
            messages: 消息列表
            temperature: 温度参数
            max_tokens: 最大token数
            response_format: 响应格式（如JSON模式）
            
        Returns:
            模型响应文本
        """
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
        if response_format:
            kwargs["response_format"] = response_format

        response = self._create_chat_completion(kwargs)
        content = response.choices[0].message.content
        # 部分模型（如MiniMax M2.5）会在content中包含<think>思考内容，需要移除
        content = re.sub(r'<think>[\s\S]*?</think>', '', content).strip()
        return content

    def _create_chat_completion(self, kwargs: Dict[str, Any]):
        """
        创建 Chat Completion 请求。

        某些 OpenAI 模型不接受 `max_tokens`，而要求使用
        `max_completion_tokens`。这里保留对旧兼容接口的支持，并在
        遇到该类错误时自动回退重试。
        """
        try:
            return self.client.chat.completions.create(**kwargs)
        except BadRequestError as exc:
            if not self._should_retry_with_max_completion_tokens(exc, kwargs):
                raise

            retry_kwargs = dict(kwargs)
            retry_kwargs["max_completion_tokens"] = retry_kwargs.pop("max_tokens")
            return self.client.chat.completions.create(**retry_kwargs)

    @staticmethod
    def _should_retry_with_max_completion_tokens(
        exc: BadRequestError,
        kwargs: Dict[str, Any]
    ) -> bool:
        if "max_tokens" not in kwargs:
            return False

        message_parts = [str(exc)]

        response = getattr(exc, "response", None)
        if response is not None:
            try:
                message_parts.append(response.text)
            except Exception:
                pass

        body = getattr(exc, "body", None)
        if body is not None:
            try:
                message_parts.append(json.dumps(body, ensure_ascii=False))
            except Exception:
                message_parts.append(str(body))

        message = " ".join(part for part in message_parts if part)
        return (
            "max_tokens" in message
            and "max_completion_tokens" in message
            and "unsupported_parameter" in message
        )
    
    def chat_json(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 4096
    ) -> Dict[str, Any]:
        """
        发送聊天请求并返回JSON
        
        Args:
            messages: 消息列表
            temperature: 温度参数
            max_tokens: 最大token数
            
        Returns:
            解析后的JSON对象
        """
        response = self.chat(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"}
        )
        # 清理markdown代码块标记
        cleaned_response = response.strip()
        cleaned_response = re.sub(r'^```(?:json)?\s*\n?', '', cleaned_response, flags=re.IGNORECASE)
        cleaned_response = re.sub(r'\n?```\s*$', '', cleaned_response)
        cleaned_response = cleaned_response.strip()

        try:
            return json.loads(cleaned_response)
        except json.JSONDecodeError:
            raise ValueError(f"LLM返回的JSON格式无效: {cleaned_response}")
