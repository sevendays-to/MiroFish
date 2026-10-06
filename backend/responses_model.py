"""Responses API bridge for MiroFish's text client and CAMEL tool loop."""
from openai.types.chat import ChatCompletion


def uses_responses(model):
    return str(model).startswith('gpt-6')


def responses_request(model, messages, tools=None, max_tokens=None, response_format=None,
                      reasoning_items=None):
    items = []
    seen = set()
    for message in messages:
        if message['role'] == 'tool':
            items.append({'type': 'function_call_output',
                          'call_id': message['tool_call_id'], 'output': message['content']})
            continue
        calls = message.get('tool_calls') or []
        for call in calls:
            for item in (reasoning_items or {}).get(call['id'], []):
                if item['id'] not in seen:
                    items.append(item)
                    seen.add(item['id'])
        if message.get('content'):
            items.append({'role': message['role'], 'content': message['content']})
        for call in calls:
            items.append({'type': 'function_call', 'call_id': call['id'],
                          'name': call['function']['name'],
                          'arguments': call['function']['arguments']})
    request = dict(model=str(model), input=items, reasoning={'effort': 'medium'},
                   store=False, include=['reasoning.encrypted_content'])
    if max_tokens:
        # Responses counts reasoning tokens too; legacy text limits need headroom.
        request['max_output_tokens'] = max(16384, max_tokens)
    if tools:
        request['tools'] = [dict(type='function', **dict(tool['function'], strict=False))
                            for tool in tools]
    if response_format:
        if not isinstance(response_format, dict):
            response_format = {'type': 'json_schema', 'json_schema': {
                'name': response_format.__name__, 'schema': response_format.model_json_schema(),
                'strict': True}}
        fmt = dict(response_format)
        if fmt['type'] == 'json_schema':
            fmt = dict(type='json_schema', **fmt['json_schema'])
        request['text'] = {'format': fmt}
    return request


def as_chat_completion(response, reasoning_items=None):
    if response.status != 'completed':
        details = getattr(response, 'incomplete_details', None)
        reason = getattr(details, 'reason', None) or getattr(response, 'error', None)
        raise RuntimeError(f'Responses API did not complete: {response.status} ({reason or "unknown reason"})')
    calls = []
    reasoning = [item.model_dump(exclude_none=True) for item in response.output
                 if item.type == 'reasoning']
    for item in response.output:
        if item.type == 'function_call':
            calls.append({'id': item.call_id, 'type': 'function',
                          'function': {'name': item.name, 'arguments': item.arguments}})
            if reasoning_items is not None:
                reasoning_items[item.call_id] = reasoning
    message = {'role': 'assistant', 'content': response.output_text or None}
    if calls:
        message['tool_calls'] = calls
    usage = response.usage
    return ChatCompletion(id=response.id, created=int(response.created_at),
                          model=response.model, object='chat.completion',
                          choices=[{'index': 0, 'message': message,
                                    'finish_reason': 'tool_calls' if calls else 'stop'}],
                          usage={'prompt_tokens': usage.input_tokens,
                                 'completion_tokens': usage.output_tokens,
                                 'total_tokens': usage.total_tokens} if usage else None)


def create_simulation_model(model_platform, model_type, **kwargs):
    from camel.models import ModelFactory, OpenAIModel
    if not uses_responses(model_type):
        return ModelFactory.create(model_platform=model_platform, model_type=model_type, **kwargs)

    class ResponsesModel(OpenAIModel):
        def __init__(self, **params):
            super().__init__(**params)
            self._reasoning_items = {}

        def _request(self, messages, response_format, tools):
            if self.stream:
                raise ValueError('MiroFish Responses adapter requires non-streaming mode')
            return responses_request(self.model_type, messages, tools,
                                     self.model_config_dict.get('max_tokens'), response_format,
                                     self._reasoning_items)

        def _run(self, messages, response_format=None, tools=None):
            return as_chat_completion(self._client.responses.create(
                **self._request(messages, response_format, tools)), self._reasoning_items)

        async def _arun(self, messages, response_format=None, tools=None):
            return as_chat_completion(await self._async_client.responses.create(
                **self._request(messages, response_format, tools)), self._reasoning_items)

    return ResponsesModel(model_type=model_type, **kwargs)
