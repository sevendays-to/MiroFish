"""Run with backend/.venv/bin/python backend/scripts/test_responses_model.py [--live]."""
import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dotenv import dotenv_values
from camel.agents import ChatAgent
from camel.types import ModelPlatformType
from responses_model import create_simulation_model, responses_request, as_chat_completion


def fake_response(output, text=''):
    return SimpleNamespace(status='completed', output=output, output_text=text,
                           id='resp_test', created_at=1, model='gpt-6.1-sol', usage=None)


async def check(live=False):
    cfg = dotenv_values(Path(__file__).resolve().parents[2] / '.env') if live else {}
    model = create_simulation_model(ModelPlatformType.OPENAI, 'gpt-6.1-sol',
                                    api_key=cfg.get('LLM_API_KEY', 'test'),
                                    url=cfg.get('LLM_BASE_URL'),
                                    model_config_dict={'max_tokens': 4096})
    posts = []

    def create_post(content: str) -> str:
        """Publish a social post.

        Args:
            content: Text of the post.
        """
        posts.append(content)
        return 'Published successfully'

    if not live:
        reasoning = SimpleNamespace(type='reasoning', model_dump=lambda **kw: {
            'type': 'reasoning', 'id': 'rs_test', 'summary': [], 'encrypted_content': 'encrypted'})
        call = SimpleNamespace(type='function_call', call_id='call_test',
                               name='create_post', arguments='{"content":"Test post"}')
        responses = [fake_response([reasoning, call]), fake_response([], 'Published')]
        model._async_client.responses.create = AsyncMock(side_effect=responses)
        model._client.responses.create = Mock(return_value=fake_response([], 'OK'))
        assert model.run([{'role': 'user', 'content': 'Hello'}]).choices[0].message.content == 'OK'

    agent = ChatAgent(system_message='Call create_post exactly once with content exactly the quoted string \"Test post\". '
                      'After the tool succeeds, reply Published without calling any more tools.',
                      model=model, tools=[create_post], max_iteration=3, retry_attempts=1)
    result = await agent.astep('Publish the test post now.')
    assert len(posts) == 1 and 'Test post' in posts[0], posts
    assert len(result.info['tool_calls']) == 1
    assert result.msgs and result.msgs[0].content
    if not live:
        requests = model._async_client.responses.create.call_args_list
        assert len(requests) == 2
        first, second = [entry.kwargs for entry in requests]
        assert first['reasoning'] == {'effort': 'medium'}
        assert first['max_output_tokens'] == 16384
        assert first['tools'][0]['name'] == 'create_post'
        assert first['tools'][0]['strict'] is False
        assert 'temperature' not in first and 'max_tokens' not in first
        assert any(item.get('encrypted_content') == 'encrypted' for item in second['input'])
        assert any(item.get('type') == 'function_call_output' and
                   item['call_id'] == 'call_test' for item in second['input'])
        req = responses_request('gpt-6.1-sol', [{'role': 'user', 'content': 'Return JSON'}],
                                response_format={'type': 'json_object'})
        assert req['text']['format'] == {'type': 'json_object'}
        assert 'max_output_tokens' not in req
        assert responses_request('gpt-6.1-sol', [], max_tokens=32768)['max_output_tokens'] == 32768
        try:
            as_chat_completion(SimpleNamespace(status='incomplete',
                incomplete_details=SimpleNamespace(reason='max_output_tokens')))
        except RuntimeError as error:
            assert 'max_output_tokens' in str(error)
        else:
            raise AssertionError('Incomplete responses must fail')
        legacy = create_simulation_model(ModelPlatformType.OPENAI, 'gpt-4o-mini', api_key='test')
        assert type(legacy).__name__ == 'OpenAIModel'
    else:
        from app.services.ontology_generator import OntologyGenerator
        from app.utils.llm_client import LLMClient
        client = LLMClient(api_key=cfg['LLM_API_KEY'], base_url=cfg.get('LLM_BASE_URL'),
                           model='gpt-6.1-sol')
        text = ('Farmers, agricultural traders, food manufacturers, consumer groups, '
                'energy companies, ministries, universities and journalists debate '
                'food prices, drought and energy costs on social media. ')
        ontology = OntologyGenerator(client).generate(
            document_texts=[(text * 300)[:50000]],
            simulation_requirement='Simulate discussion of rising food and energy prices.')
        assert len(ontology['entity_types']) == 10
        assert 6 <= len(ontology['edge_types']) <= 10
        print('PASS: live ontology generation with 50,000 characters and a legacy 4,096-token limit')
    print('PASS: CAMEL async tool execution and continuation' + (' (live API)' if live else ' (mock API)'))


if __name__ == '__main__':
    asyncio.run(check('--live' in sys.argv))
