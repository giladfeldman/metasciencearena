"""Exact PowerMate prompt/settings/parser, one bounded request and no fallback."""
import json
import time
import requests
from players.adapters.power_review_http import bounded_post
from framework.player_adapter import register_adapter_class
from players.adapters.openai_compatible import OpenAIChatCompletionsAdapter, _usage_from_response, _response_meta

_REVIEW_TASKS=('extraction','audit','rewrite','code')


def review_task(arena_id):
    """PowerBench review task for an arena: power-reporting-v1 is extraction,
    power-review-<task>-v<N> is <task> (integrated arenas and replayed draft ids)."""
    if arena_id=='power-reporting-v1':return 'extraction'
    task=arena_id.removeprefix('power-review-').rsplit('-v',1)[0]
    if not arena_id.startswith('power-review-') or task not in _REVIEW_TASKS:
        raise ValueError(f'{arena_id} is not a power-review arena')
    return task


class PowerReviewFreeAdapter(OpenAIChatCompletionsAdapter):
    def play_task(self,envelope,*,timeout_s):
        from powerbench.review.assistant import assistant_prompt, validate_assistance
        if self._url is None or self._key is None:self.prepare()
        self._last_usage=None;self._last_meta=None;self._last_request_max_tokens=None
        text=envelope['input']['text'];task=review_task(envelope['arena_id'])
        confirmed=envelope['input'].get('confirmed')
        prompt=assistant_prompt(task,text,confirmed)
        settings=dict(self.openai_extra_body or {})
        body={**settings,'model':self.openai_model,'messages':[{'role':'system','content':prompt['system']},{'role':'user','content':prompt['prompt']}]}
        self._last_request_max_tokens=body.get('max_tokens')
        wait=0
        if self.openai_api_key_env=='GROQ_API_KEY':
            # Keep diagnostic calls under the account token-per-minute budget.
            wait=max(0,30-(time.monotonic()-getattr(self,'_last_started',0)))
            if wait:time.sleep(wait)
        self._last_started=time.monotonic()
        captured=bounded_post(self._url,self._key,body,timeout_s)
        inference_seconds=time.monotonic()-self._last_started
        response=requests.Response();response.status_code=captured['status'];response._content=captured['body'].encode('utf-8')
        self._last_usage=_usage_from_response(response);self._last_meta=_response_meta(response,1)
        if not 200<=response.status_code<300:raise RuntimeError(f'Provider request failed (HTTP {response.status_code}); response body redacted')
        content=None
        instrument={'prompt_version':prompt['prompt_version'],'contract_sha256':prompt['contract_sha256'],'settings':settings,'timeout_seconds':min(float(timeout_s),55),'attempts':1,
                    'pacing_seconds':wait,'inference_seconds':inference_seconds}
        try:
            data=response.json()
            if not isinstance(data,dict):raise ValueError('Provider envelope must be an object')
            choices=data.get('choices')
            choice=choices[0] if isinstance(choices,list) and choices and isinstance(choices[0],dict) else {}
            message=choice.get('message')
            content=message.get('content') if isinstance(message,dict) else None
            instrument.update({'served_model':data.get('model'),'finish_reason':choice.get('finish_reason')})
            if data.get('model')!=self.openai_model:raise ValueError('Served model differs from evaluated instrument')
            if choice.get('finish_reason')!='stop':raise ValueError('Incomplete generation')
            if not isinstance(content,str) or not content.strip():raise ValueError('No structured proposal content')
            def reject_constant(value):
                raise ValueError('Nonfinite JSON constant')
            proposed=json.loads(content,parse_constant=reject_constant)
            from powerbench.contracts import finite_tree
            finite_tree(proposed)
            validated=validate_assistance(task,text,proposed,confirmed)
        except (ValueError,TypeError,KeyError,OverflowError) as error:
            # A received but invalid proposal is a scored wrong answer. Preserve
            # its original content in custody; never turn rejection into a
            # transport failure that hides critical proposal errors.
            defaults={'extraction':{'has_power_analysis':False,'analyses':[]},
                      'audit':{'findings':[]},'code':{'fields':{},'findings':[]},
                      'rewrite':{'paragraph':'','confirmed':confirmed}}
            return {**defaults[task], '_validated':None, '_instrument':instrument,
                    '_attempt':{'status':'invalid_model_output','response_received':True,
                                'failure_class':type(error).__name__,'raw_content':content,
                                'raw_response_body':captured['body']}}
        return {**proposed,'_validated':validated,'_instrument':instrument,
                '_attempt':{'status':'validated_proposal','response_received':True,'raw_content':content,
                            'raw_response_body':captured['body']}}

register_adapter_class('PowerReviewFreeAdapter',PowerReviewFreeAdapter)
