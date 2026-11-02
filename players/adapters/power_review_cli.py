"""PowerMate's power-review prompt played through a subscription CLI (Claude Max, ChatGPT).

Same prompt and contract as PowerReviewFreeAdapter, but the call goes through
SubprocessCliAdapter, so the CLI runs contained by framework.hermetic (no tools, a
fresh workspace outside the repo) and no API key is involved (owner 2026-10-29: no API
or cloud usage). The arena scorer validates the proposal itself; this adapter only
marks a reply with no JSON as a scored invalid output, never as a transport failure.
"""
from framework.player_adapter import SubprocessCliAdapter, register_adapter_class
from players.adapters.power_review_free import review_task


class PowerReviewCliAdapter(SubprocessCliAdapter):
    def play_task(self, envelope, *, timeout_s):
        from powerbench.review.assistant import assistant_prompt
        text = envelope['input']['text']
        confirmed = envelope['input'].get('confirmed')
        prompt = assistant_prompt(review_task(envelope['arena_id']), text, confirmed)
        # The prompt template is the bare {{INPUT_TEXT}} slot: the whole PowerBench
        # prompt (system instructions, then the task) is the input it receives.
        combined = prompt['system'] + '\n\n' + prompt['prompt']
        instrument = {'prompt_version': prompt['prompt_version'], 'contract_sha256': prompt['contract_sha256'],
                      'cli': self.cli_command[0], 'attempts': 1}
        self._last_stdout = None
        try:
            proposed = super().play_task({**envelope, 'input': {'text': combined}}, timeout_s=timeout_s)
        except ValueError as error:
            return {'_validated': None, '_instrument': instrument,
                    '_attempt': {'status': 'invalid_model_output', 'response_received': True,
                                 'failure_class': type(error).__name__, 'raw_content': self._last_stdout}}
        if not isinstance(proposed, dict):
            return {'_validated': None, '_instrument': instrument,
                    '_attempt': {'status': 'invalid_model_output', 'response_received': True,
                                 'failure_class': 'NotAnObject', 'raw_content': self._last_stdout}}
        return {**proposed, '_instrument': instrument,
                '_attempt': {'status': 'proposal_received', 'response_received': True,
                             'raw_content': self._last_stdout}}


register_adapter_class('PowerReviewCliAdapter', PowerReviewCliAdapter)
