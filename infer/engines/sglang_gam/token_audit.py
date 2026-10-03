"""Opt-in native output-token audit at the scheduler's finished boundary."""
import hashlib
import json
import os
from pathlib import Path


def install_token_audit():
    path = os.environ.get('GAM_SGLANG_TOKEN_AUDIT_PATH')
    if not path:
        return False
    from sglang.srt.managers.schedule_batch import Req
    if getattr(Req, '_gam_native_token_audit', False):
        return True
    original = Req.check_finished
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    def check_finished(self, new_accepted_len=1):
        result = original(self, new_accepted_len)
        if self.finished() and not getattr(self, '_gam_token_audit_written', False):
            tokens = list(self.output_ids_through_stop)
            encoded = json.dumps(tokens, separators=(',', ':')).encode()
            record = dict(request_id=self.rid, output_token_ids=tokens,
                          token_sha256=hashlib.sha256(encoded).hexdigest(),
                          completion_tokens=len(tokens),
                          discarded_tail_tokens=len(self.output_ids)-len(tokens),
                          finished_reason=(self.finished_reason.to_json()
                                           if hasattr(self.finished_reason,'to_json')
                                           else str(self.finished_reason)))
            # Exactly one append per completed request; no device access and
            # no per-block trace/synchronization. Diagnostic routes only.
            with open(path, 'a', encoding='utf-8') as stream:
                stream.write(json.dumps(record, separators=(',', ':'))+'\n')
            self._gam_token_audit_written = True
        return result

    Req.check_finished = check_finished
    Req._gam_native_token_audit = True
    return True
