"""GAM-special-token-preserving batch reward manager for DARE/verl."""

from __future__ import annotations

from collections import defaultdict
import math

import torch

from train.rl.shared.reward import MultiRouteGAMRewardAdapter


class GAMBatchRewardManager:
    def __init__(self, tokenizer, num_examine=0, **_: object):
        self.tokenizer = tokenizer
        self.num_examine = int(num_examine)
        self.adapter = MultiRouteGAMRewardAdapter()
        self.eos_ids = {
            int(value)
            for value in (
                tokenizer.eos_token_id,
                tokenizer.convert_tokens_to_ids("<|im_end|>"),
            )
            if value is not None and int(value) >= 0
        }

    def _visible_ids_and_last_position(
        self, response_ids: torch.Tensor, response_mask: torch.Tensor
    ) -> tuple[torch.Tensor, int]:
        positions = torch.nonzero(response_mask.bool(), as_tuple=False).flatten()
        valid = response_ids[positions]
        visible_length = int(valid.numel())
        for index, value in enumerate(valid.tolist()):
            if int(value) in self.eos_ids:
                visible_length = index + 1
                break
        visible = valid[:visible_length]
        if visible_length <= 0:
            return visible, -1
        return visible, int(positions[visible_length - 1].item())

    def __call__(self, data, return_dict=False):
        response_mask = data.batch.get("response_mask")
        if response_mask is None:
            prompt_length = data.batch["prompts"].shape[-1]
            response_mask = data.batch["attention_mask"][:, prompt_length:]
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        extra = defaultdict(list)
        for index in range(len(data)):
            visible_ids, last_position = self._visible_ids_and_last_position(
                data.batch["responses"][index], response_mask[index]
            )
            if visible_ids.numel() == 0:
                raise RuntimeError("RLV3 rollout produced an empty response")
            decode_kwargs = {
                "skip_special_tokens": False,
                "clean_up_tokenization_spaces": False,
                "spaces_between_special_tokens": False,
            }
            try:
                response = self.tokenizer.decode(visible_ids, **decode_kwargs)
            except TypeError:
                # Transformers tokenizers do not all expose SGLang's spacing
                # keyword.  Special tokens still remain intact; only remove
                # the unsupported compatibility keyword.
                decode_kwargs.pop("spaces_between_special_tokens")
                response = self.tokenizer.decode(visible_ids, **decode_kwargs)
            metadata = data.non_tensor_batch["extra_info"][index]
            row = metadata["rlv3_row"]
            result = self.adapter.score(response, row)
            score = sum(result.components[key] * result.weights[key] for key in result.weights)
            if not math.isfinite(score) or not 0.0 <= score <= 1.0 + 1e-6:
                raise RuntimeError(f"invalid RLV3 reward {score} for {row['id']}")
            # Put the sequence reward on the final visible token.  Normally
            # response_mask is already EOS-truncated by verl; deriving this
            # index from the independently audited visible prefix also makes
            # the manager fail-safe if an upstream mask contains padded tail.
            reward_tensor[index, last_position] = float(score)
            extra["score"].append(float(score))
            extra["format_valid"].append(float(result.format_valid))
            extra["route"].append(str(row["rlv3_route"]))
            for key, value in result.components.items():
                extra[key].append(float(value))
            if index < self.num_examine:
                print(
                    {"id": row["id"], "route": row["rlv3_route"], "response": response, "score": score},
                    flush=True,
                )
        if return_dict:
            return {"reward_tensor": reward_tensor, "reward_extra_info": extra}
        return reward_tensor
