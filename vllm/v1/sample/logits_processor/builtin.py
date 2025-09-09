# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Optional

import torch

from vllm.v1.sample.logits_processor.interface import (BatchUpdate,
                                                       LogitsProcessor,
                                                       MoveDirectionality)

if TYPE_CHECKING:
    from vllm.config import VllmConfig


class MinPLogitsProcessor(LogitsProcessor):

    def __init__(self, vllm_config: "VllmConfig", device: torch.device,
                 is_pin_memory: bool):
        max_num_reqs = vllm_config.scheduler_config.max_num_seqs
        self.min_p_count: int = 0

        self.min_p_cpu_tensor = torch.zeros((max_num_reqs, ),
                                            dtype=torch.float32,
                                            device="cpu",
                                            pin_memory=is_pin_memory)
        self.min_p_cpu = self.min_p_cpu_tensor.numpy()

        self.use_double_tensor = torch.device(device).type != "cpu"

        if self.use_double_tensor:
            # Pre-allocated device tensor
            self.min_p_device: torch.Tensor = torch.empty((max_num_reqs, ),
                                                          dtype=torch.float32,
                                                          device=device)
        else:
            self.min_p_device = self.min_p_cpu_tensor
        # Current slice of the device tensor
        self.min_p: torch.Tensor = self.min_p_device[:0]

    def is_argmax_invariant(self) -> bool:
        """Min-p never impacts greedy sampling"""
        return True

    def get_min_p_by_index(self, index: int) -> float:
        return float(self.min_p_cpu[index])

    def update_state(self, batch_update: Optional[BatchUpdate]):
        if not batch_update:
            return

        needs_update = False
        # Process added requests.
        for index, params, _, _ in batch_update.added:
            min_p = params.min_p
            if self.min_p_cpu[index] != min_p:
                needs_update = True
                self.min_p_cpu[index] = min_p
            if min_p:
                self.min_p_count += 1

        if self.min_p_count:
            # Process removed requests.
            needs_update |= bool(batch_update.removed)
            for index in batch_update.removed:
                if self.min_p_cpu[index]:
                    self.min_p_count -= 1

            # Process moved requests, unidirectional (a->b) and swap (a<->b)
            for adx, bdx, direct in batch_update.moved:
                change = (min_p_a :=
                          self.min_p_cpu[adx]) != (min_p_b :=
                                                   self.min_p_cpu[bdx])
                needs_update |= change
                if change:
                    self.min_p_cpu[bdx] = min_p_a
                    if direct == MoveDirectionality.SWAP:
                        self.min_p_cpu[adx] = min_p_b

        # Update tensors if needed.
        size = batch_update.batch_size
        if self.min_p_count and (needs_update or self.min_p.shape[0] != size):
            self.min_p = self.min_p_device[:size]
            if self.use_double_tensor:
                self.min_p.copy_(self.min_p_cpu_tensor[:size],
                                 non_blocking=True)
            self.min_p.unsqueeze_(1)

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if not self.min_p_count:
            return logits

        # Convert logits to probability distribution
        probability_values = torch.nn.functional.softmax(logits, dim=-1)
        # Calculate maximum probabilities per sequence
        max_probabilities = torch.amax(probability_values,
                                       dim=-1,
                                       keepdim=True)
        # Adjust min_p
        adjusted_min_p = max_probabilities.mul_(self.min_p)
        # Identify valid tokens using threshold comparison
        invalid_token_mask = probability_values < adjusted_min_p
        # Apply mask using boolean indexing
        logits[invalid_token_mask] = -float('inf')
        return logits


class LogitBiasLogitsProcessor(LogitsProcessor):

    def __init__(self, _, device: torch.device, is_pin_memory: bool):
        self.device = device
        self.pin_memory = is_pin_memory
        self.biases: dict[int, dict[int, float]] = {}

        self.bias_tensor: torch.Tensor = torch.tensor(())
        self.logits_slice = (self._device_tensor([], torch.int32),
                             self._device_tensor([], torch.int32))

    def is_argmax_invariant(self) -> bool:
        """Logit bias can rebalance token probabilities and change the
        outcome of argmax in greedy sampling."""
        return False

    def update_state(self, batch_update: Optional[BatchUpdate]):
        if not batch_update:
            return

        needs_update: bool = False
        # Process added requests.
        for index, params, _, _ in batch_update.added:
            if lb := params.logit_bias:
                self.biases[index] = lb
                needs_update = True
            else:
                # Drop biases metadata at batch index
                if self.biases.pop(index, None) is not None:
                    # If a new request replaces an old request which
                    # specified biases, we should update processor tensors
                    needs_update = True

        if self.biases:
            # Process removed requests.
            for index in batch_update.removed:
                if self.biases.pop(index, None):
                    needs_update = True

            # Process moved requests, unidirectional (a->b) and swap (a<->b)
            for a_index, b_index, direct in batch_update.moved:
                if direct == MoveDirectionality.UNIDIRECTIONAL:
                    if (a_entry := self.biases.pop(a_index, None)) is None:
                        if self.biases.pop(b_index, None) is not None:
                            needs_update = True
                    else:
                        self.biases[b_index] = a_entry
                        needs_update = True
                else:
                    a_entry = self.biases.pop(a_index, None)
                    if (b_entry := self.biases.pop(b_index, None)) is not None:
                        self.biases[a_index] = b_entry
                        needs_update = True
                    if a_entry is not None:
                        self.biases[b_index] = a_entry
                        needs_update = True

        # Update tensors if needed.
        if needs_update:
            reqs, tok_ids, biases = [], [], []
            for req, lb in self.biases.items():
                reqs.extend([req] * len(lb))
                tok_ids.extend(lb.keys())
                biases.extend(lb.values())

            self.bias_tensor = self._device_tensor(biases, torch.float32)
            self.logits_slice = (self._device_tensor(reqs, torch.int32),
                                 self._device_tensor(tok_ids, torch.int32))

    def _device_tensor(self, data: list, dtype: torch.dtype) -> torch.Tensor:
        return (torch.tensor(data,
                             device="cpu",
                             dtype=dtype,
                             pin_memory=self.pin_memory).to(device=self.device,
                                                            non_blocking=True))

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if self.biases:
            logits[self.logits_slice] += self.bias_tensor
        return logits


class MinTokensLogitsProcessor(LogitsProcessor):

    def __init__(self, vllm_config: "VllmConfig", device: torch.device,
                 is_pin_memory: bool):
        # index -> (min_toks, output_token_ids, stop_token_ids)
        self.device = device
        self.pin_memory = is_pin_memory
        self.min_toks: dict[int, tuple[int, Sequence[int], set[int]]] = {}

        # (req_idx_tensor,eos_tok_id_tensor)
        self.logits_slice: tuple[torch.Tensor,
                                 torch.Tensor] = (self._device_tensor(
                                     [], torch.int32),
                                                  self._device_tensor(
                                                      [], torch.int32))

    def is_argmax_invariant(self) -> bool:
        """By censoring stop tokens, min-tokens can change the outcome
        of the argmax operation in greedy sampling."""
        return False

    def update_state(self, batch_update: Optional[BatchUpdate]):
        needs_update = False

        if batch_update:
            # Process added requests.
            for index, params, _, output_tok_ids in batch_update.added:
                if ((min_tokens := params.min_tokens)
                        and len(output_tok_ids) < min_tokens):
                    # Replace request metadata at batch index
                    self.min_toks[index] = (min_tokens, output_tok_ids,
                                            params.all_stop_token_ids)
                    needs_update = True
                else:
                    # Drop min_toks metadata at batch index
                    if self.min_toks.pop(index, None) is not None:
                        # If a new request replaces an old request which
                        # specified min_toks, we should update processor tensors
                        needs_update = True

            if self.min_toks:
                # Process removed requests.
                for index in batch_update.removed:
                    if self.min_toks.pop(index, None):
                        needs_update = True

                # Process moved requests, unidirectional (a->b) and
                # swapped (a<->b)
                for a_index, b_index, direct in batch_update.moved:
                    if direct == MoveDirectionality.UNIDIRECTIONAL:
                        if (a_entry := self.min_toks.pop(a_index,
                                                         None)) is None:
                            if self.min_toks.pop(b_index, None) is not None:
                                needs_update = True
                        else:
                            self.min_toks[b_index] = a_entry
                            needs_update = True
                    else:
                        a_entry = self.min_toks.pop(a_index, None)
                        if (b_entry := self.min_toks.pop(b_index,
                                                         None)) is not None:
                            self.min_toks[a_index] = b_entry
                            needs_update = True
                        if a_entry is not None:
                            self.min_toks[b_index] = a_entry
                            needs_update = True

        if self.min_toks:
            # Check for any requests that have attained their min tokens.
            to_remove = tuple(index for index, (min_toks, out_tok_ids,
                                                _) in self.min_toks.items()
                              if len(out_tok_ids) >= min_toks)
            if to_remove:
                needs_update = True
                for index in to_remove:
                    del self.min_toks[index]

        # Update tensors if needed.
        if needs_update:
            reqs: list[int] = []
            tok_ids: list[int] = []
            for req, (_, _, stop_tok_ids) in self.min_toks.items():
                reqs.extend([req] * len(stop_tok_ids))
                tok_ids.extend(stop_tok_ids)

            self.logits_slice = (self._device_tensor(reqs, torch.int32),
                                 self._device_tensor(tok_ids, torch.int32))

    def _device_tensor(self, data: list, dtype: torch.dtype) -> torch.Tensor:
        return (torch.tensor(data,
                             device="cpu",
                             dtype=dtype,
                             pin_memory=self.pin_memory).to(device=self.device,
                                                            non_blocking=True))

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if self.min_toks:
            # Inhibit EOS token for requests which have not reached min length
            logits[self.logits_slice] = -float("inf")
        return logits


class ThinkingTokenBudgetLogitsProcessor(LogitsProcessor):
    """Limits the number of tokens allowed inside a 'thinking' section."""

    def __init__(self, vllm_config: "VllmConfig", device: torch.device,
                 is_pin_memory: bool):
        """
        Args:
          vllm_config: Configuration for vllm, which includes
            the token IDs for thinking start and end.
          device (torch.device): Device to use for tensor operations.
          is_pin_memory (bool): Whether to use pinned memory for tensors.
        """
        reasoning_config = vllm_config.reasoning_config
        max_num_reqs = vllm_config.scheduler_config.max_num_seqs

        # Check if thinking is enabled
        self.is_enabled = (reasoning_config is not None
                           and reasoning_config.is_thinking_enabled())
        
        if not self.is_enabled:
            return

        self.think_start_token_ids = tuple(
            getattr(reasoning_config, "think_start_token_ids", []))
        self.think_end_token_ids = tuple(
            getattr(reasoning_config, "think_end_token_ids", []))

        if not self.think_start_token_ids or not self.think_end_token_ids:
            self.is_enabled = False
            return
        
        # The length of the longest marker sequence minus one, for the sliding window
        self.window_size = max(len(self.think_start_token_ids),
                               len(self.think_end_token_ids)) - 1

        self.pin_memory = is_pin_memory
        self.device = device
        self._state: dict[int, dict[str, Any]] = {}
        # Preallocate reusable tensors
        self.mask = torch.zeros(max_num_reqs, dtype=torch.bool, device=device)
        self.force_token_ids = torch.full((max_num_reqs, ),
                                          -1,
                                          dtype=torch.long,
                                          device=device)

    @staticmethod
    def _find_last_sequence_index(target_list: list[int],
                                  token_ids: list[int]) -> int:
        """
        Returns the index of the last occurrence of token_ids in target_list.
        Args:
          target_list (list[int]): The list of token IDs.
          token_ids (list[int]): The sequence of token IDs to find.
        """
        if not token_ids:
            return -1

        len_tok = len(token_ids)
        for i in range(len(target_list) - len_tok, -1, -1):
            if target_list[i:i + len_tok] == token_ids:
                return i
        return -1


    @staticmethod
    def _get_initial_window(token_ids: list[int],
                            window_size: int) -> list[int]:
        """Gets the initial token window from the end of a list."""
        if window_size <= 0:
            return []
        return token_ids[-window_size:]

    def _init_state_entry(self, prompt_tok_ids: list[int],
                          thinking_token_budget: int) -> dict[str, Any]:
        """Initializes the tracking state for a given sequence index."""
        last_start = self._find_last_sequence_index(prompt_tok_ids,
                                                    list(self.think_start_token_ids))
        last_end = self._find_last_sequence_index(prompt_tok_ids,
                                                  list(self.think_end_token_ids))

        in_think = last_start > last_end
        think_count = 0
        if in_think:
            think_count = len(prompt_tok_ids) - (
                last_start + len(self.think_start_token_ids))

        return {
            "in_think": in_think,  # Currently in thinking mode
            "in_end": False,  # Currently forcing end tokens
            "think_count": think_count,  # Number of tokens in thinking section
            "end_count": 0,  # Number of end tokens forced so far
            "thinking_token_budget": thinking_token_budget,
            "prev_output_len": 0, # To track new tokens
            # A sliding window to detect start/end sequences efficiently
            "token_window": self._get_initial_window(prompt_tok_ids, self.window_size),
        }

    def _update_think_state(self, state: dict[str, Any], output_tok_ids: list[int]):
        """Updates the state based on newly generated output tokens."""
        current_len = len(output_tok_ids)
        prev_len = state["prev_output_len"]
        
        if current_len <= prev_len:
            return

        new_tokens = output_tok_ids[prev_len:]
        state["prev_output_len"] = current_len
        
        # If we are already forcing an end, just advance the counter
        if state["in_end"]:
            state["end_count"] += len(new_tokens)
            if state["end_count"] >= len(self.think_end_token_ids):
                state["in_end"] = False
                state["end_count"] = 0
                state["token_window"].clear() # Reset window after forced end
            return

        for token in new_tokens:
            # 1. CHECK FIRST: Create the sequence for checking using the *current* state
            # of the window *before* it gets updated.
            current_sequence = tuple(state["token_window"] + [token])

            if state["in_think"]:
                state["think_count"] += 1
                # Check if we are exiting think mode
                if current_sequence.endswith(self.think_end_token_ids):
                    state["in_think"] = False
                    state["think_count"] = 0
            else:
                # Check if we are entering think mode
                if current_sequence.endswith(self.think_start_token_ids):
                    state["in_think"] = True
                    state["think_count"] = 0

            # Check budget *after* state change
            if state["in_think"] and state["think_count"] >= state["thinking_token_budget"]:
                state["in_think"] = False
                state["in_end"] = True
                state["end_count"] = 0
                
            # 2. UPDATE LATER: Now that all checks for the current token are done,
            # update the window to prepare for the *next* token.
            if self.window_size > 0:
                state["token_window"].append(token)
                if len(state["token_window"]) > self.window_size:
                    state["token_window"].pop(0)
            
            # If we've entered forced-end mode, stop processing other new tokens
            if state["in_end"]:
                break
                
    def is_argmax_invariant(self) -> bool:
        """This logits processor can change the outcome of
        greedy sampling by forcing that the thinking section
        ends after a certain number of tokens."""
        return False

    def update_state(self, batch_update: Optional["BatchUpdate"]):
        if not self.is_enabled or not batch_update:
            return

        for index in batch_update.removed:
            self._state.pop(index, None)

        for (index, params, prompt_tok_ids, output_tok_ids) in batch_update.added:
            # Use getattr for safer access to params
            budget = getattr(params, 'thinking_token_budget', None)
            if budget is not None:
                self._state[index] = self._init_state_entry(prompt_tok_ids, budget)
                self._update_think_state(self._state[index], output_tok_ids)
            else:
                self._state.pop(index, None)

        for i1, i2, direction in batch_update.moved:
            if direction == MoveDirectionality.SWAP:
                self._state[i1], self._state[i2] = self._state.get(i2), self._state.get(i1)
            else: # Move
                if i1 in self._state:
                    self._state[i2] = self._state.pop(i1)
        
        # After all structural changes, update states for all active sequences
        for index, state in self._state.items():
            output_tok_ids = batch_update.get_output_tokens(index) # Fictional method
            self._update_think_state(state, output_tok_ids)

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if not self.is_enabled or not self._state:
            return logits

        batch_size = logits.size(0)
        self.mask.zero_() # Use in-place zeroing for efficiency

        has_active_forcing = False
        for i in range(batch_size):
            state = self._state.get(i)
            if state and state["in_end"]:
                end_token_idx = min(state["end_count"], len(self.think_end_token_ids) - 1)
                self.mask[i] = True
                self.force_token_ids[i] = self.think_end_token_ids[end_token_idx]
                has_active_forcing = True

        if has_active_forcing:
            # Masking operation on GPU
            active_indices = self.mask.nonzero(as_tuple=True)[0]
            logits[active_indices] = -float('inf')
            force_tokens = self.force_token_ids[active_indices]
            # Apply a large value for the end thinking token id index
            logits[active_indices, force_tokens] = 1e9

        return logits
