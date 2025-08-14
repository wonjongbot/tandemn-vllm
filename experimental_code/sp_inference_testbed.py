
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

"""
A single-machine harness to profile `src/utils/inference_utils.py`
without MongoDB, tensor-iroh, or real vLLM engines.


Usage
-----
# quick run
python experimental_code/testbed.py --tokens

# profile with cProfile + view with snakeviz (8k token generated)
python experimental_code/testbed.py --profile

snakeviz prof.out

Notes
-----
• Purposely pass a bogus url so the final-result POST just logs.
• vLLM should work just fine since all the signitures match.
"""

import asyncio
import threading
import pickle
import time
from types import SimpleNamespace
from typing import Any, Dict, Optional, List

import numpy as np
import torch
import torch.nn as nn

# Import the code under test
from src.utils.inference_utils import (
    register_inference_hooks,
    INFERENCE_CONTEXT,
    STEP_EVENTS,
    STEP_EVENTS_SAMPLER,
)
from src.utils.message_processing import extract_request_metadata

import src.utils.inference_utils as iu

class _NoLock:
    def __enter__(self): return None
    def __exit__(self, *args): return False

iu.INFERENCE_MUTEX = _NoLock()


class MockTensorTransport:
    """
    Fake transport class to mirror the behavior of TensorTransport.
    
    send(name,tensor): 
        * parses th ename and populates INFERENCE_CONTEXT.
        * sets the STEP_EVENTS[req_id][step].event().
    """
    def __init__(self) -> None:
        self._ticket = "mock_ticket"

    async def start(self) -> None:
        return

    @property
    def ticket(self) -> str:
        return self._ticket

    @staticmethod
    def _rewrite_name_for_dest(name: str, dest_peer: str) -> str:
        """
        This method rewrites request_id portion of the tensor name so event/context
        dicts are local to the peer. We wouldn't need this in an actual setup.
        
        If name is              "{req}_step{n}_..." 
        and req looks like      "base::peerX",
        we rewrite to           "base::{dest_peer}".
        """
        meta = extract_request_metadata(name)
        if not meta:
            return name
        req_id, step_idx, _ = meta
        # Detect base id (support ids without suffix too)
        base = req_id.split("::", 1)[0]
        new_req = f"{base}::{dest_peer}"
        return name.replace(req_id, new_req, 1)

    async def send(self, peer_addr: str, name: str, tensor) -> None:
        """
        Simulate network by directly updating INFERENCE_CONTEXT and STEP_EVENTS.
        uses _rewrite_name_for_dest() to precent two peers from using the same request
        """
        name = self._rewrite_name_for_dest(name, peer_addr)
        req_meta = extract_request_metadata(name)
        if not req_meta:
            return

        request_id, step_idx, msg_type = req_meta

        # Convert tensor payloads
        if hasattr(tensor, "numpy"):
            arr = tensor.numpy()
        else:
            arr = tensor

        # Initialize INFERENCE_CONTEXT and steps storage
        if request_id not in INFERENCE_CONTEXT:
            INFERENCE_CONTEXT[request_id] = {}
        if str(step_idx) not in INFERENCE_CONTEXT[request_id]:
            INFERENCE_CONTEXT[request_id][str(step_idx)] = {}

        if msg_type == "combined":
            # expected shape: [2, ...]
            if isinstance(arr, np.ndarray):
                hidden = torch.from_numpy(arr[0].copy())
                residual = torch.from_numpy(arr[1].copy())
            else:
                hidden = arr[0]
                residual = arr[1]
            INFERENCE_CONTEXT[request_id][str(step_idx)]["hidden_state"] = hidden
            INFERENCE_CONTEXT[request_id][str(step_idx)]["residual"] = residual
        elif msg_type == "sampler_output":
            # expected: np.uint8 buffer of pickled object
            if isinstance(arr, np.ndarray):
                obj = pickle.loads(arr.tobytes())
            else:
                # torch.Tensor → numpy first
                obj = pickle.loads(arr.detach().cpu().numpy().tobytes())
            INFERENCE_CONTEXT[request_id][str(step_idx)]["sampler_output"] = obj
        else:
            # ignore other names in this harness
            return

        # Signal any waiter (now peer-local)
        event_map = STEP_EVENTS_SAMPLER if msg_type == "sampler_output" else STEP_EVENTS
        event = event_map[request_id].setdefault(step_idx, threading.Event())
        # event = STEP_EVENTS[request_id].setdefault(step_idx, threading.Event())
        event.set()

    async def recv(self) -> Optional[Dict[str, Any]]:
        return None

class TinyLayer(nn.Module):
    """
    Fake model to trigger the same hook signitures as in inference_utils.py
    * First/last layer are the same layer
    * Forward call returns (hiddenStates, residual) for post_hook
    """
    def __init__(self, hidden_size: int):
        super().__init__()
        self.proj_h = nn.Linear(hidden_size, hidden_size, bias=False)
        self.proj_r = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, positions: torch.Tensor, hidden: torch.Tensor, residual: torch.Tensor):
        # dummy hidden and residual projections
        h_out = self.proj_h(hidden)
        r_out = self.proj_r(residual)
        # post_hook will receive (hidden_states, residual)
        return h_out, r_out


class TinySampler(nn.Module):
    """
    Fake sampler to trigger sampler_post_hook
    """
    def forward(self, *args, **kwargs):
        # The sampler_post_hook doesn't care what this returns for non last peers.
        return SimpleNamespace(outputs=[SimpleNamespace(text="<dummy>")])


class MiniLLM:
    """
    Fake implementation of `LLM` class to emulate its behavior and to trigger hooks
    """

    def __init__(self, hidden_size: int = 64, seq_len: int = 4, steps: int = 4):
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.default_steps = steps

        layer = TinyLayer(hidden_size)
        sampler = TinySampler()

        # dummy attr chain expected by inference_utils
        self.llm_engine = SimpleNamespace(
            model_executor=SimpleNamespace(
                driver_worker=SimpleNamespace(
                    model_runner=SimpleNamespace(
                        model=SimpleNamespace(
                            model=SimpleNamespace(layers=nn.ModuleList([layer])),
                            config=SimpleNamespace(hidden_size=hidden_size, vocab_size=1000),
                        ),
                        sampler=sampler,
                    )
                )
            )
        )

    def generate(self, prompts: List[str], sampling_params: Any):
        # Drive N steps so hooks fire multiple times
        max_steps = int(getattr(sampling_params, "max_tokens", self.default_steps))
        layer = self.llm_engine.model_executor.driver_worker.model_runner.model.model.layers[0]
        sampler = self.llm_engine.model_executor.driver_worker.model_runner.sampler

        # Step 0: prompt phase (seq_len tokens)
        positions = torch.arange(self.seq_len).unsqueeze(0)
        hidden = torch.randn(self.seq_len, self.hidden_size)
        residual = torch.randn(self.seq_len, self.hidden_size)
        h, r = layer(positions, hidden, residual)  # triggers pre+post hooks
        _ = sampler(h, r)                          # triggers sampler_post_hook

        # Decode steps: 1 token each
        for t in range(1, max_steps):
            positions = torch.tensor([[t]])
            hidden = torch.randn(1, self.hidden_size)
            residual = torch.randn(1, self.hidden_size)
            h, r = layer(positions, hidden, residual)
            _ = sampler(h, r)

        # Mimic vLLM completion object (enough for inference_utils last peer path)
        return [SimpleNamespace(outputs=[SimpleNamespace(text="fakeoutput")])]

class SamplingParams(SimpleNamespace):
    """
    dummy class to avoide vLLM import
    """
    def __init__(self, max_tokens: int = 4):
        super().__init__(max_tokens=max_tokens)


async def run_two_peer_sim(max_tokens: int = 4):
    """
    Run two peers within the same process.
    Real:
        * hooks
    Fake:
        * transport
        * vLLM (for now)
    """
    transport = MockTensorTransport()
    await transport.start()

    peer0 = "peer0"
    peer1 = "peer1"
    pipeline = [peer0, peer1]

    # Create two dummy LLM instance
    llm0 = MiniLLM(hidden_size=64)
    llm1 = MiniLLM(hidden_size=64)

    # Register hooks for both peers
    start0 = register_inference_hooks(
        llm=llm0,
        node=transport,
        peer_id=peer0,
        server_url="http://127.0.0.1:9999",  # dummy addr
        next_peer_ticket=peer1,
        pipeline=pipeline,
    )

    start1 = register_inference_hooks(
        llm=llm1,
        node=transport,
        peer_id=peer1,
        server_url="http://127.0.0.1:9999",
        next_peer_ticket=None,
        pipeline=pipeline,
    )

    base = f"req_{int(time.time()*1000)}"
    req0 = f"{base}::{peer0}"
    req1 = f"{base}::{peer1}"
    sp = SamplingParams(max_tokens=max_tokens)
    assigned_layers = {peer0: [0], peer1: [0]}

    loop = asyncio.get_running_loop()
    f0 = loop.run_in_executor(None, start0, req0, pipeline, "hello", sp, assigned_layers)
    f1 = loop.run_in_executor(None, start1, req1, pipeline, "hello", sp, assigned_layers)

    await asyncio.gather(f0, f1)

async def warmup_then_profile():
    """
    Runs minimal run_two_peer_sim for warmup, initiate profiling after.
    """
    # warmup to pay CUDA/Triton import/init once
    await run_two_peer_sim(max_tokens=1)

    # profile a longer decode run
    import cProfile, pstats, io
    pr = cProfile.Profile()
    pr.enable()
    await run_two_peer_sim(max_tokens=8192)
    pr.disable()
    pr.dump_stats("prof.out")

    # Print stats for inference_utils.py elements
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).strip_dirs().sort_stats('cumtime')
    ps.print_stats('src/utils/inference_utils.py')
    print(s.getvalue())

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--profile', action='store_true', help='Warmup once, then profile a 64-token run into prof.out and print focused stats')
    parser.add_argument('--tokens', type=int, default=4, help='Tokens for the normal run (ignored for --profile)')
    args = parser.parse_args()

    if args.profile:
        asyncio.run(warmup_then_profile())
    else:
        asyncio.run(run_two_peer_sim(max_tokens=args.tokens))
