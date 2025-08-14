
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

"""
A single-machine harness to profile `src/utils/inference_utils.py`
without MongoDB, or real vLLM engines.

Uses real tensorTransport Object!

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
import multiprocessing as mp

from src.utils.deployment_utils import load_model_with_selective_layers

from src.utils.tensor_protocol_adapter import TensorTransport
from src.utils.inference_utils import (
    register_inference_hooks,
    INFERENCE_CONTEXT,
    STEP_EVENTS,
    STEP_EVENTS_SAMPLER,
)
from src.utils.message_processing import extract_request_metadata
from experimental_code.sp_inference_testbed import MiniLLM, SamplingParams

import importlib
import contextlib

async def _start_unified_gateway(tt, cur_ticket):
    """
    Find unified_message_gateway(), set its module globals to our child’s
    TensorTransport and ticket, and start it as a background task.
    """
    gateway_fn = None
    owner_mod = None

    # Try most likely homes
    for mod_path in ("src.machine_runner", "machine_runner", "server", "src.utils.inference_utils"):
        try:
            mod = importlib.import_module(mod_path)
            fn = getattr(mod, "unified_message_gateway", None)
            if callable(fn):
                gateway_fn = fn
                owner_mod = mod
                break
        except Exception:
            pass

    if gateway_fn is None:
        print("⚠️ Could not locate unified_message_gateway(); inbound handlers will NOT run.")
        return None

    # Wire the module-globals that gateway expects
    try:
        setattr(owner_mod, "tensor_transport", tt)      # used by gateway.recv()
    except Exception:
        pass
    try:
        setattr(owner_mod, "current_peer_ticket", cur_ticket)  # nice-to-have for logs/handlers
    except Exception:
        pass

    # Gateway takes no args per your snippet
    task = asyncio.create_task(gateway_fn())
    # tiny delay so it can subscribe before we start sending
    await asyncio.sleep(0.05)
    return task

async def _cancel_and_drain(*tasks):
    tasks = [t for t in tasks if t is not None]
    if not tasks:
        return
    for t in tasks:
        try:
            t.cancel()
        except Exception:
            pass
    # drain; don't raise CancelledError or RuntimeError
    await asyncio.gather(*tasks, return_exceptions=True)
    # give the loop a tick to process any final callbacks
    await asyncio.sleep(0)

async def peer_main(role, conn, start_evt, max_tokens):
    """
    One peer process that spwans LLM(mock for now) but uses real tensortransport.
    ctrl_{recv,send}: duplex Pipe endpoints to talk to parent
    start_evt: multiprocessing.Event shared from parent
    """
    
    # Start tenstortransport
    tt = TensorTransport()
    await tt.start()

    cur_ticket = tt.ticket
    conn.send({"type": "ticket", "ticket":cur_ticket}) # send to parent

    # start unified gateway
    gateway_task = await _start_unified_gateway(tt, cur_ticket)
    
    # Block until parent provides pipeline, assigned layers and stuff
    msg = await asyncio.to_thread(conn.recv)
    assert msg.get("type") == "pipeline", f"{role}: expected 'pipeline', got {msg}"

    pipeline        = msg["pipeline"]
    assigned_layers = msg.get("assigned_layers", {})
    run_id          = msg["run_id"]

    cur_idx = pipeline.index(cur_ticket)
    next_peer_ticket = pipeline[cur_idx + 1] if cur_idx < len(pipeline) - 1 else None


    # Instantiate LLM
    llm = MiniLLM(hidden_size=64)
    
    # load_model_with_selective_layers()
    

    # Register hooks
    start_infer = register_inference_hooks(
        llm=llm,
        node=tt,
        peer_id=cur_ticket,
        server_url="http://127.0.0.1:9999",
        next_peer_ticket=next_peer_ticket,
        pipeline=pipeline
    )

    # set default runid context to avoid exceptions
    INFERENCE_CONTEXT.setdefault(run_id, {})
    STEP_EVENTS.setdefault(run_id, {})
    STEP_EVENTS_SAMPLER.setdefault(run_id, {})
    
    # Check if everyone is ready
    is_first = (pipeline[0] == cur_ticket)
    is_last  = (pipeline[-1] == cur_ticket)
    conn.send({"type": "ready", "peer_id": cur_ticket, "is_first": is_first, "is_last": is_last})

    await asyncio.sleep(0.05)

    # # Wait til parent says go
    # start_evt.wait()

    # Run
    sp = SamplingParams(max_tokens=max_tokens)
    loop = asyncio.get_running_loop()

    try:
        await loop.run_in_executor(
            None, start_infer, run_id, pipeline, "foomsg", sp, assigned_layers
        )
    finally:
        await _cancel_and_drain(gateway_task)

def _wrap(role, child_conn, start_evt, max_tokens):
    asyncio.run(peer_main(role, child_conn, start_evt, max_tokens))

def spawn_two_peers(max_tokens=32):
    ctx = mp.get_context("spawn")
    start_evt = ctx.Event()

    p0_parent, p0_child = ctx.Pipe(duplex=True)
    p1_parent, p1_child = ctx.Pipe(duplex=True)

    p0 = ctx.Process(target=_wrap, args=("peer0",p0_child,start_evt,max_tokens))
    p1 = ctx.Process(target=_wrap, args=("peer1",p1_child,start_evt,max_tokens))
    p0.start()
    p1.start()

    t0 = p0_parent.recv()["ticket"]
    t1 = p1_parent.recv()["ticket"]
    pipeline = [t0,t1]

    assigned_layers = {t0:[0], t1:[1]}
    run_id = f"req_{int(time.time()*1000)}"

    msg = {
        "type": "pipeline", 
        "pipeline": pipeline, 
        "assigned_layers": assigned_layers,
        "run_id": run_id
        }
    p0_parent.send(msg)
    p1_parent.send(msg)

    # check if everyone is ready
    r0 = p0_parent.recv()
    r1 = p1_parent.recv()
    assert r0.get("type") == "ready" and r1.get("type") == "ready", (r0, r1)
    print("BOTH ARE READY")

    time.sleep(0.1)

    start_evt.set()
    p0.join()
    p1.join()


if __name__ == "__main__":
    spawn_two_peers(max_tokens=4)
