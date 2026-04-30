"""
server.py  -  FastAPI + WebSocket back-end for the RRT real-time visualiser.

Usage:
    cd rrt_ui
    python server.py
Then open http://localhost:8000
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Optional

# Windows: force SelectorEventLoop (avoids ProactorEventLoop WS issues)
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from obstacles import list_configs

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI()
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def root():
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(html)


@app.get("/configs")
async def configs():
    return {"configs": list_configs()}


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    current_task: Optional[asyncio.Task] = None

    async def run_rrt(params: dict):
        from rrt_streamer import rrt_stream
        import concurrent.futures

        config    = params.get("config", "random")
        seed      = int(params.get("seed", 42))
        iters     = int(params.get("iterations", 4000))
        gbias     = float(params.get("goal_bias", 0.15))
        batch_sz  = max(1, int(params.get("speed", 10)))

        loop = asyncio.get_event_loop()
        gen  = rrt_stream(
            config_name=config,
            seed=seed,
            num_iterations=iters,
            goal_bias=gbias,
        )

        batch: list = []

        async def flush():
            nonlocal batch
            if batch:
                await ws.send_text(json.dumps({"type": "batch", "events": batch}))
                batch = []

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                while True:
                    try:
                        event = await loop.run_in_executor(pool, next, gen)
                    except StopIteration:
                        await flush()
                        break

                    if event["type"] == "node":
                        batch.append(event)
                        if len(batch) >= batch_sz:
                            await flush()
                            await asyncio.sleep(0)
                    else:
                        await flush()
                        await ws.send_text(json.dumps(event))
                        await asyncio.sleep(0)

        except asyncio.CancelledError:
            await ws.send_text(json.dumps({"type": "cancelled"}))
        except Exception as exc:
            try:
                await ws.send_text(json.dumps({"type": "error", "msg": str(exc)}))
            except Exception:
                pass

    try:
        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)
            action = msg.get("action", "")

            if action == "start":
                if current_task and not current_task.done():
                    current_task.cancel()
                    try:
                        await current_task
                    except asyncio.CancelledError:
                        pass
                current_task = asyncio.create_task(run_rrt(msg))

            elif action == "stop":
                if current_task and not current_task.done():
                    current_task.cancel()
                    try:
                        await current_task
                    except asyncio.CancelledError:
                        pass
                await ws.send_text(json.dumps({"type": "cancelled"}))

    except WebSocketDisconnect:
        if current_task and not current_task.done():
            current_task.cancel()
    except Exception:
        if current_task and not current_task.done():
            current_task.cancel()


if __name__ == "__main__":
    print()
    print("  RRT Unicycle  -  Real-time Visualiser")
    print("  ----------------------------------------")
    print("  Open ->  http://localhost:8000")
    print()
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        ws="websockets",       # explicit websockets backend
        log_level="warning",   # suppress INFO noise
    )
