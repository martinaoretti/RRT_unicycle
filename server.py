"""
server.py  -  FastAPI + WebSocket back-end for the RRT real-time visualiser.

Usage:
    cd rrt_ui
    python server.py
Then open http://localhost:8000
"""
from __future__ import annotations

import asyncio #gestione concorrenza(per real time)
import json
import sys
from pathlib import Path
from typing import Optional

# Windows: force SelectorEventLoop (avoids ProactorEventLoop WebSocket issues) 
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


@app.get("/")   #apre la pagina web
async def root():
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(html)


@app.get("/configs") #per dropdown nel frontend ostacoli
async def configs():
    return {"configs": list_configs()}


@app.websocket("/ws")   #canale comunicazione in tempo reale con il browser
async def ws_endpoint(ws: WebSocket):   
    await ws.accept()
    current_task: Optional[asyncio.Task] = None

    async def run_rrt(params: dict): #esegue eventi passo passo
        from rrt_streamer import rrt_stream
        import concurrent.futures

        config    = params.get("config", "random")  #tipo di mappa
        seed      = int(params.get("seed", 42))     #riproducibilità
        iters     = int(params.get("iterations", 4000)) #iterazioni RRT
        gbias     = float(params.get("goal_bias", 0.15)) #probabilità del goal
        batch_sz  = max(1, int(params.get("speed", 10))) #quanti nodi inviare insieme

        loop = asyncio.get_event_loop()
        gen  = rrt_stream(
            config_name=config,
            seed=seed,
            num_iterations=iters,
            goal_bias=gbias,
        )

        batch: list = []

        async def flush():  #manda dati al browser
            nonlocal batch
            if batch:
                await ws.send_text(json.dumps({"type": "batch", "events": batch}))
                batch = []

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool: #crea thread separato
                while True:
                    try:
                        event = await loop.run_in_executor(pool, next, gen) #prende il prossimo evento dal generatore RRT
                    except StopIteration:
                        await flush()
                        break

                    if event["type"] == "node": #controlla se è un nodo RRT
                        batch.append(event)     #lo accumola
                        if len(batch) >= batch_sz: #se batch pieno lo invia a browser
                            await flush()
                            await asyncio.sleep(0)
                    else: #se non è un nodo invia goal trovato, collisione, errori
                        await flush()
                        await ws.send_text(json.dumps(event))
                        await asyncio.sleep(0)

        #gestione errori
        except asyncio.CancelledError: #se l'utente ferma l'RRT informa il frontend
            await ws.send_text(json.dumps({"type": "cancelled"}))
        except Exception as exc:
            try:
                await ws.send_text(json.dumps({"type": "error", "msg": str(exc)})) 
            except Exception:
                pass

    try:
        while True:     #gestione dei messaggi dal browser 
            raw = await ws.receive_text() #riceve messaggi dal frontend
            msg = json.loads(raw)
            action = msg.get("action", "")

            if action == "start":   #avvia RRT
                if current_task and not current_task.done(): #sono gia in esecuzione-->lo ferma
                    current_task.cancel()
                    try:
                        await current_task  
                    except asyncio.CancelledError:
                        pass
                current_task = asyncio.create_task(run_rrt(msg)) #avvia nuovo RRT

            elif action == "stop": #stop manuale
                if current_task and not current_task.done():
                    current_task.cancel()
                    try:
                        await current_task
                    except asyncio.CancelledError:
                        pass
                await ws.send_text(json.dumps({"type": "cancelled"})) #aggiorna frontend

    except WebSocketDisconnect: #disconnessione 
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
    uvicorn.run( #avvia server FastAPI
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        ws="websockets",       # explicit websockets backend
        log_level="warning",   # suppress INFO noise
    )
