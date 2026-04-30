const state = {
  ws: null,
  running: false,
  treeNodes: [],
  pathNodes: [],
  pathTrace: [],
  pathIndex: 0,
  pathAnimAccum: 0,
  pathAnimLast: performance.now(),
  pathRunning: false,
  pathPose: null,
  obstacles: [],
  start: null,
  goal: null,
  workspace: [0,0,100,100],
  totalIter: 4000,
  curIter: 0,
  nodeCount: 0,
  lastT: performance.now(),
  nodeRate: 0,
  dirty: false,
  animFrame: null,
};

const cv      = document.getElementById('cv');
const ctx     = cv.getContext('2d');
const wrap    = document.getElementById('canvas-wrap');
const selCfg  = document.getElementById('sel-config');
const btnRun  = document.getElementById('btn-start');
const btnStop = document.getElementById('btn-stop');
const dot     = document.getElementById('dot');
const status  = document.getElementById('status-label');
const logEl   = document.getElementById('log');
const logLine = document.getElementById('log-line');
const wsStat  = document.getElementById('ws-status');

const stNodes = document.getElementById('st-nodes');
const stIter  = document.getElementById('st-iter');
const stPath  = document.getElementById('st-path');
const stFps   = document.getElementById('st-fps');
const progBar = document.getElementById('progress-bar');
const badge   = document.getElementById('result-badge');

const mkSlider = (id, lblId, fmt) => {
  const sl = document.getElementById(id);
  const lb = document.getElementById(lblId);
  lb.textContent = fmt(sl.value);
  sl.addEventListener('input', () => lb.textContent = fmt(sl.value));
  return sl;
};
const slIter  = mkSlider('sl-iter',  'lbl-iter',  v => (+v).toLocaleString());
const slSeed  = mkSlider('sl-seed',  'lbl-seed',  v => v);
const slGbias = mkSlider('sl-gbias', 'lbl-gbias', v => v + '%');
const slSpeed = mkSlider('sl-speed', 'lbl-speed', v => '×' + v);

let cvSize = 600, scale = 1, offX = 0, offY = 0;

function resizeCanvas() {
  const r  = wrap.getBoundingClientRect();
  const sz = Math.min(r.width, r.height) - 24;
  cvSize   = sz;
  cv.width = cv.height = sz;
  const [x0,,x1] = state.workspace;
  scale = sz / (x1 - x0);
  offX  = 0; offY = 0;
  state.dirty = true;
}

function wx(x) { return (x - state.workspace[0]) * scale; }
function wy(y) { return cvSize - (y - state.workspace[1]) * scale; }

function drawAll() {
  ctx.clearRect(0, 0, cvSize, cvSize);
  drawGrid();
  drawObstacles();
  drawTree();
  drawPath();
  drawPathMarkers();
  drawPoses();
  state.dirty = false;
}

function drawGrid() {
  ctx.strokeStyle = 'rgba(26,45,74,.6)';
  ctx.lineWidth = .5;
  const step = 10 * scale;
  for (let x = 0; x <= cvSize; x += step) {
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, cvSize); ctx.stroke();
  }
  for (let y = 0; y <= cvSize; y += step) {
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(cvSize, y); ctx.stroke();
  }
  ctx.fillStyle = 'rgba(74,96,128,.7)';
  ctx.font = '9px JetBrains Mono';
  ctx.textAlign = 'center';
  for (let v = 0; v <= 100; v += 20) {
    ctx.fillText(v, wx(v), cvSize - 3);
    ctx.fillText(v, 3, wy(v) + 3);
  }
}

function drawObstacles() {
  for (const pts of state.obstacles) {
    if (!pts.length) continue;
    ctx.beginPath();
    ctx.moveTo(wx(pts[0][0]), wy(pts[0][1]));
    for (let i = 1; i < pts.length; i++) ctx.lineTo(wx(pts[i][0]), wy(pts[i][1]));
    ctx.closePath();
    ctx.fillStyle   = 'rgba(255,160,50,.18)';
    ctx.strokeStyle = '#ffb347';
    ctx.lineWidth   = 1.2;
    ctx.fill();
    ctx.stroke();
  }
}

function drawTree() {
  ctx.strokeStyle = 'rgba(255,61,90,.22)';
  ctx.lineWidth   = .9;
  for (const nd of state.treeNodes) {
    if (!nd.arc || nd.arc.length < 2) continue;
    ctx.beginPath();
    ctx.moveTo(wx(nd.arc[0][0]), wy(nd.arc[0][1]));
    for (let i = 1; i < nd.arc.length; i++) ctx.lineTo(wx(nd.arc[i][0]), wy(nd.arc[i][1]));
    ctx.stroke();
  }
}

function drawPath() {
  if (!state.pathNodes.length) return;
  ctx.lineWidth = 2.8;
  ctx.strokeStyle = '#ff3d5a';
  ctx.shadowColor = 'rgba(255,61,90,.5)';
  ctx.shadowBlur  = 8;
  for (const nd of state.pathNodes) {
    if (!nd.arc || nd.arc.length < 2) continue;
    ctx.beginPath();
    ctx.moveTo(wx(nd.arc[0][0]), wy(nd.arc[0][1]));
    for (let i = 1; i < nd.arc.length; i++) ctx.lineTo(wx(nd.arc[i][0]), wy(nd.arc[i][1]));
    ctx.stroke();
  }
  ctx.shadowBlur = 0;
}

function drawPathMarkers() {
  if (!state.pathRunning || !state.pathNodes.length) return;
  ctx.fillStyle = 'rgba(255,255,255,0.75)';
  ctx.strokeStyle = 'rgba(255,61,90,0.9)';
  ctx.lineWidth = 1;
  for (const nd of state.pathNodes) {
    if (typeof nd.x !== 'number' || typeof nd.y !== 'number') continue;
    const px = wx(nd.x);
    const py = wy(nd.y);
    ctx.beginPath();
    ctx.arc(px, py, 2.2, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
  }
}

function drawRobot(pose, fill, stroke) {
  const [x, y, t] = pose;
  const px = wx(x), py = wy(y);
  const bverts = [[-1.5,-1],[ 1.5,-1],[ 1.5, 1],[-1.5, 1]];
  const wverts = bverts.map(([bx,by]) => [
    px + (bx * Math.cos(t) - by * Math.sin(t)) * scale,
    py - (bx * Math.sin(t) + by * Math.cos(t)) * scale,
  ]);
  ctx.beginPath();
  ctx.moveTo(wverts[0][0], wverts[0][1]);
  for (let i = 1; i < wverts.length; i++) ctx.lineTo(wverts[i][0], wverts[i][1]);
  ctx.closePath();
  ctx.fillStyle   = fill;
  ctx.strokeStyle = stroke;
  ctx.lineWidth   = 1.8;
  ctx.fill(); ctx.stroke();
  const ar = 3.5 * scale;
  ctx.beginPath();
  ctx.moveTo(px, py);
  ctx.lineTo(px + Math.cos(t) * ar, py - Math.sin(t) * ar);
  ctx.strokeStyle = stroke;
  ctx.lineWidth = 2;
  ctx.stroke();
}

function drawPoses() {
  if (state.start) drawRobot(state.start, 'rgba(30,80,30,.7)', '#39ff14');
  if (state.goal)  drawRobot(state.goal,  'rgba(20,40,100,.7)', '#4488ff');
  if (state.pathPose) drawRobot(state.pathPose, 'rgba(255,155,50,.88)', '#ff8c00');
}

function buildPathTrace(pathNodes) {
  const trace = [];
  for (const node of pathNodes) {
    const arc = node.arc || [];
    for (const pose of arc) {
      if (!trace.length || trace[trace.length - 1][0] !== pose[0] || trace[trace.length - 1][1] !== pose[1]) {
        trace.push(pose);
      }
    }
  }
  return trace;
}

function startPathAnimation(pathNodes, pathPoses) {
  state.pathNodes = pathNodes;
  state.pathTrace = pathPoses || buildPathTrace(pathNodes);
  state.pathIndex = 0;
  state.pathAnimAccum = 0;
  state.pathAnimLast = performance.now();
  state.pathRunning = state.pathTrace.length > 1;
  state.pathPose = state.pathTrace.length ? state.pathTrace[0] : null;
  state.dirty = true;
}

function advancePathAnimation(dt) {
  if (!state.pathRunning || !state.pathTrace.length) return;
  const stepMs = 35;
  state.pathAnimAccum += dt;
  while (state.pathAnimAccum >= stepMs && state.pathRunning) {
    state.pathAnimAccum -= stepMs;
    state.pathIndex += 1;
    if (state.pathIndex >= state.pathTrace.length - 1) {
      state.pathIndex = state.pathTrace.length - 1;
      state.pathRunning = false;
      break;
    }
  }
  state.pathPose = state.pathTrace[state.pathIndex];
  state.dirty = true;
}

function loop() {
  const now = performance.now();
  const dt  = now - state.pathAnimLast;
  state.pathAnimLast = now;
  if (state.pathRunning) advancePathAnimation(dt);
  state.animFrame = requestAnimationFrame(loop);
  if (state.dirty) drawAll();
}

function addLog(msg, cls = '') {
  const ts = new Date().toTimeString().slice(0,8);
  const el = document.createElement('div');
  el.className = 'log-entry';
  el.innerHTML = `<span class="log-ts">${ts}</span><span class="${cls}">${msg}</span>`;
  logEl.appendChild(el);
  logEl.scrollTop = logEl.scrollHeight;
  logLine.innerHTML = msg;
}

let _lastNodeCount = 0, _lastT = performance.now();
function updateStats() {
  stNodes.textContent = state.nodeCount;
  stIter.textContent  = state.curIter;
  const prog = state.totalIter > 0 ? state.curIter / state.totalIter : 0;
  progBar.style.width = Math.min(prog * 100, 100) + '%';
  const now = performance.now();
  const dt  = (now - _lastT) / 1000;
  if (dt >= .5) {
    const rate = (state.nodeCount - _lastNodeCount) / dt;
    stFps.textContent = Math.round(rate);
    _lastNodeCount = state.nodeCount;
    _lastT = now;
  }
}

function resetVis() {
  state.treeNodes = [];
  state.pathNodes = [];
  state.pathTrace = [];
  state.pathIndex = 0;
  state.pathAnimAccum = 0;
  state.pathAnimLast = performance.now();
  state.pathRunning = false;
  state.pathPose = null;
  state.obstacles = [];
  state.start     = null;
  state.goal      = null;
  state.curIter   = 0;
  state.nodeCount = 0;
  _lastNodeCount  = 0;
  stNodes.textContent = '0';
  stIter.textContent  = '0';
  stPath.textContent  = '—';
  stFps.textContent   = '—';
  progBar.style.width = '0%';
  badge.className = '';
  badge.style.display = 'none';
  badge.textContent = '';
  state.dirty = true;
}

function connectWS() {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  const ws = new WebSocket(`${proto}://${location.host}/ws`);
  state.ws = ws;

  ws.onopen = () => {
    wsStat.textContent = 'connected';
    wsStat.style.color = 'var(--cyan)';
    addLog('WebSocket connected', 'log-info');
    fetch('/configs').then(r => r.json()).then(d => {
      selCfg.innerHTML = '';
      d.configs.forEach(c => {
        const opt = document.createElement('option');
        opt.value = c; opt.textContent = c;
        if (c === 'random') opt.selected = true;
        selCfg.appendChild(opt);
      });
      addLog(`Loaded ${d.configs.length} configurations`, 'log-info');
    });
  };

  ws.onmessage = (e) => {
    const msg = JSON.parse(e.data);
    handleMessage(msg);
  };

  ws.onclose = () => {
    wsStat.textContent = 'disconnected';
    wsStat.style.color = 'var(--red)';
    addLog('WebSocket disconnected — retrying in 2s…', 'log-err');
    setRunning(false);
    setTimeout(connectWS, 2000);
  };

  ws.onerror = () => {
    addLog('WebSocket error', 'log-err');
  };
}

function handleMessage(msg) {
  switch (msg.type) {
    case 'init': {
      state.obstacles  = msg.obstacles;
      state.start      = msg.start;
      state.goal       = msg.goal;
      state.workspace  = msg.workspace;
      resizeCanvas();
      addLog(`Init: config loaded, ${msg.obstacles.length} obstacle(s)`, 'log-info');
      state.dirty = true;
      break;
    }
    case 'batch': {
      for (const ev of msg.events) {
        state.treeNodes.push(ev.node);
        state.nodeCount = ev.node.idx + 1;
        state.curIter   = ev.iter;
      }
      updateStats();
      state.dirty = true;
      break;
    }
    case 'node': {
      state.treeNodes.push(msg.node);
      state.nodeCount = msg.node.idx + 1;
      state.curIter   = msg.iter;
      updateStats();
      state.dirty = true;
      break;
    }
    case 'path': {
      const pathNodes = msg.path_nodes || msg.nodes || [];
      const pathPoses = msg.path_poses || null;
      stPath.textContent = msg.length;
      addLog(`Path found! ${msg.length} nodes`, 'log-ok');
      startPathAnimation(pathNodes, pathPoses);
      state.dirty = true;
      break;
    }
    case 'done': {
      if (msg.success) {
        badge.className = 'success';
        badge.textContent = '✓ PATH FOUND';
        badge.style.display = 'block';
        addLog(`Done — SUCCESS  tree=${msg.tree_size} iter=${msg.iterations}`, 'log-ok');
        dot.className = 'top-dot';
        status.textContent = 'SUCCESS';
      } else {
        badge.className = 'fail';
        badge.textContent = '✗ NO PATH FOUND';
        badge.style.display = 'block';
        addLog(`Done — FAILED  tree=${msg.tree_size} after ${msg.iterations} iterations`, 'log-err');
        dot.className = 'top-dot';
        status.textContent = 'FAILED';
      }
      setRunning(false);
      progBar.style.width = '100%';
      updateStats();
      break;
    }
    case 'cancelled': {
      addLog('Run cancelled', 'log-warn');
      setRunning(false);
      break;
    }
    case 'error': {
      addLog('Error: ' + msg.msg, 'log-err');
      setRunning(false);
      break;
    }
  }
}

function setRunning(v) {
  state.running = v;
  btnRun.disabled  = v;
  btnStop.disabled = !v;
  if (v) {
    dot.className = 'top-dot active';
    status.textContent = 'RUNNING';
  }
}

btnRun.addEventListener('click', () => {
  if (!state.ws || state.ws.readyState !== 1) {
    addLog('Not connected to server', 'log-err'); return;
  }
  resetVis();
  setRunning(true);
  state.totalIter = +slIter.value;
  const params = {
    action:     'start',
    config:     selCfg.value,
    seed:       +slSeed.value,
    iterations: +slIter.value,
    goal_bias:  +slGbias.value / 100,
    speed:      +slSpeed.value,
  };
  addLog(`Starting: config=${params.config} seed=${params.seed} iter=${params.iterations}`, 'log-info');
  state.ws.send(JSON.stringify(params));
});

btnStop.addEventListener('click', () => {
  if (state.ws) state.ws.send(JSON.stringify({ action: 'stop' }));
});

const ro = new ResizeObserver(() => { resizeCanvas(); });
ro.observe(wrap);

resizeCanvas();
loop();
connectWS();
