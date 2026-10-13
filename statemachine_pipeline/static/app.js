/* State Diagram Studio: editor for the canonical state-machine JSON.
 *
 * The canonical JSON (`model`) is the only state. Every edit goes through
 * commit(): it pushes undo history, re-renders the graph, JSON and inspector,
 * schedules server-side validation and marks generated code stale.
 * `Ops` holds the pure graph operations (unit-tested with Node).
 */
'use strict';

const Ops = (() => {
  const clone = (x) => JSON.parse(JSON.stringify(x));

  function nextId(items, prefix) {
    const ids = new Set(items.map((x) => x.id));
    let i = 0;
    while (ids.has(prefix + i)) i += 1;
    return prefix + i;
  }

  function emptyMachine() {
    return { schema_version: '1.1', diagram_id: 'untitled', states: [], transitions: [],
      metadata: { source: 'user', warnings: [] }, network: null, uncertainties: [], source_image: null };
  }

  function normalize(m) {
    const out = clone(m);
    out.states = out.states || [];
    out.transitions = out.transitions || [];
    out.uncertainties = out.uncertainties || [];
    out.metadata = out.metadata || { warnings: [] };
    for (const s of out.states) {
      s.type = s.type || 'normal';
      if (s.label === undefined) s.label = null;
    }
    for (const t of out.transitions) {
      for (const k of ['event', 'guard', 'action', 'label', 'network']) if (t[k] === undefined) t[k] = null;
      t.direction = t.direction || 'unknown';
      t.kind = t.kind || 'state_change';
      t.points = t.points || [];
    }
    return out;
  }

  function addState(m, pos, label = null) {
    const out = clone(m);
    const id = nextId(out.states, 'S');
    out.states.push({ id, label, type: out.states.length ? 'normal' : 'initial', position: { x: pos.x, y: pos.y },
      confidence: null, bbox: null, shape: 'circle', label_confidence: null, provenance: 'user' });
    return [out, id];
  }

  function updateState(m, id, changes) {
    const out = clone(m);
    if (changes.type === 'initial') {
      for (const s of out.states) if (s.id !== id && s.type === 'initial') s.type = 'normal';
    }
    out.states = out.states.map((s) => (s.id === id ? { ...s, ...changes, provenance: 'user' } : s));
    return out;
  }

  function deleteState(m, id) {
    const out = clone(m);
    const removed = new Set(out.transitions.filter((t) => t.source === id || t.target === id).map((t) => t.id));
    out.states = out.states.filter((s) => s.id !== id);
    out.transitions = out.transitions.filter((t) => !removed.has(t.id));
    out.uncertainties = (out.uncertainties || []).filter((u) => u.element_id !== id && !removed.has(u.element_id));
    if (out.network) {
      out.network.devices = out.network.devices.map((d) => ({ ...d, state_ids: (d.state_ids || []).filter((s) => s !== id) }));
    }
    return out;
  }

  function addTransition(m, source, target, fields = {}) {
    const out = clone(m);
    const id = nextId(out.transitions, 'T');
    out.transitions.push({ id, source, target, event: null, guard: null, action: null, label: null,
      confidence: null, direction: source === target ? 'loop' : 'forward', points: [], label_bbox: null,
      label_confidence: null, kind: 'state_change', network: null, provenance: 'user', ...fields });
    return [out, id];
  }

  function updateTransition(m, id, changes) {
    const out = clone(m);
    out.transitions = out.transitions.map((t) => {
      if (t.id !== id) return t;
      const next = { ...t, ...changes, provenance: 'user' };
      if (('source' in changes || 'target' in changes)) {
        next.points = [];  // the traced path no longer describes the edited edge
        if (next.source === next.target) next.direction = 'loop';
        else if (next.direction === 'loop') next.direction = 'forward';
      }
      return next;
    });
    return out;
  }

  function reverseTransition(m, id) {
    const t = m.transitions.find((x) => x.id === id);
    if (!t) return m;
    const out = updateTransition(m, id, { source: t.target, target: t.source });
    const edited = out.transitions.find((x) => x.id === id);
    if (edited.direction === 'unknown') edited.direction = 'forward';
    out.uncertainties = (out.uncertainties || []).filter((u) => !(u.element_id === id && u.field === 'direction'));
    return out;
  }

  function deleteTransition(m, id) {
    const out = clone(m);
    out.transitions = out.transitions.filter((t) => t.id !== id);
    out.uncertainties = (out.uncertainties || []).filter((u) => u.element_id !== id);
    return out;
  }

  function resolveUncertainty(m, index) {
    const out = clone(m);
    out.uncertainties.splice(index, 1);
    return out;
  }

  function ensureNetwork(m) {
    const out = clone(m);
    if (!out.network) out.network = { devices: [], links: [] };
    return out;
  }

  function addDevice(m, label, stateIds = []) {
    const out = ensureNetwork(m);
    const id = nextId(out.network.devices, 'D');
    out.network.devices.push({ id, label: label || null, role: null, state_ids: stateIds, position: null });
    return [out, id];
  }

  function addLink(m, source, target, medium = null) {
    const out = ensureNetwork(m);
    const id = nextId(out.network.links, 'L');
    out.network.links.push({ id, source, target, medium, label: null });
    return [out, id];
  }

  function deleteDevice(m, id) {
    const out = ensureNetwork(m);
    out.network.devices = out.network.devices.filter((d) => d.id !== id);
    out.network.links = out.network.links.filter((l) => l.source !== id && l.target !== id);
    if (!out.network.devices.length && !out.network.links.length) out.network = null;
    return out;
  }

  function deleteLink(m, id) {
    const out = ensureNetwork(m);
    out.network.links = out.network.links.filter((l) => l.id !== id);
    if (!out.network.devices.length && !out.network.links.length) out.network = null;
    return out;
  }

  /* Layered layout: BFS distance from the initial state sets the column. */
  function autoLayout(m) {
    const out = clone(m);
    if (!out.states.length) return out;
    const start = (out.states.find((s) => s.type === 'initial') || out.states[0]).id;
    const depth = { [start]: 0 };
    const queue = [start];
    while (queue.length) {
      const cur = queue.shift();
      for (const t of out.transitions) {
        if (t.source === cur && !(t.target in depth)) { depth[t.target] = depth[cur] + 1; queue.push(t.target); }
      }
    }
    const maxDepth = Math.max(0, ...Object.values(depth));
    const columns = {};
    for (const s of out.states) {
      const d = s.id in depth ? depth[s.id] : maxDepth + 1;
      (columns[d] = columns[d] || []).push(s);
    }
    for (const [d, items] of Object.entries(columns)) {
      items.forEach((s, i) => {
        s.position = { x: 110 + Number(d) * 190, y: 110 + i * 150 + (Number(d) % 2) * 40 };
        s.bbox = null;
      });
    }
    for (const t of out.transitions) t.points = [];
    return out;
  }

  function transitionText(t) {
    const parts = [];
    if (t.event) parts.push(t.event);
    if (t.guard) parts.push('[' + t.guard + ']');
    if (t.action) parts.push('/ ' + t.action);
    if (parts.length) return parts.join(' ');
    return t.label || '';
  }

  /* Edge geometry for drawing: quadratic curves, offsets for parallel edges, loops. */
  function edgeGeometry(m, radiusOf) {
    const pos = Object.fromEntries(m.states.map((s) => [s.id, s.position || { x: 80, y: 80 }]));
    const groups = {};
    for (const t of m.transitions) {
      const key = [t.source, t.target].sort().join('|');
      (groups[key] = groups[key] || []).push(t);
    }
    const result = {};
    for (const items of Object.values(groups)) {
      const loops = items.filter((t) => t.source === t.target);
      loops.forEach((t, i) => {
        const p = pos[t.source];
        const r = radiusOf(t.source);
        // Point loops away from the state's neighbours so they do not cover other edges.
        let sx = 0, sy = 0;
        for (const o of m.transitions) {
          const other = o.source === t.source ? o.target : o.target === t.source ? o.source : null;
          if (!other || other === t.source || !pos[other]) continue;
          const len = Math.hypot(pos[other].x - p.x, pos[other].y - p.y) || 1;
          sx += (pos[other].x - p.x) / len; sy += (pos[other].y - p.y) / len;
        }
        const base = Math.hypot(sx, sy) > 0.01 ? Math.atan2(-sy, -sx) : -Math.PI / 2;
        const angle = base + i * (Math.PI / 3);
        const a1 = angle - 0.45, a2 = angle + 0.45;
        const s = { x: p.x + r * Math.cos(a1), y: p.y + r * Math.sin(a1) };
        const e = { x: p.x + r * Math.cos(a2), y: p.y + r * Math.sin(a2) };
        const reach = r * 2.3 + 18;
        const c1 = { x: p.x + reach * Math.cos(a1 - 0.25), y: p.y + reach * Math.sin(a1 - 0.25) };
        const c2 = { x: p.x + reach * Math.cos(a2 + 0.25), y: p.y + reach * Math.sin(a2 + 0.25) };
        result[t.id] = { d: `M${s.x},${s.y} C${c1.x},${c1.y} ${c2.x},${c2.y} ${e.x},${e.y}`,
          label: { x: p.x + (reach * 0.8) * Math.cos(angle), y: p.y + (reach * 0.8) * Math.sin(angle) } };
      });
      const edges = items.filter((t) => t.source !== t.target);
      edges.forEach((t, i) => {
        const a = pos[t.source], b = pos[t.target];
        // Offsets are measured in a canonical orientation so opposite edges separate.
        const canonical = t.source < t.target;
        const n = edges.length;
        let offset = (i - (n - 1) / 2) * 46;
        if (n === 1) offset = 0;
        const dx = b.x - a.x, dy = b.y - a.y;
        const len = Math.hypot(dx, dy) || 1;
        let nx = -dy / len, ny = dx / len;
        if (!canonical) { nx = -nx; ny = -ny; }
        const mid = { x: (a.x + b.x) / 2 + nx * offset, y: (a.y + b.y) / 2 + ny * offset };
        const ctrl = { x: (a.x + b.x) / 2 + nx * offset * 2, y: (a.y + b.y) / 2 + ny * offset * 2 };
        const ra = radiusOf(t.source), rb = radiusOf(t.target);
        const sa = Math.atan2(ctrl.y - a.y, ctrl.x - a.x), sb = Math.atan2(ctrl.y - b.y, ctrl.x - b.x);
        const s = { x: a.x + ra * Math.cos(sa), y: a.y + ra * Math.sin(sa) };
        const e = { x: b.x + (rb + 2) * Math.cos(sb), y: b.y + (rb + 2) * Math.sin(sb) };
        const d = offset === 0 ? `M${s.x},${s.y} L${e.x},${e.y}` : `M${s.x},${s.y} Q${ctrl.x},${ctrl.y} ${e.x},${e.y}`;
        result[t.id] = { d, label: { x: mid.x + nx * 10, y: mid.y + ny * 10 } };
      });
    }
    return result;
  }

  class History {
    constructor(limit = 200) { this.undoStack = []; this.redoStack = []; this.limit = limit; }
    push(snapshot) { this.undoStack.push(clone(snapshot)); if (this.undoStack.length > this.limit) this.undoStack.shift(); this.redoStack = []; }
    undo(current) { if (!this.undoStack.length) return null; this.redoStack.push(clone(current)); return this.undoStack.pop(); }
    redo(current) { if (!this.redoStack.length) return null; this.undoStack.push(clone(current)); return this.redoStack.pop(); }
    get canUndo() { return this.undoStack.length > 0; }
    get canRedo() { return this.redoStack.length > 0; }
  }

  return { clone, nextId, emptyMachine, normalize, addState, updateState, deleteState, addTransition,
    updateTransition, reverseTransition, deleteTransition, resolveUncertainty, addDevice, addLink,
    deleteDevice, deleteLink, autoLayout, transitionText, edgeGeometry, History };
})();

if (typeof module !== 'undefined') module.exports = Ops;

if (typeof document !== 'undefined') (() => {
  const SVG = 'http://www.w3.org/2000/svg';
  const $ = (id) => document.getElementById(id);
  const el = (tag, attrs = {}, text) => {
    const node = document.createElementNS(SVG, tag);
    for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
    if (text !== undefined) node.textContent = text;
    return node;
  };
  const html = (tag, attrs = {}, children = []) => {
    const node = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (v === null || v === undefined) continue;
      if (k === 'text') node.textContent = v;
      else if (k.startsWith('on')) node.addEventListener(k.slice(2), v);
      else node.setAttribute(k, v);
    }
    for (const c of [].concat(children)) if (c) node.append(c);
    return node;
  };

  let model = Ops.emptyMachine();
  const history = new Ops.History();
  let selection = null;           // {kind: 'state'|'transition', id}
  let validation = { valid: true, errors: [], warnings: [], element_errors: {}, element_warnings: {} };
  let generated = null;           // last /api/generate response
  let generatedFor = null;        // JSON string the code was generated from
  let session = { has_image: false, detections: [] };
  let view = { x: 0, y: 0, k: 1 };
  let connectFrom = null;
  let jsonDirty = false;
  let validateTimer = null;

  // ---------------------------------------------------------------- status
  function status(message, kind = '') { const s = $('status'); s.textContent = message; s.className = kind; s.title = message; }

  async function api(path, body, options = {}) {
    const init = body === undefined ? {} : { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) };
    const response = await fetch(path, { ...init, ...options });
    let data = null;
    try { data = await response.json(); } catch (_) { data = null; }
    if (!response.ok) {
      const errors = (data && data.errors) || [`${response.status} ${response.statusText}`];
      const error = new Error(errors.slice(0, 3).join('; '));
      error.data = data;
      throw error;
    }
    return data;
  }

  // ---------------------------------------------------------------- commit
  function commit(next, message) {
    history.push(model);
    model = next;
    afterChange(message);
  }

  function afterChange(message) {
    if (selection) {
      const pool = selection.kind === 'state' ? model.states : model.transitions;
      if (!pool.some((x) => x.id === selection.id)) selection = null;
    }
    renderAll();
    scheduleValidation();
    if (message) status(message);
  }

  function undo() { const prev = history.undo(model); if (prev) { model = prev; afterChange('Undone'); } }
  function redo() { const next = history.redo(model); if (next) { model = next; afterChange('Redone'); } }

  function scheduleValidation() {
    clearTimeout(validateTimer);
    validateTimer = setTimeout(async () => {
      try {
        validation = await api('/api/validate', model);
      } catch (e) {
        validation = { valid: false, errors: [e.message], warnings: [], element_errors: {}, element_warnings: {} };
      }
      renderGraph();
      renderIssues();
    }, 200);
  }

  // ---------------------------------------------------------------- geometry helpers
  function radiusOf(id) {
    const s = model.states.find((x) => x.id === id);
    if (!s) return 30;
    if (s.bbox) return Math.max(22, Math.min(70, Math.max(s.bbox[2] - s.bbox[0], s.bbox[3] - s.bbox[1]) / 2));
    return 32;
  }
  function svgPoint(evt) {
    const svg = $('graph');
    const rect = svg.getBoundingClientRect();
    return { x: (evt.clientX - rect.left - view.x) / view.k, y: (evt.clientY - rect.top - view.y) / view.k };
  }
  function uncertainFor(id) { return (model.uncertainties || []).filter((u) => u.element_id === id); }
  function classesFor(id) {
    const cls = [];
    if (validation.element_errors && validation.element_errors[id]) cls.push('err');
    else if (uncertainFor(id).length) cls.push('warn');
    if (selection && selection.id === id) cls.push('selected');
    return cls;
  }

  // ---------------------------------------------------------------- graph rendering
  function applyView() { $('viewport').setAttribute('transform', `translate(${view.x},${view.y}) scale(${view.k})`); }

  function renderGraph() {
    const root = $('viewport');
    root.replaceChildren();
    applyView();
    const geometry = Ops.edgeGeometry(model, radiusOf);
    const byId = Object.fromEntries(model.states.map((s) => [s.id, s]));

    // Network devices: dashed hulls around their states, links between hull centres.
    const hulls = {};
    if (model.network) {
      for (const d of model.network.devices || []) {
        const members = (d.state_ids || []).map((id) => byId[id]).filter(Boolean);
        let box;
        if (members.length) {
          const xs = members.flatMap((s) => [s.position.x - radiusOf(s.id), s.position.x + radiusOf(s.id)]);
          const ys = members.flatMap((s) => [s.position.y - radiusOf(s.id), s.position.y + radiusOf(s.id)]);
          box = { x1: Math.min(...xs) - 22, y1: Math.min(...ys) - 30, x2: Math.max(...xs) + 22, y2: Math.max(...ys) + 22 };
        } else {
          const p = d.position || { x: 60 + 120 * Object.keys(hulls).length, y: 30 };
          box = { x1: p.x - 45, y1: p.y - 20, x2: p.x + 45, y2: p.y + 20 };
        }
        hulls[d.id] = box;
      }
      for (const l of model.network.links || []) {
        const a = hulls[l.source], b = hulls[l.target];
        if (!a || !b) continue;
        root.append(el('line', { class: 'link', x1: (a.x1 + a.x2) / 2, y1: (a.y1 + a.y2) / 2, x2: (b.x1 + b.x2) / 2, y2: (b.y1 + b.y2) / 2 }));
      }
      for (const d of model.network.devices || []) {
        const box = hulls[d.id];
        root.append(el('rect', { class: 'device', x: box.x1, y: box.y1, width: box.x2 - box.x1, height: box.y2 - box.y1, rx: 10 }));
        root.append(el('text', { class: 'device-label', x: box.x1 + 6, y: box.y1 + 13 }, (d.label || d.id) + (d.role ? ` (${d.role})` : '')));
      }
    }

    for (const t of model.transitions) {
      const g = geometry[t.id];
      if (!g || !byId[t.source] || !byId[t.target]) continue;
      const cls = classesFor(t.id);
      const net = t.kind === 'network_event';
      const unknown = t.direction === 'unknown';
      const group = el('g', { class: 'transition', 'data-id': t.id });
      const path = el('path', { d: g.d, class: ['edge', net ? 'net' : '', unknown ? 'warn' : '', ...cls].join(' '),
        'marker-end': `url(#${cls.includes('warn') || unknown ? 'arrow-warn' : net ? 'arrow-net' : 'arrow'})` });
      const hit = el('path', { d: g.d, class: 'edge-hit', tabindex: '0', role: 'button',
        'aria-label': `Transition ${t.id} from ${t.source} to ${t.target}` });
      const text = Ops.transitionText(t);
      const label = el('text', { x: g.label.x, y: g.label.y, class: 'edge-label' + (text ? '' : ' missing'), 'text-anchor': 'middle' },
        (text || `(${t.id}: no label)`) + (unknown ? ' ?' : ''));
      for (const node of [hit, label]) {
        node.addEventListener('click', (e) => { e.stopPropagation(); select('transition', t.id); });
        node.addEventListener('keydown', (e) => { if (e.key === 'Enter') select('transition', t.id); });
      }
      group.append(path, hit, label);
      root.append(group);
    }

    for (const s of model.states) {
      const p = s.position || { x: 80, y: 80 };
      const r = radiusOf(s.id);
      const cls = ['state', ...classesFor(s.id)];
      if (connectFrom === s.id) cls.push('pending');
      const g = el('g', { class: cls.join(' '), 'data-id': s.id, tabindex: '0', role: 'button',
        'aria-label': `State ${s.label || s.id}${s.type !== 'normal' ? ', ' + s.type : ''}` });
      if (s.type === 'initial') {
        g.append(el('circle', { class: 'initial-dot', cx: p.x - r - 34, cy: p.y, r: 5 }));
        g.append(el('line', { class: 'initial-line', x1: p.x - r - 30, y1: p.y, x2: p.x - r - 3, y2: p.y, 'marker-end': 'url(#arrow)' }));
      }
      if (s.shape === 'rounded_rect') {
        const w = s.bbox ? (s.bbox[2] - s.bbox[0]) / 2 : r * 1.3, h = s.bbox ? (s.bbox[3] - s.bbox[1]) / 2 : r * 0.8;
        g.append(el('rect', { class: 'state-shape', x: p.x - w, y: p.y - h, width: 2 * w, height: 2 * h, rx: 10 }));
      } else {
        g.append(el('circle', { class: 'state-shape', cx: p.x, cy: p.y, r }));
      }
      if (s.type === 'final') g.append(el('circle', { class: 'state-inner', cx: p.x, cy: p.y, r: Math.max(8, r - 6) }));
      const name = s.label || '(unnamed)';
      const inner = (s.shape === 'rounded_rect' ? 2.4 : 1.7) * r;
      const size = Math.max(8, Math.min(13, inner / (0.58 * Math.max(1, name.length))));
      g.append(el('text', { class: 'state-label', x: p.x, y: p.y - 2, 'font-size': size.toFixed(1), style: `font-size:${size.toFixed(1)}px` }, name));
      g.append(el('text', { class: 'state-id', x: p.x, y: p.y + 14 }, s.id));
      attachStateHandlers(g, s.id);
      root.append(g);
    }
    $('graph-meta').textContent = `${model.states.length} states · ${model.transitions.length} transitions` +
      (model.network ? ` · ${model.network.devices.length} devices` : '');
  }

  function attachStateHandlers(g, id) {
    let dragging = null;
    g.addEventListener('pointerdown', (e) => {
      if (e.button !== 0) return;
      e.stopPropagation();
      if ($('connect').getAttribute('aria-pressed') === 'true') return;
      const s = model.states.find((x) => x.id === id);
      dragging = { start: svgPoint(e), origin: { ...(s.position || { x: 80, y: 80 }) }, moved: false, snapshot: Ops.clone(model) };
      g.setPointerCapture(e.pointerId);
    });
    g.addEventListener('pointermove', (e) => {
      if (!dragging) return;
      const p = svgPoint(e);
      const dx = p.x - dragging.start.x, dy = p.y - dragging.start.y;
      if (!dragging.moved && Math.hypot(dx, dy) < 3) return;
      dragging.moved = true;
      model.states = model.states.map((x) => (x.id === id ? { ...x, position: { x: dragging.origin.x + dx, y: dragging.origin.y + dy } } : x));
      // Drawn paths/boxes describe the image, not the edited layout.
      model.transitions = model.transitions.map((t) => (t.source === id || t.target === id ? { ...t, points: [] } : t));
      renderGraph();
    });
    g.addEventListener('pointerup', (e) => {
      if (!dragging) return;
      const moved = dragging.moved;
      const snapshot = dragging.snapshot;
      dragging = null;
      if (moved) {
        history.push(snapshot);
        afterChange(`Moved ${id}`);
      }
    });
    g.addEventListener('click', (e) => {
      e.stopPropagation();
      if ($('connect').getAttribute('aria-pressed') === 'true') { connectClick(id); return; }
      select('state', id);
    });
    g.addEventListener('dblclick', (e) => { e.stopPropagation(); renameState(id); });
    g.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') select('state', id);
      const step = e.shiftKey ? 20 : 5;
      const moves = { ArrowLeft: [-step, 0], ArrowRight: [step, 0], ArrowUp: [0, -step], ArrowDown: [0, step] };
      if (moves[e.key]) {
        e.preventDefault();
        const s = model.states.find((x) => x.id === id);
        const p = s.position || { x: 80, y: 80 };
        commit(Ops.updateState(model, id, { position: { x: p.x + moves[e.key][0], y: p.y + moves[e.key][1] } }), `Moved ${id}`);
      }
    });
  }

  function connectClick(id) {
    if (!connectFrom) { connectFrom = id; renderGraph(); status(`Choose the target state for a transition from ${id}`); return; }
    const [next, tid] = Ops.addTransition(model, connectFrom, id);
    connectFrom = null;
    setConnect(false);
    commit(next, `Added ${tid}`);
    select('transition', tid);
  }

  function setConnect(on) {
    $('connect').setAttribute('aria-pressed', String(on));
    $('graph').classList.toggle('connecting', on);
    if (!on) connectFrom = null;
  }

  function select(kind, id) {
    selection = id ? { kind, id } : null;
    renderGraph();
    renderInspector();
    showTab('inspector');
    highlightOverlay();
  }

  function renameState(id) {
    const s = model.states.find((x) => x.id === id);
    const label = prompt(`Name for state ${id}`, s.label || '');
    if (label !== null) commit(Ops.updateState(model, id, { label: label.trim() || null }), `Renamed ${id}`);
  }

  // ---------------------------------------------------------------- inspector
  function field(labelText, input) { return html('div', {}, [html('label', { text: labelText, for: input.id || '' }), input]); }
  function textInput(id, value, onChange, placeholder = '') {
    const input = html('input', { id, value: value == null ? '' : value, placeholder });
    input.addEventListener('change', () => onChange(input.value.trim() === '' ? null : input.value.trim()));
    return input;
  }
  function selectInput(id, options, value, onChange) {
    const s = html('select', { id });
    for (const [v, label] of options) {
      const o = html('option', { value: v, text: label });
      if (String(v) === String(value)) o.selected = true;
      s.append(o);
    }
    s.addEventListener('change', () => onChange(s.value));
    return s;
  }
  function stateOptions() { return model.states.map((s) => [s.id, `${s.label || '(unnamed)'} [${s.id}]`]); }

  function uncertaintyList(id) {
    const list = html('div');
    (model.uncertainties || []).forEach((u, index) => {
      if (u.element_id !== id) return;
      list.append(html('div', { class: 'note' }, [
        html('strong', { text: `Review ${u.field}: ` }), document.createTextNode(u.reason),
        u.candidates && u.candidates.length ? html('div', { class: 'hint', text: 'Candidates: ' + u.candidates.join(', ') }) : null,
        html('div', { class: 'row-buttons' }, [html('button', { text: 'Mark reviewed', onclick: () => commit(Ops.resolveUncertainty(model, index), 'Marked as reviewed') })]),
      ]));
    });
    return list;
  }

  function renderInspector() {
    const box = $('inspector');
    box.replaceChildren();
    if (!selection) {
      box.append(html('p', { class: 'hint', text: 'Select a state or transition. Double-click the canvas to add a state; use Connect to draw a transition (click the same state twice for a self-loop).' }));
      return;
    }
    if (selection.kind === 'state') {
      const s = model.states.find((x) => x.id === selection.id);
      if (!s) return;
      const errs = (validation.element_errors || {})[s.id] || [];
      box.append(html('h3', { text: `State ${s.id}` }));
      errs.forEach((e) => box.append(html('div', { class: 'note err', text: e })));
      box.append(uncertaintyList(s.id));
      box.append(field('Name', textInput('f-label', s.label, (v) => commit(Ops.updateState(model, s.id, { label: v }), `Renamed ${s.id}`))));
      box.append(html('div', { class: 'grid2' }, [
        field('Type', selectInput('f-type', [['normal', 'normal'], ['initial', 'initial'], ['final', 'final']], s.type,
          (v) => commit(Ops.updateState(model, s.id, { type: v, shape: v === 'final' && s.shape === 'circle' ? 'double_circle' : (s.shape === 'double_circle' && v !== 'final' ? 'circle' : s.shape) }), `Changed ${s.id} type`))),
        field('Shape', selectInput('f-shape', [['circle', 'circle'], ['double_circle', 'double circle'], ['rounded_rect', 'rounded rect'], ['unknown', 'unknown']], s.shape || 'unknown',
          (v) => commit(Ops.updateState(model, s.id, { shape: v }), `Changed ${s.id} shape`))),
      ]));
      if (s.label_confidence != null) box.append(html('p', { class: 'hint', text: `OCR confidence ${(s.label_confidence * 100).toFixed(0)}%` }));
      const outgoing = model.transitions.filter((t) => t.source === s.id);
      const incoming = model.transitions.filter((t) => t.target === s.id && t.source !== s.id);
      box.append(html('p', { class: 'hint', text: `${outgoing.length} outgoing, ${incoming.length} incoming` }));
      box.append(html('div', { class: 'row-buttons' }, [
        html('button', { text: 'Add self-loop', onclick: () => { const [next, tid] = Ops.addTransition(model, s.id, s.id); commit(next, `Added self-loop ${tid}`); select('transition', tid); } }),
        html('button', { text: 'Add transition from here', onclick: () => { setConnect(true); connectFrom = s.id; renderGraph(); status(`Choose the target for a transition from ${s.id}`); } }),
        html('button', { class: 'danger', text: 'Delete state', onclick: () => { if (confirm(`Delete ${s.id} and its transitions?`)) { commit(Ops.deleteState(model, s.id), `Deleted ${s.id}`); select(null); } } }),
      ]));
      return;
    }
    const t = model.transitions.find((x) => x.id === selection.id);
    if (!t) return;
    const set = (changes, message) => commit(Ops.updateTransition(model, t.id, changes), message || `Updated ${t.id}`);
    box.append(html('h3', { text: `Transition ${t.id}` }));
    ((validation.element_errors || {})[t.id] || []).forEach((e) => box.append(html('div', { class: 'note err', text: e })));
    box.append(uncertaintyList(t.id));
    box.append(html('div', { class: 'grid2' }, [
      field('Source', selectInput('f-source', stateOptions(), t.source, (v) => set({ source: v }))),
      field('Target', selectInput('f-target', stateOptions(), t.target, (v) => set({ target: v }))),
    ]));
    box.append(html('div', { class: 'row-buttons' }, [
      html('button', { text: 'Reverse direction', disabled: t.source === t.target ? '' : null, onclick: () => commit(Ops.reverseTransition(model, t.id), `Reversed ${t.id}`) }),
      t.direction === 'unknown' && t.source !== t.target
        ? html('button', { text: 'Confirm direction', onclick: () => { const next = Ops.updateTransition(model, t.id, { direction: 'forward' }); next.uncertainties = next.uncertainties.filter((u) => !(u.element_id === t.id && u.field === 'direction')); commit(next, `Confirmed ${t.id}`); } })
        : null,
    ].filter(Boolean)));
    if (t.source === t.target) box.lastChild.firstChild.disabled = true;
    box.append(field('Event (trigger)', textInput('f-event', t.event, (v) => set({ event: v }), t.event == null ? `none: triggered by "${t.id}"` : '')));
    box.append(html('div', { class: 'grid2' }, [
      field('Guard', textInput('f-guard', t.guard, (v) => set({ guard: v }), 'e.g. retries < 3')),
      field('Action', textInput('f-action', t.action, (v) => set({ action: v }), 'e.g. send ACK')),
    ]));
    box.append(field('Diagram label (raw OCR)', textInput('f-tlabel', t.label, (v) => set({ label: v }))));
    if (t.label_confidence != null) box.append(html('p', { class: 'hint', text: `OCR confidence ${(t.label_confidence * 100).toFixed(0)}%` }));
    box.append(html('div', { class: 'grid2' }, [
      field('Direction', selectInput('f-direction', [['forward', 'forward'], ['loop', 'loop'], ['unknown', 'unknown'], ['backward', 'backward']], t.direction, (v) => set({ direction: v }))),
      field('Kind', selectInput('f-kind', [['state_change', 'state change'], ['network_event', 'network event'], ['internal', 'internal']], t.kind,
        (v) => set(v === 'network_event' ? { kind: v, network: t.network || { event_type: 'other', protocol: null, packet: null } } : { kind: v, network: null }))),
    ]));
    if (t.kind === 'network_event') {
      const ne = t.network || { event_type: 'other', protocol: null, packet: null };
      const types = ['connect', 'accept', 'send', 'receive', 'ack', 'timeout', 'failure', 'retry', 'close', 'reset', 'other'];
      box.append(html('div', { class: 'grid2' }, [
        field('Network event', selectInput('f-ntype', types.map((x) => [x, x]), ne.event_type, (v) => set({ network: { ...ne, event_type: v } }))),
        field('Protocol', textInput('f-proto', ne.protocol, (v) => set({ network: { ...ne, protocol: v } }), 'e.g. TCP')),
      ]));
      box.append(field('Packet', textInput('f-packet', ne.packet, (v) => set({ network: { ...ne, packet: v } }), 'e.g. SYN-ACK')));
    }
    if (t.confidence != null) box.append(html('p', { class: 'hint', text: `Detection confidence ${(t.confidence * 100).toFixed(0)}%` }));
    box.append(html('div', { class: 'row-buttons' }, [
      html('button', { class: 'danger', text: 'Delete transition', onclick: () => { commit(Ops.deleteTransition(model, t.id), `Deleted ${t.id}`); select(null); } }),
    ]));
  }

  function renderNetworkEditor() {
    const box = $('network-editor');
    box.replaceChildren();
    const devices = (model.network && model.network.devices) || [];
    const links = (model.network && model.network.links) || [];
    for (const d of devices) {
      const states = html('select', { multiple: '', 'aria-label': `States of ${d.id}`, size: Math.min(4, Math.max(2, model.states.length)) });
      for (const s of model.states) {
        const o = html('option', { value: s.id, text: `${s.label || s.id}` });
        if ((d.state_ids || []).includes(s.id)) o.selected = true;
        states.append(o);
      }
      const update = (changes) => {
        const next = Ops.clone(model);
        next.network.devices = next.network.devices.map((x) => (x.id === d.id ? { ...x, ...changes } : x));
        commit(next, `Updated device ${d.id}`);
      };
      states.addEventListener('change', () => update({ state_ids: [...states.selectedOptions].map((o) => o.value) }));
      box.append(html('div', { class: 'suggestion' }, [
        html('div', { class: 'list-row' }, [html('strong', { text: `Device ${d.id}` }),
          html('button', { class: 'danger', text: 'Remove', onclick: () => commit(Ops.deleteDevice(model, d.id), `Removed ${d.id}`) })]),
        html('div', { class: 'grid2' }, [
          field('Name', textInput(`d-${d.id}-label`, d.label, (v) => update({ label: v }), 'e.g. Client')),
          field('Role', textInput(`d-${d.id}-role`, d.role, (v) => update({ role: v }), 'e.g. router')),
        ]),
        field('States of this device', states),
      ]));
    }
    for (const l of links) {
      box.append(html('div', { class: 'list-row' }, [
        html('span', { text: `${l.id}: ${l.source} ↔ ${l.target}${l.medium ? ' (' + l.medium + ')' : ''}` }),
        html('button', { class: 'danger', text: 'Remove', onclick: () => commit(Ops.deleteLink(model, l.id), `Removed ${l.id}`) }),
      ]));
    }
    const buttons = [html('button', { text: 'Add device', onclick: () => { const label = prompt('Device name (e.g. Client, Server, Router)'); if (label === null) return; const [next, id] = Ops.addDevice(model, label.trim() || null); commit(next, `Added ${id}`); } })];
    if (devices.length >= 2) {
      buttons.push(html('button', { text: 'Add link', onclick: () => {
        const ids = devices.map((d) => d.id).join(', ');
        const a = prompt(`Link from device (${ids})`); if (!a) return;
        const b = prompt(`to device (${ids})`); if (!b) return;
        const medium = prompt('Medium (optional, e.g. Ethernet, Wi-Fi)') || null;
        if (!devices.some((d) => d.id === a.trim()) || !devices.some((d) => d.id === b.trim())) { status('Unknown device id', 'error'); return; }
        const [next, id] = Ops.addLink(model, a.trim(), b.trim(), medium); commit(next, `Added link ${id}`);
      } }));
    }
    box.append(html('div', { class: 'row-buttons' }, buttons));
  }

  // ---------------------------------------------------------------- issues / json / code
  function renderIssues() {
    const box = $('issues');
    box.replaceChildren();
    const items = [];
    for (const e of validation.errors || []) items.push({ kind: 'err', text: e });
    (model.uncertainties || []).forEach((u) => items.push({ kind: 'warn', text: `${u.element_id}.${u.field}: ${u.reason}`, id: u.element_id }));
    for (const w of validation.warnings || []) items.push({ kind: 'info', text: w });
    for (const w of (model.metadata && model.metadata.warnings) || []) items.push({ kind: 'info', text: `detector: ${w}` });
    if (validation.analysis && validation.analysis.unreachable_states.length) {
      items.push({ kind: 'warn', text: `Unreachable from the initial state: ${validation.analysis.unreachable_states.join(', ')}` });
    }
    if (!items.length) box.append(html('p', { class: 'note ok', text: 'No validation issues or open uncertainties.' }));
    const lookupId = (text) => { const m = /\$\.(states|transitions)\[(\d+)\]/.exec(text); if (!m) return null; const pool = m[1] === 'states' ? model.states : model.transitions; return pool[Number(m[2])] ? pool[Number(m[2])].id : null; };
    for (const item of items) {
      const id = item.id || lookupId(item.text);
      const b = html('button', { class: `issue ${item.kind}`, text: item.text });
      if (id) b.addEventListener('click', () => select(model.states.some((s) => s.id === id) ? 'state' : 'transition', id));
      box.append(b);
    }
    const count = (validation.errors || []).length + (model.uncertainties || []).length;
    $('issue-count').textContent = count ? String(count) : '';
  }

  function renderJson() { if (!jsonDirty) $('json').value = JSON.stringify(model, null, 2); }

  function renderCodeState() {
    const stale = generated && generatedFor !== JSON.stringify(model);
    $('code-stale').hidden = !stale;
  }

  function renderAll() {
    renderGraph();
    renderInspector();
    renderNetworkEditor();
    renderIssues();
    renderJson();
    renderCodeState();
    renderOverlay();
    $('undo').disabled = !history.canUndo;
    $('redo').disabled = !history.canRedo;
  }

  // ---------------------------------------------------------------- image + overlay
  function renderOverlay() {
    const img = $('image');
    const overlay = $('overlay');
    overlay.replaceChildren();
    if (img.hidden || !img.naturalWidth) return;
    overlay.setAttribute('viewBox', `0 0 ${img.naturalWidth} ${img.naturalHeight}`);
    overlay.setAttribute('width', img.clientWidth);
    overlay.setAttribute('height', img.clientHeight);
    if (!$('toggle-overlay').checked) return;
    for (const d of session.detections || []) {
      const [x1, y1, x2, y2] = d.bbox;
      const r = el('rect', { class: `ov-box ov-${d.kind}`, x: x1, y: y1, width: x2 - x1, height: y2 - y1 });
      r.append(el('title', {}, `${d.kind} ${(d.confidence * 100).toFixed(0)}%`));
      overlay.append(r);
    }
    for (const t of model.transitions) {
      if (!t.points || t.points.length < 2) continue;
      overlay.append(el('polyline', { class: 'ov-path', points: t.points.map((p) => `${p.x},${p.y}`).join(' ') }));
    }
    highlightOverlay();
  }

  function highlightOverlay() {
    const overlay = $('overlay');
    overlay.querySelectorAll('.ov-uncertain').forEach((n) => n.remove());
    if (!$('toggle-overlay').checked) return;
    const ids = new Set((model.uncertainties || []).map((u) => u.element_id));
    if (selection) ids.add(selection.id);
    for (const s of model.states) {
      if (!ids.has(s.id) || !s.bbox) continue;
      const [x1, y1, x2, y2] = s.bbox;
      overlay.append(el('rect', { class: 'ov-uncertain', x: x1 - 3, y: y1 - 3, width: x2 - x1 + 6, height: y2 - y1 + 6, rx: 6 }));
    }
    for (const t of model.transitions) {
      if (!ids.has(t.id) || !t.points || t.points.length < 2) continue;
      overlay.append(el('polyline', { class: 'ov-uncertain', fill: 'none', points: t.points.map((p) => `${p.x},${p.y}`).join(' ') }));
    }
  }

  async function loadImage() {
    const img = $('image');
    if (!session.has_image) { img.hidden = true; $('image-empty').hidden = false; return; }
    img.src = `/api/image?ts=${Date.now()}`;
    await img.decode().catch(() => {});
    img.hidden = false;
    $('image-empty').hidden = true;
    $('image-meta').textContent = `${img.naturalWidth}×${img.naturalHeight}`;
    renderOverlay();
  }

  // ---------------------------------------------------------------- zoom / pan
  function fit() {
    const svg = $('graph');
    if (!model.states.length) { view = { x: 0, y: 0, k: 1 }; applyView(); return; }
    const xs = model.states.map((s) => (s.position || { x: 80 }).x), ys = model.states.map((s) => (s.position || { y: 80 }).y);
    const minX = Math.min(...xs) - 110, maxX = Math.max(...xs) + 110, minY = Math.min(...ys) - 110, maxY = Math.max(...ys) + 110;
    const w = svg.clientWidth || 800, h = svg.clientHeight || 600;
    const k = Math.max(0.2, Math.min(2.5, Math.min(w / (maxX - minX), h / (maxY - minY))));
    view = { k, x: (w - (maxX + minX) * k) / 2, y: (h - (maxY + minY) * k) / 2 };
    applyView();
  }
  function zoom(factor, cx, cy) {
    const svg = $('graph');
    const rect = svg.getBoundingClientRect();
    const px = cx === undefined ? rect.width / 2 : cx, py = cy === undefined ? rect.height / 2 : cy;
    const k = Math.max(0.1, Math.min(6, view.k * factor));
    view.x = px - ((px - view.x) * k) / view.k;
    view.y = py - ((py - view.y) * k) / view.k;
    view.k = k;
    applyView();
  }

  function setupCanvas() {
    const svg = $('graph');
    let pan = null;
    svg.addEventListener('wheel', (e) => { e.preventDefault(); const rect = svg.getBoundingClientRect(); zoom(e.deltaY < 0 ? 1.12 : 1 / 1.12, e.clientX - rect.left, e.clientY - rect.top); }, { passive: false });
    svg.addEventListener('pointerdown', (e) => {
      if (e.target.closest('.state') || e.target.closest('.transition')) return;
      pan = { x: e.clientX, y: e.clientY, vx: view.x, vy: view.y, moved: false };
      svg.classList.add('panning');
      svg.setPointerCapture(e.pointerId);
    });
    svg.addEventListener('pointermove', (e) => {
      if (!pan) return;
      if (Math.hypot(e.clientX - pan.x, e.clientY - pan.y) > 3) pan.moved = true;
      view.x = pan.vx + e.clientX - pan.x; view.y = pan.vy + e.clientY - pan.y; applyView();
    });
    svg.addEventListener('pointerup', () => {
      if (pan && !pan.moved) { select(null); if (connectFrom) { connectFrom = null; renderGraph(); } }
      pan = null; svg.classList.remove('panning');
    });
    svg.addEventListener('dblclick', (e) => {
      if (e.target.closest('.state') || e.target.closest('.transition')) return;
      const [next, id] = Ops.addState(model, svgPoint(e));
      commit(next, `Added ${id}`);
      select('state', id);
    });
  }

  // ---------------------------------------------------------------- tabs
  function showTab(name) {
    document.querySelectorAll('.tabs button').forEach((b) => b.setAttribute('aria-selected', String(b.dataset.tab === name)));
    document.querySelectorAll('.tab').forEach((t) => { t.hidden = t.id !== `tab-${name}`; });
  }

  // ---------------------------------------------------------------- actions
  async function upload(file) {
    if (!file) return;
    status(`Detecting ${file.name}…`);
    try {
      const response = await fetch('/api/upload', { method: 'POST', headers: { 'X-Filename': file.name, 'Content-Type': 'application/octet-stream' }, body: file });
      const data = await response.json();
      if (!response.ok) throw new Error((data.errors || ['upload failed']).join('; '));
      history.push(model);
      model = Ops.normalize(data.machine);
      validation = data.validation;
      session = await api('/api/session');
      generated = null; generatedFor = null; $('code').textContent = ''; $('code-report').replaceChildren();
      jsonDirty = false;
      selection = null;
      $('workspace').classList.remove('no-image');
      $('toggle-image').checked = true;
      await loadImage();
      renderAll();
      fit();
      const open = (model.uncertainties || []).length;
      status(`Detected ${model.states.length} states and ${model.transitions.length} transitions${open ? `; ${open} item(s) need review` : ''}`, open ? '' : 'ok');
    } catch (e) {
      status(`Detection failed: ${e.message}`, 'error');
    }
  }

  async function save() {
    try {
      const saved = await api('/api/machine', model);
      status('Saved and validated', 'ok');
      model = Ops.normalize(saved);
      renderAll();
    } catch (e) {
      status(`Not saved: ${e.message}`, 'error');
      if (e.data) { validation = e.data; renderIssues(); renderGraph(); showTab('issues'); }
    }
  }

  async function applyJson() {
    const box = $('json-errors');
    box.replaceChildren();
    let parsed;
    try { parsed = JSON.parse($('json').value); } catch (e) { box.append(html('div', { class: 'note err', text: `JSON syntax error: ${e.message}` })); return; }
    try {
      const report = await api('/api/validate', parsed);
      if (!report.valid) { report.errors.forEach((x) => box.append(html('div', { class: 'note err', text: x }))); status('JSON rejected', 'error'); return; }
      jsonDirty = false;
      commit(Ops.normalize(parsed), 'Applied JSON');
      validation = report;
      box.append(html('div', { class: 'note ok', text: 'Applied. The graph now reflects the JSON.' }));
    } catch (e) { box.append(html('div', { class: 'note err', text: e.message })); }
  }

  async function generate() {
    const report = $('code-report');
    report.replaceChildren(html('p', { class: 'hint', text: 'Generating and running conformance tests in the sandbox…' }));
    try {
      const snapshot = JSON.stringify(model);
      const data = await api('/api/generate', { machine: model, backend: $('backend').value });
      generated = data; generatedFor = snapshot;
      $('code').textContent = data.source;
      const c = data.conformance;
      report.replaceChildren(html('div', { class: `note ${c.valid ? 'ok' : 'err'}` }, [
        html('strong', { text: c.valid ? 'Verified: ' : 'Rejected: ' }),
        document.createTextNode(c.valid
          ? `${c.steps_checked} steps in ${c.traces} traces match the reference semantics; ${c.covered_transitions.length}/${model.transitions.length} transitions exercised.`
          : (c.errors.concat(c.mismatches.map((m) => JSON.stringify(m)))).slice(0, 4).join(' | ')),
      ]));
      if (c.uncovered_transitions.length) report.append(html('div', { class: 'note', text: `Not exercised: ${c.uncovered_transitions.join(', ')}` }));
      for (const w of data.warnings || []) report.append(html('div', { class: 'note', text: w }));
      renderCodeState();
      status(c.valid ? 'Code generated and verified' : 'Generated code failed verification', c.valid ? 'ok' : 'error');
    } catch (e) {
      report.replaceChildren(html('div', { class: 'note err', text: e.message }));
      status(`Generation failed: ${e.message}`, 'error');
    }
  }

  async function explainCode() {
    const out = $('code-explanation');
    if (!generated) { out.replaceChildren(html('div', { class: 'note', text: 'Generate code first.' })); return; }
    out.replaceChildren(html('p', { class: 'hint', text: 'Asking Gemini…' }));
    try {
      const data = await api('/api/explain-code', { machine: model, source: generated.source });
      out.replaceChildren(html('h3', { text: 'Explanation' }), html('p', { text: data.explanation }), html('h3', { text: 'Suggested tests (checked against the reference semantics)' }));
      const ul = html('ul', { class: 'compact' });
      for (const t of data.test_cases) {
        ul.append(html('li', { text: `${t.agrees_with_reference ? '✔' : '✘'} ${t.name}: ${t.events.join(' → ') || '(no events)'} ⇒ ${t.expected_error || t.expected_states.join(', ')}${t.agrees_with_reference ? '' : `  (reference: ${t.reference_error || t.reference_states.join(', ')})`}` }));
      }
      out.append(ul);
    } catch (e) { out.replaceChildren(html('div', { class: 'note err', text: e.message })); }
  }

  async function explain() {
    const out = $('gemini-output');
    out.replaceChildren(html('p', { class: 'hint', text: 'Asking Gemini…' }));
    try {
      const data = await api('/api/explain', { machine: model, question: $('question').value });
      const name = (id) => { const s = model.states.find((x) => x.id === id); return s ? `${s.label || id} (${id})` : id; };
      out.replaceChildren(html('h3', { text: 'Summary' }), html('p', { text: data.summary }));
      if (data.answer) out.append(html('h3', { text: 'Answer' }), html('p', { text: data.answer }));
      const section = (title, items) => { if (!items.length) return; out.append(html('h3', { text: title })); const ul = html('ul', { class: 'compact' }); items.forEach((x) => ul.append(html('li', { text: x }))); out.append(ul); };
      section('States', data.states.map((s) => `${name(s.id)}: ${s.description}`));
      section('Transitions', data.transitions.map((t) => `${t.id}: ${t.description}`));
      section('Network events', data.network_events.map((n) => `${n.transition_id}: ${n.description}`));
      section('Execution flows', data.execution_flows);
      section('Ambiguities', data.ambiguities);
      section('Warnings', data.warnings);
    } catch (e) { out.replaceChildren(html('div', { class: 'note err', text: e.message })); }
  }

  function applySuggestion(s) {
    const isState = model.states.some((x) => x.id === s.element_id);
    const value = s.suggested_value;
    let next;
    if (isState) {
      next = Ops.updateState(model, s.element_id, { [s.field]: value });
    } else if (s.field === 'direction' && value === 'reverse') {
      next = Ops.reverseTransition(model, s.element_id);
    } else {
      next = Ops.updateTransition(model, s.element_id, { [s.field]: value });
    }
    next.uncertainties = (next.uncertainties || []).filter((u) => !(u.element_id === s.element_id && u.field === s.field));
    commit(next, `Applied suggestion to ${s.element_id}.${s.field}`);
  }

  async function suggest() {
    const out = $('gemini-output');
    out.replaceChildren(html('p', { class: 'hint', text: 'Asking Gemini to compare the image with the graph…' }));
    try {
      const data = await api('/api/suggest', { machine: model });
      out.replaceChildren(html('h3', { text: `Suggestions (${data.suggestions.length})` }));
      if (!data.suggestions.length) out.append(html('p', { class: 'hint', text: 'No corrections suggested.' }));
      for (const s of data.suggestions) {
        const card = html('div', { class: 'suggestion' }, [
          html('div', {}, [html('strong', { text: `${s.element_id}.${s.field} → ${s.suggested_value === null ? 'null' : JSON.stringify(s.suggested_value)}` }), html('span', { class: 'hint', text: `  confidence ${(s.confidence * 100).toFixed(0)}%` })]),
          html('div', { class: 'hint', text: s.evidence }),
        ]);
        const buttons = html('div', { class: 'row-buttons' }, [
          html('button', { class: 'primary', text: 'Accept', onclick: () => { applySuggestion(s); card.remove(); } }),
          html('button', { text: 'Reject', onclick: () => card.remove() }),
        ]);
        card.append(buttons);
        out.append(card);
      }
      if (data.rejected.length) out.append(html('p', { class: 'hint', text: `${data.rejected.length} malformed or out-of-diagram suggestion(s) were discarded.` }));
      for (const n of data.notes) out.append(html('p', { class: 'hint', text: n }));
    } catch (e) { out.replaceChildren(html('div', { class: 'note err', text: e.message })); }
  }

  async function saveCorrection() {
    if (!session.has_image) { status('Upload an image first: corrections pair an image with its reviewed graph', 'error'); return; }
    const dialog = $('correction-dialog');
    dialog.returnValue = '';
    dialog.showModal();
    dialog.addEventListener('close', async function onClose() {
      dialog.removeEventListener('close', onClose);
      if (dialog.returnValue !== 'save') return;
      const split = dialog.querySelector('input[name=split]:checked').value;
      try {
        const data = await api('/api/correction', { machine: model, split, reviewer: $('reviewer').value });
        status(`Saved correction (${data.changes.length} change(s), ${data.split} split)`, 'ok');
      } catch (e) { status(`Correction not saved: ${e.message}`, 'error'); }
    });
  }

  function download(name, text, type) {
    const url = URL.createObjectURL(new Blob([text], { type }));
    const a = html('a', { href: url, download: name });
    document.body.append(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  function exportSvg() {
    const svg = $('graph').cloneNode(true);
    const css = [...document.styleSheets].flatMap((sheet) => { try { return [...sheet.cssRules].map((r) => r.cssText); } catch (_) { return []; } }).join('\n');
    const computed = getComputedStyle(document.documentElement);
    const vars = ['--ink', '--net', '--warn', '--err', '--node', '--canvas', '--sel', '--muted'].map((v) => `${v}:${computed.getPropertyValue(v)}`).join(';');
    svg.setAttribute('xmlns', SVG);
    svg.setAttribute('style', vars);
    svg.querySelectorAll('.edge-hit').forEach((n) => n.remove());
    const style = document.createElementNS(SVG, 'style');
    style.textContent = css;
    svg.prepend(style);
    svg.setAttribute('width', $('graph').clientWidth);
    svg.setAttribute('height', $('graph').clientHeight);
    download(`${model.diagram_id || 'diagram'}.svg`, new XMLSerializer().serializeToString(svg), 'image/svg+xml');
  }

  function exportAs(kind) {
    const base = model.diagram_id || 'diagram';
    if (kind === 'json') download(`${base}.json`, JSON.stringify(model, null, 2) + '\n', 'application/json');
    else if (kind === 'svg') exportSvg();
    else if (kind === 'py' || kind === 'tests') {
      if (!generated) { status('Generate code first (Code tab)', 'error'); showTab('code'); return; }
      if (generatedFor !== JSON.stringify(model)) status('Exporting code generated before your latest edits; regenerate to sync', 'error');
      if (kind === 'py') download('state_machine.py', generated.source, 'text/x-python');
      else if (generated.tests) download('test_state_machine.py', generated.tests, 'text/x-python');
      else status('Tests are only produced for verified code', 'error');
    }
  }

  // ---------------------------------------------------------------- wiring
  function wire() {
    $('upload').addEventListener('change', (e) => { upload(e.target.files[0]); e.target.value = ''; });
    $('save').addEventListener('click', save);
    $('save-correction').addEventListener('click', saveCorrection);
    $('undo').addEventListener('click', undo);
    $('redo').addEventListener('click', redo);
    $('add-state').addEventListener('click', () => {
      const svg = $('graph');
      const center = { x: (svg.clientWidth / 2 - view.x) / view.k, y: (svg.clientHeight / 2 - view.y) / view.k };
      const [next, id] = Ops.addState(model, center);
      commit(next, `Added ${id}`); select('state', id);
    });
    $('connect').addEventListener('click', () => {
      const on = $('connect').getAttribute('aria-pressed') !== 'true';
      setConnect(on);
      renderGraph();
      status(on ? 'Connect: click the source state, then the target state' : 'Connect mode off');
    });
    $('layout').addEventListener('click', () => { commit(Ops.autoLayout(model), 'Applied automatic layout'); fit(); });
    $('zoom-in').addEventListener('click', () => zoom(1.25));
    $('zoom-out').addEventListener('click', () => zoom(0.8));
    $('fit').addEventListener('click', fit);
    $('toggle-image').addEventListener('change', (e) => $('workspace').classList.toggle('no-image', !e.target.checked));
    $('toggle-overlay').addEventListener('change', renderOverlay);
    $('export').addEventListener('change', (e) => { if (e.target.value) exportAs(e.target.value); e.target.value = ''; });
    $('json').addEventListener('input', () => { jsonDirty = true; });
    $('apply-json').addEventListener('click', applyJson);
    $('reset-json').addEventListener('click', () => { jsonDirty = false; renderJson(); $('json-errors').replaceChildren(); });
    $('import-json').addEventListener('change', async (e) => {
      const file = e.target.files[0]; e.target.value = '';
      if (!file) return;
      $('json').value = await file.text(); jsonDirty = true; showTab('json'); applyJson();
    });
    $('generate').addEventListener('click', generate);
    $('explain-code').addEventListener('click', explainCode);
    $('explain').addEventListener('click', explain);
    $('suggest').addEventListener('click', suggest);
    document.querySelectorAll('.tabs button').forEach((b) => b.addEventListener('click', () => showTab(b.dataset.tab)));
    window.addEventListener('resize', renderOverlay);
    $('image').addEventListener('load', renderOverlay);
    document.addEventListener('keydown', (e) => {
      const typing = ['INPUT', 'TEXTAREA', 'SELECT'].includes(document.activeElement.tagName);
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'z' && !typing) { e.preventDefault(); e.shiftKey ? redo() : undo(); }
      else if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'y' && !typing) { e.preventDefault(); redo(); }
      else if ((e.key === 'Delete' || e.key === 'Backspace') && selection && !typing) {
        e.preventDefault();
        if (selection.kind === 'state') { if (confirm(`Delete ${selection.id} and its transitions?`)) commit(Ops.deleteState(model, selection.id), `Deleted ${selection.id}`); }
        else commit(Ops.deleteTransition(model, selection.id), `Deleted ${selection.id}`);
        select(null);
      } else if (e.key === 'Escape') { setConnect(false); renderGraph(); }
    });
    setupCanvas();
  }

  async function init() {
    wire();
    try {
      model = Ops.normalize(await api('/api/machine'));
      session = await api('/api/session');
    } catch (e) { status(`Could not load: ${e.message}`, 'error'); }
    if (!session.has_image) $('workspace').classList.add('no-image');
    await loadImage();
    renderAll();
    fit();
    scheduleValidation();
    status(model.states.length ? 'Loaded canonical JSON' : 'Upload a diagram image or start drawing (double-click the canvas)');
  }

  init();
})();
