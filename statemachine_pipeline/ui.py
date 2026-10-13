"""Small dependency-free local graph editor for the canonical JSON contract."""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse
from .models import StateMachine
from .validation import validate_state_machine
from .network import NetworkGraph, validate_network_graph


PAGE = r"""<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>State Diagram Editor</title>
<style>
:root{color-scheme:light;--blue:#1769aa;--border:#c7d0d9;--panel:#fff;--bg:#f6f8fa;--danger:#b42318}
*{box-sizing:border-box}body{font:15px system-ui,sans-serif;margin:0;background:var(--bg);color:#17202a}
.app{display:grid;grid-template-columns:minmax(420px,1fr) 430px;height:100vh}
main{min-width:0;display:flex;flex-direction:column}.toolbar{display:flex;flex-wrap:wrap;gap:6px;padding:10px;background:var(--panel);border-bottom:1px solid var(--border)}
button,input,select,textarea{font:inherit}button{border:1px solid #9aa8b5;background:#fff;border-radius:5px;padding:7px 10px;cursor:pointer}
button:hover,button:focus{border-color:var(--blue);outline:3px solid #b9ddf8}button.primary{background:var(--blue);color:#fff;border-color:var(--blue)}
button.danger{color:var(--danger)}button:disabled{opacity:.45;cursor:not-allowed}
#graph{flex:1;width:100%;min-height:300px;background:#eef3f7;touch-action:none}
aside{padding:14px;overflow:auto;background:var(--panel);border-left:1px solid var(--border)}
fieldset{border:1px solid var(--border);border-radius:6px;margin:0 0 14px;padding:12px}legend{font-weight:650;padding:0 5px}
label{display:block;font-weight:600;margin:8px 0 3px}input,select,textarea{width:100%;border:1px solid #9aa8b5;border-radius:4px;padding:7px;background:#fff}
input:focus,select:focus,textarea:focus{outline:3px solid #b9ddf8;border-color:var(--blue)}
.row{display:grid;grid-template-columns:1fr 1fr;gap:8px}.hint{font-size:13px;color:#53616d}.status{min-height:22px;margin:5px 0}.error{color:var(--danger);font-weight:600}.warning{color:#8a4b00}
.transition-list{list-style:none;padding:0;margin:0;max-height:190px;overflow:auto}.transition-list button{display:block;width:100%;text-align:left;margin:4px 0}
.transition-list button[aria-current=true]{background:#e4f2fc;border-color:var(--blue)}
circle{fill:#fff;stroke:var(--blue);stroke-width:2;cursor:grab}circle:focus{stroke-width:4}line{stroke:#53616d;stroke-width:2}
text{font-size:13px;pointer-events:none}.edge-label{font-size:12px;fill:#17202a}.selected{stroke:#d97706;stroke-width:4}
@media(max-width:850px){.app{grid-template-columns:1fr;height:auto}.app main{height:55vh}aside{border-left:0;border-top:1px solid var(--border)}}
</style></head>
<body><div class="app"><main>
<div class="toolbar" role="toolbar" aria-label="Graph actions">
 <button class="primary" id="add-state">Add state</button><button id="undo">Undo</button><button id="redo">Redo</button>
 <button id="layout">Auto layout</button><button id="save">Save JSON</button>
</div><svg id="graph" role="img" aria-label="Editable state machine graph" tabindex="0"></svg>
</main><aside>
<h1>State machine editor</h1><p class="hint">Select a transition to edit it. Use Tab to reach controls; drag states with the mouse or keyboard.</p>
<div id="status" class="status" role="status" aria-live="polite"></div>
<fieldset><legend>Transition editor</legend>
 <div class="row"><div><label for="source">Source state</label><select id="source"></select></div>
 <div><label for="target">Target state</label><select id="target"></select></div></div>
 <p class="hint">Choose the same source and target for a self-loop.</p>
 <label for="event">Event <span class="hint">(optional)</span></label><input id="event" placeholder="e.g. submit">
 <label for="guard">Guard <span class="hint">(optional)</span></label><input id="guard" placeholder="e.g. is_valid">
 <label for="action">Action <span class="hint">(optional)</span></label><input id="action" placeholder="e.g. save_record">
 <label for="label">Diagram label <span class="hint">(optional)</span></label><input id="label">
 <label for="direction">Direction</label><select id="direction"><option>forward</option><option>backward</option><option>loop</option><option>unknown</option></select>
 <div class="toolbar"><button class="primary" id="new-transition">New transition</button><button id="apply-transition">Apply changes</button><button class="danger" id="delete-transition">Delete transition</button></div>
 <p class="hint" id="transition-help">New transition mode. Source and target are required.</p>
</fieldset>
<fieldset><legend>Transitions</legend><ul id="transitions" class="transition-list" aria-label="Transitions"></ul></fieldset>
<fieldset><legend>Canonical JSON</legend><label for="json">Edit JSON</label>
 <textarea id="json" rows="13" spellcheck="false" aria-describedby="json-help"></textarea>
 <p id="json-help" class="hint">Use “Apply JSON” to validate changes before they enter the graph.</p>
 <button id="apply-json">Apply JSON</button>
</fieldset>
</aside></div>
<script>
let model,history=[],future=[],drag=null,selected=null;
const $=id=>document.getElementById(id),svg=$('graph'),jsonBox=$('json');
const fields=['source','target','event','guard','action','label','direction'];
function announce(message,kind=''){let el=$('status');el.textContent=message;el.className='status '+kind}
function clone(x){return JSON.parse(JSON.stringify(x))}
function sync(){jsonBox.value=JSON.stringify(model,null,2);renderForm();renderGraph();renderList()}
function commit(next,message='Unsaved changes'){history.push(clone(model));future=[];model=next;sync();announce(message,'warning')}
function options(selectId,chosen){let select=$(selectId);select.innerHTML='';model.states.forEach(s=>{let o=document.createElement('option');o.value=s.id;o.textContent=(s.label||'(unlabelled)')+' ['+s.id+']';o.selected=s.id===chosen;select.append(o)})}
function renderForm(){let t=model.transitions.find(x=>x.id===selected);options('source',t?t.source:model.states[0]?.id);options('target',t?t.target:model.states[0]?.id);
 fields.slice(2).forEach(id=>$(id).value=t&&t[id]!=null?t[id]:'');$('apply-transition').disabled=!t;$('delete-transition').disabled=!t;
 $('transition-help').textContent=t?'Editing '+t.id+'. Changes are validated when applied.':'New transition mode. Source and target are required.'}
function renderList(){let list=$('transitions');list.innerHTML='';if(!model.transitions.length){let p=document.createElement('p');p.className='hint';p.textContent='No transitions yet.';list.append(p);return}
 model.transitions.forEach(t=>{let b=document.createElement('button');b.type='button';b.textContent=t.id+': '+t.source+' -> '+t.target+(t.event?' ['+t.event+']':' [no event]');b.setAttribute('aria-current',String(t.id===selected));b.onclick=()=>{selected=t.id;renderForm();renderList();announce('Editing '+t.id)};list.append(b)})}
function position(s){return s.position||{x:80,y:80}}
function renderGraph(){svg.innerHTML='';let by=Object.fromEntries(model.states.map(s=>[s.id,s]));
 model.transitions.forEach(t=>{let a=by[t.source],b=by[t.target];if(!a||!b)return;let pa=position(a),pb=position(b);
 let l=document.createElementNS('http://www.w3.org/2000/svg','line');l.setAttribute('x1',pa.x);l.setAttribute('y1',pa.y);l.setAttribute('x2',pb.x);l.setAttribute('y2',pb.y);l.setAttribute('aria-label',t.id+' '+t.source+' to '+t.target);l.classList.toggle('selected',t.id===selected);l.onclick=()=>{selected=t.id;renderForm();renderList()};svg.append(l);
 let label=document.createElementNS('http://www.w3.org/2000/svg','text');label.className.baseVal='edge-label';label.setAttribute('x',(pa.x+pb.x)/2);label.setAttribute('y',(pa.y+pb.y)/2-5);label.textContent=t.event||t.label||t.id;svg.append(label)});
 model.states.forEach(s=>{let p=position(s),g=document.createElementNS('http://www.w3.org/2000/svg','g'),c=document.createElementNS('http://www.w3.org/2000/svg','circle');
 c.setAttribute('cx',p.x);c.setAttribute('cy',p.y);c.setAttribute('r',30);c.setAttribute('tabindex','0');c.setAttribute('role','button');c.setAttribute('aria-label',(s.label||s.id)+' state. Press Enter to rename.');c.classList.toggle('selected',s.id===selected);
 c.onpointerdown=e=>{c.setPointerCapture(e.pointerId);drag={id:s.id};};c.onpointermove=e=>{if(!drag||drag.id!==s.id)return;let r=svg.getBoundingClientRect();model={...model,states:model.states.map(x=>x.id===s.id?({...x,position:{x:e.clientX-r.left,y:e.clientY-r.top}}):x)};sync()};c.onpointerup=()=>{if(drag){history.push(clone(model));future=[];drag=null;announce('Moved '+s.id,'warning')}};c.onkeydown=e=>{if(e.key==='Enter'){e.preventDefault();rename(s.id)}};c.onclick=()=>rename(s.id);g.append(c);
 let tx=document.createElementNS('http://www.w3.org/2000/svg','text');tx.setAttribute('x',p.x-24);tx.setAttribute('y',p.y+5);tx.textContent=s.label||s.id;g.append(tx);svg.append(g)})}
function rename(id){let s=model.states.find(x=>x.id===id),label=prompt('State label for '+id+':',s.label||'');if(label!==null)commit({...model,states:model.states.map(x=>x.id===id?({...x,label:label||null}):x)},'Renamed '+id)}
function newTransition(){selected=null;renderForm();renderList();announce('New transition mode')}
function readTransition(){let value={source:$('source').value,target:$('target').value,event:$('event').value||null,guard:$('guard').value||null,action:$('action').value||null,label:$('label').value||null,direction:$('direction').value};return value}
function applyTransition(){let value=readTransition();if(!value.source||!value.target){announce('Choose both a source and target state','error');return}
 if(selected){commit({...model,transitions:model.transitions.map(t=>t.id===selected?({...t,...value}):t)},'Updated '+selected)}
 else{let i=0,ids=new Set(model.transitions.map(t=>t.id));while(ids.has('T'+i))i++;commit({...model,transitions:[...model.transitions,{id:'T'+i,...value,confidence:null}]},'Added transition T'+i);selected='T'+i;renderForm();renderList()}}
function deleteTransition(){if(!selected)return;if(!confirm('Delete transition '+selected+'?'))return;commit({...model,transitions:model.transitions.filter(t=>t.id!==selected)},'Deleted '+selected);selected=null;renderForm();renderList()}
function undo(){if(history.length){future.push(clone(model));model=history.pop();selected=null;sync();announce('Undo complete')}}
function redo(){if(future.length){history.push(clone(model));model=future.pop();sync();announce('Redo complete')}}
function addState(){let ids=new Set(model.states.map(s=>s.id)),i=0;while(ids.has('S'+i))i++;commit({...model,states:[...model.states,{id:'S'+i,label:null,type:'normal',position:{x:100+i*110,y:100},confidence:null}]},'Added state S'+i)}
function autoLayout(){commit({...model,states:model.states.map((s,i)=>({...s,position:{x:100+(i%5)*140,y:100+Math.floor(i/5)*110}}))},'Applied automatic layout')}
async function save(){try{let parsed=JSON.parse(jsonBox.value),r=await fetch('/api/machine',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(parsed)}),x=await r.json();if(!r.ok){announce((x.errors||['Invalid JSON']).join('; '),'error');return}model=x;history=[];future=[];sync();announce('Saved and validated')}catch(e){announce('Invalid JSON: '+e.message,'error')}}
function applyJson(){try{let next=JSON.parse(jsonBox.value);if(!next.states||!next.transitions)throw Error('states and transitions are required');commit(next,'Applied JSON changes')}catch(e){announce('Invalid JSON: '+e.message,'error')}}
async function load(){let r=await fetch('/api/machine');model=await r.json();sync();announce('Loaded canonical JSON')}
$('add-state').onclick=addState;$('undo').onclick=undo;$('redo').onclick=redo;$('layout').onclick=autoLayout;$('save').onclick=save;$('new-transition').onclick=newTransition;$('apply-transition').onclick=applyTransition;$('delete-transition').onclick=deleteTransition;$('apply-json').onclick=applyJson;
document.addEventListener('keydown',e=>{if((e.ctrlKey||e.metaKey)&&e.key==='z'){e.preventDefault();undo()}if((e.ctrlKey||e.metaKey)&&e.key==='y'){e.preventDefault();redo()}});
load();
</script></body></html>"""

NETWORK_PAGE = r"""<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Network Diagram Editor</title>
<style>
body{font:15px system-ui;margin:0;color:#17202a}header{padding:10px;border-bottom:1px solid #ccd6df}
main{display:grid;grid-template-columns:1fr 330px;height:calc(100vh - 52px)}svg{width:100%;height:100%;background:#eef3f7}
aside{padding:14px;border-left:1px solid #ccd6df;overflow:auto}button,input,textarea{font:inherit;padding:7px;margin:3px 0}
button{cursor:pointer}input,textarea{width:100%;box-sizing:border-box}.node{fill:#fff;stroke:#1769aa;stroke-width:2;cursor:grab}
.edge{stroke:#53616d;stroke-width:2}.selected{stroke:#d97706;stroke-width:4}.hint{color:#53616d;font-size:13px}
</style><header><button id="add">Add component</button><button id="edge">Add connection</button>
<button id="save">Save JSON</button><span id="status" class="hint"></span></header><main><svg id="graph" tabindex="0"></svg>
<aside><h2>Network diagram</h2><p class="hint">Drag circles, click a component to rename it, or use Add connection then click two components.</p>
<label>Selected component <input id="label" placeholder="component label"></label><button id="apply">Apply label</button>
<h3>Canonical JSON</h3><textarea id="json" rows="18" spellcheck="false"></textarea>
<button id="apply-json">Apply JSON</button></aside></main>
<script>
let model,selected=null,connecting=false,first=null,drag=null;
const $=id=>document.getElementById(id),svg=$('graph');
function announce(x){$('status').textContent=x}
function pos(n){return n.position||{x:80,y:80}}
function sync(){ $('json').value=JSON.stringify(model,null,2); render() }
function render(){svg.innerHTML='';let by=Object.fromEntries(model.nodes.map(n=>[n.id,n]));
 model.edges.forEach(e=>{let a=by[e.source],b=by[e.target];if(!a||!b)return;let l=document.createElementNS('http://www.w3.org/2000/svg','line');
 let pa=pos(a),pb=pos(b);l.setAttribute('x1',pa.x);l.setAttribute('y1',pa.y);l.setAttribute('x2',pb.x);l.setAttribute('y2',pb.y);l.className='edge';svg.append(l)});
 model.nodes.forEach(n=>{let p=pos(n),g=document.createElementNS('http://www.w3.org/2000/svg','g'),c=document.createElementNS('http://www.w3.org/2000/svg','circle');
 c.setAttribute('cx',p.x);c.setAttribute('cy',p.y);c.setAttribute('r',n.radius||30);c.className='node';c.classList.toggle('selected',n.id===selected);
 c.onpointerdown=e=>{c.setPointerCapture(e.pointerId);drag=n.id};c.onpointermove=e=>{if(drag!==n.id)return;let r=svg.getBoundingClientRect();n.position={x:e.clientX-r.left,y:e.clientY-r.top};sync()};
 c.onpointerup=()=>{drag=null};c.onclick=()=>{if(connecting){if(!first){first=n.id;announce('Choose the second component')}else if(first!==n.id){let i=0,ids=new Set(model.edges.map(e=>e.id));while(ids.has('E'+i))i++;model.edges.push({id:'E'+i,source:first,target:n.id,directed:false,label:null,confidence:null});connecting=false;first=null;announce('Connection added');sync()}return}selected=n.id;$('label').value=n.label||'';render()};g.append(c);
 let t=document.createElementNS('http://www.w3.org/2000/svg','text');t.setAttribute('x',p.x-22);t.setAttribute('y',p.y+5);t.textContent=n.label||n.id;g.append(t);svg.append(g)})}
 $('add').onclick=()=>{let i=0,ids=new Set(model.nodes.map(n=>n.id));while(ids.has('N'+i))i++;model.nodes.push({id:'N'+i,label:null,position:{x:100+i*100,y:100},radius:30,confidence:null});sync()};
 $('edge').onclick=()=>{connecting=true;first=null;announce('Choose the first component')};
 $('apply').onclick=()=>{if(!selected)return;model.nodes.find(n=>n.id===selected).label=$('label').value||null;sync()};
 $('apply-json').onclick=()=>{try{let next=JSON.parse($('json').value);if(!Array.isArray(next.nodes)||!Array.isArray(next.edges))throw Error('nodes and edges are required');model=next;sync();announce('JSON applied')}catch(e){announce('Invalid JSON: '+e.message)}};
 $('save').onclick=async()=>{try{let r=await fetch('/api/machine',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(model)}),x=await r.json();if(!r.ok)throw Error((x.errors||['Invalid graph']).join('; '));model=x;sync();announce('Saved')}catch(e){announce(e.message)}};
 fetch('/api/machine').then(r=>r.json()).then(x=>{model=x;sync();announce('Loaded')});
</script>"""


def serve(
    path: str | Path, host: str = "127.0.0.1", port: int = 8000,
    *, diagram_type: str = "auto",
) -> None:
    target = Path(path)

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, body: str, content_type: str) -> None:
            encoded = body.encode()
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def do_GET(self) -> None:
            if urlparse(self.path).path == "/api/machine":
                self._send(200, target.read_text(encoding="utf-8"), "application/json")
            else:
                try:
                    raw = json.loads(target.read_text(encoding="utf-8"))
                    selected = diagram_type
                    if selected == "auto":
                        selected = "network" if "nodes" in raw else "state-machine"
                    page = NETWORK_PAGE if selected == "network" else PAGE
                except (OSError, json.JSONDecodeError):
                    page = PAGE
                self._send(200, page, "text/html; charset=utf-8")

        def do_POST(self) -> None:
            if urlparse(self.path).path != "/api/machine":
                self._send(404, "{}", "application/json")
                return
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
                if "nodes" in body:
                    errors = validate_network_graph(body)
                    payload = NetworkGraph.from_dict(body)
                else:
                    machine = StateMachine.from_dict(body)
                    report = validate_state_machine(machine)
                    errors = report.errors
                    payload = machine
                if errors:
                    self._send(400, json.dumps({"errors": errors}), "application/json")
                    return
                target.write_text(payload.to_json() + "\n", encoding="utf-8")
                self._send(200, payload.to_json(), "application/json")
            except (json.JSONDecodeError, KeyError, TypeError, ValueError, OSError) as exc:
                self._send(400, json.dumps({"errors": [str(exc)]}), "application/json")

        def log_message(self, *_: object) -> None:
            return

    ThreadingHTTPServer((host, port), Handler).serve_forever()
