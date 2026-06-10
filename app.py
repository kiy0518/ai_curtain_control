#!/usr/bin/env python3
"""AI Curtain Control — dashboard server (Phase 2).

Low-latency MJPEG stream + JSON API + web/mobile (PWA) dashboard, built on the
standard library (no FastAPI dependency). Features:
  * live video + curtain state + current gesture + system stats
  * curtain control buttons (placeholder until motor — Phase M)
  * admin: runtime model/profile switch, confidence, gesture on/off
Open http://<board-ip>:8080

    python3 app.py --profile hand_near        # 근거리 손 (기본)
    python3 app.py --profile body_far         # 원거리 전신
"""

import argparse
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, "src")

from camera import isp_gst_pipeline                # noqa: E402
from draw import draw_fps                          # noqa: E402
from streaming import CameraThread, ProcessThread  # noqa: E402
from engine import PoseEngine                      # noqa: E402
from controller import CurtainController           # noqa: E402
from constants import GESTURE_KR                   # noqa: E402
import store                                        # noqa: E402
from scheduler import SchedulerThread              # noqa: E402


# --- small system-info helper (no psutil dependency) -----------------------
def system_info():
    info = {}
    try:
        with open("/proc/loadavg") as f:
            info["load"] = f.read().split()[0]
    except Exception:
        info["load"] = "?"
    try:
        mem = {}
        with open("/proc/meminfo") as f:
            for line in f:
                k, v = line.split(":")[0], line.split()[1]
                mem[k] = int(v)
        used = mem["MemTotal"] - mem["MemAvailable"]
        info["mem_used_mb"] = used // 1024
        info["mem_total_mb"] = mem["MemTotal"] // 1024
    except Exception:
        info["mem_used_mb"] = info["mem_total_mb"] = 0
    try:
        import glob
        temps = []
        for p in glob.glob("/sys/class/thermal/thermal_zone*/temp"):
            with open(p) as f:
                temps.append(int(f.read()) / 1000.0)
        info["temp_c"] = round(max(temps), 1) if temps else None
    except Exception:
        info["temp_c"] = None
    return info


DASHBOARD_HTML = """<!doctype html>
<html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1">
<meta name="theme-color" content="#141218">
<link rel="manifest" href="/manifest.json">
<title>AI 커튼 제어</title>
<style>
 /* Material 3 (dark) tokens */
 :root{
   --bg:#141218; --surface:#1D1B20; --sc:#211F26; --sch:#2B2930;
   --primary:#D0BCFF; --on-primary:#381E72; --on-surface:#E6E1E9; --on-var:#CAC4D0;
   --outline:#49454F; --error:#F2B8B5; --sec-c:#4A4458; --on-sec-c:#E8DEF8;
   --open:#A6D7A8; --open-c:#0A3D12; --stop:#FFCC80; --stop-c:#3A2600; --close:#F2B8B5; --close-c:#601410;
 }
 *{box-sizing:border-box}
 body{margin:0;background:var(--bg);color:var(--on-surface);
   font-family:Roboto,system-ui,"Noto Sans KR",sans-serif;font-size:15px}
 .appbar{display:flex;justify-content:space-between;align-items:center;
   padding:18px 20px;font-size:22px;font-weight:600}
 .appbar .t{display:flex;align-items:center;gap:10px}
 .chip{font-size:12px;font-weight:500;padding:6px 12px;border-radius:8px;
   background:var(--sch);color:var(--on-var)}
 .chip.on{background:#1b3a1f;color:var(--open)} .chip.off{background:#4a2024;color:var(--error)}
 main{max-width:760px;margin:0 auto;padding:8px 16px 40px}
 .card{background:var(--sc);border-radius:24px;padding:20px;margin:14px 0}
 .card h3{margin:0 0 14px;font-size:15px;font-weight:600;color:var(--on-var)}
 img#cam{width:100%;border-radius:20px;display:block;background:#000;min-height:200px}
 .state{font-size:26px;font-weight:600;text-align:center;padding:6px 0 16px}
 .state b{color:var(--primary)}
 .ctl{display:flex;gap:12px}
 .ctl button{flex:1;padding:16px 8px;border:0;border-radius:20px;font-size:16px;
   font-weight:600;cursor:pointer;color:#fff}
 .b-open{background:var(--open);color:var(--open-c)}
 .b-stop{background:var(--stop);color:var(--stop-c)}
 .b-close{background:var(--close);color:var(--close-c)}
 .note{font-size:12px;color:var(--on-var);opacity:.8;margin-top:12px}
 .kv{display:flex;justify-content:space-between;font-size:14px;padding:8px 0;
   border-bottom:1px solid var(--outline)}
 .kv:last-child{border:0} .kv b{color:var(--on-surface);font-weight:600}
 label{font-size:13px;color:var(--on-var);display:block;margin:14px 0 6px}
 select,input[type=text],input[type=time],input[type=number]{
   width:100%;padding:12px 14px;border-radius:12px;border:1px solid var(--outline);
   background:var(--sch);color:var(--on-surface);font-size:15px}
 input[type=range]{width:100%;accent-color:var(--primary)}
 .btn{display:inline-block;padding:10px 24px;border:0;border-radius:100px;
   background:var(--primary);color:var(--on-primary);font-weight:600;font-size:14px;cursor:pointer}
 .btn.tonal{background:var(--sec-c);color:var(--on-sec-c)}
 .switch{position:relative;width:48px;height:28px;flex:0 0 auto}
 .switch input{display:none}
 .switch .tr{position:absolute;inset:0;border-radius:100px;background:var(--sch);
   border:2px solid var(--outline);transition:.2s}
 .switch .kn{position:absolute;top:6px;left:6px;width:14px;height:14px;border-radius:50%;
   background:var(--on-var);transition:.2s}
 .switch input:checked+.tr{background:var(--primary);border-color:var(--primary)}
 .switch input:checked+.tr+.kn{left:26px;width:18px;height:18px;top:4px;background:var(--on-primary)}
 .sched{display:flex;align-items:center;gap:12px;padding:12px 0;border-bottom:1px solid var(--outline)}
 .sched .info{flex:1} .sched .info .a{font-weight:600} .sched .info .d{font-size:12px;color:var(--on-var)}
 .del{background:none;border:0;color:var(--error);font-size:18px;cursor:pointer}
 .days{display:flex;gap:6px;flex-wrap:wrap;margin-top:6px}
 .day{padding:8px 0;width:38px;text-align:center;border-radius:100px;background:var(--sch);
   color:var(--on-var);font-size:13px;cursor:pointer;user-select:none}
 .day.sel{background:var(--primary);color:var(--on-primary);font-weight:600}
 details summary{cursor:pointer;color:var(--on-var);font-weight:600;list-style:none}
 details summary::-webkit-details-marker{display:none}
 .grid2{display:flex;gap:12px} .grid2>*{flex:1}
</style></head>
<body>
<div class="appbar"><span class="t">🪟 AI 커튼 제어</span><span id="conn" class="chip off">연결 확인…</span></div>
<main>
 <div class="card" style="padding:8px"><img id="cam" src="/stream.mjpg" alt="live"></div>

 <div class="card">
   <div class="state">커튼 <b id="curtain">—</b></div>
   <div class="ctl">
     <button class="b-open"  onclick="ctl('OPEN')">열기</button>
     <button class="b-stop"  onclick="ctl('STOP')">정지</button>
     <button class="b-close" onclick="ctl('CLOSE')">닫기</button>
   </div>
   <div class="note" id="motornote">모터 미연결(Phase M) — 버튼/상태는 표시용 placeholder</div>
 </div>

 <div class="card">
   <h3>스케줄</h3>
   <div id="schedlist"></div>
   <details style="margin-top:12px"><summary>＋ 스케줄 추가</summary>
     <div class="grid2">
       <div><label>동작</label><select id="s_act"><option value="OPEN">열기</option><option value="CLOSE">닫기</option><option value="STOP">정지</option></select></div>
       <div><label>방식</label><select id="s_kind" onchange="kindUI()"><option value="time">시간</option><option value="sun">일출/일몰</option></select></div>
     </div>
     <div id="s_time_box"><label>시각</label><input type="time" id="s_time" value="07:00"></div>
     <div id="s_sun_box" style="display:none">
       <div class="grid2">
         <div><label>기준</label><select id="s_sun"><option value="sunrise">일출</option><option value="sunset">일몰</option></select></div>
         <div><label>오프셋(분, ±)</label><input type="number" id="s_off" value="0"></div>
       </div>
     </div>
     <label>요일 (없으면 매일)</label>
     <div class="days" id="s_days"></div>
     <label>이름(선택)</label><input type="text" id="s_name" placeholder="예: 아침 열기">
     <div style="margin-top:14px"><button class="btn" onclick="addSched()">추가</button></div>
   </details>
 </div>

 <div class="card">
   <h3>상태</h3>
   <div class="kv"><span>현재 제스처</span><b id="gesture">—</b></div>
   <div class="kv"><span>모델 / 입력</span><b id="model">—</b></div>
   <div class="kv"><span>추론</span><b id="perf">—</b></div>
   <div class="kv"><span>검출 수</span><b id="count">—</b></div>
   <div class="kv"><span>load / 메모리 / 온도</span><b id="sys">—</b></div>
 </div>

 <details class="card"><summary>⚙️ 관리자 설정</summary>
   <label>모델 / 프로파일 (런타임 전환)</label>
   <select id="profile" onchange="setModel()"></select>
   <div class="note" id="profdesc"></div>
   <label>신뢰도(conf): <span id="confv"></span></label>
   <input type="range" id="conf" min="0.1" max="0.8" step="0.05" onchange="setConf()">
   <label style="display:flex;align-items:center;gap:12px;margin-top:16px">제스처 인식 사용
     <span class="switch"><input type="checkbox" id="gest" onchange="setGest()"><span class="tr"></span><span class="kn"></span></span>
   </label>
   <label>위치(일출/일몰 계산용)</label>
   <div class="grid2">
     <input type="number" id="lat" step="0.0001" placeholder="위도">
     <input type="number" id="lon" step="0.0001" placeholder="경도">
   </div>
   <div style="margin-top:12px"><button class="btn tonal" onclick="saveLoc()">위치 저장</button></div>
 </details>
</main>
<script>
const $=id=>document.getElementById(id);
const KR={OPEN:'열림',CLOSE:'닫힘',STOP:'정지'};
const stMap={OPEN:'열림',CLOSED:'닫힘',STOPPED:'정지',UNKNOWN:'—'};
const DOW=['월','화','수','목','금','토','일'];
let profilesLoaded=false, daysSel=new Set();

// build weekday chips
DOW.forEach((d,i)=>{const e=document.createElement('div');e.className='day';e.textContent=d;
  e.onclick=()=>{e.classList.toggle('sel'); e.classList.contains('sel')?daysSel.add(i):daysSel.delete(i);};
  $('s_days').appendChild(e);});
function kindUI(){const sun=$('s_kind').value==='sun';$('s_sun_box').style.display=sun?'block':'none';$('s_time_box').style.display=sun?'none':'block';}

async function ctl(a){ try{await fetch('/api/control',{method:'POST',body:JSON.stringify({action:a})});}catch(e){} }
async function setModel(){ $('conn').textContent='모델 전환중…';
  try{ await fetch('/api/model',{method:'POST',body:JSON.stringify({profile:$('profile').value})}); }catch(e){} }
async function setConf(){ const v=parseFloat($('conf').value); $('confv').textContent=v.toFixed(2);
  fetch('/api/settings',{method:'POST',body:JSON.stringify({conf:v})}); }
async function setGest(){ fetch('/api/settings',{method:'POST',body:JSON.stringify({gesture_enabled:$('gest').checked})}); }
async function saveLoc(){ await fetch('/api/settings',{method:'POST',body:JSON.stringify({lat:parseFloat($('lat').value),lon:parseFloat($('lon').value)})}); }

async function addSched(){
  const kind=$('s_kind').value;
  const s={name:$('s_name').value,action:$('s_act').value,kind:kind,
    days:[...daysSel].sort().join(','),enabled:true};
  if(kind==='time'){const[h,m]=$('s_time').value.split(':');s.hh=+h;s.mm=+m;}
  else{s.sun_event=$('s_sun').value;s.sun_offset=parseInt($('s_off').value||'0');}
  await fetch('/api/schedules',{method:'POST',body:JSON.stringify(s)});
  $('s_name').value=''; daysSel.clear(); document.querySelectorAll('.day.sel').forEach(e=>e.classList.remove('sel'));
}
async function delSched(id){ await fetch('/api/schedules/delete',{method:'POST',body:JSON.stringify({id})}); }
async function togSched(id,en){ await fetch('/api/schedules/toggle',{method:'POST',body:JSON.stringify({id,enabled:en})}); }

function renderSched(list){
  const box=$('schedlist'); box.innerHTML = list.length?'':'<div class="note">스케줄 없음</div>';
  list.forEach(s=>{
    const when = s.kind==='time' ? String(s.hh).padStart(2,'0')+':'+String(s.mm).padStart(2,'0')
      : (s.sun_event==='sunrise'?'☀️일출':'🌇일몰')+(s.sun_offset?(s.sun_offset>0?'+':'')+s.sun_offset+'분':'');
    const days = s.days? s.days.split(',').map(i=>DOW[+i]).join('') : '매일';
    const row=document.createElement('div'); row.className='sched';
    row.innerHTML='<div class="info"><div class="a">'+KR[s.action]+' · '+when+'</div>'
      +'<div class="d">'+days+(s.name?' · '+s.name:'')+'</div></div>'
      +'<label class="switch"><input type="checkbox" '+(s.enabled?'checked':'')+'><span class="tr"></span><span class="kn"></span></label>'
      +'<button class="del">🗑</button>';
    row.querySelector('input').onchange=e=>togSched(s.id,e.target.checked);
    row.querySelector('.del').onclick=()=>delSched(s.id);
    box.appendChild(row);
  });
}

async function poll(){
 try{
   const s=await (await fetch('/api/state')).json();
   $('conn').textContent='온라인'; $('conn').className='chip on';
   $('curtain').textContent=stMap[s.curtain.state]||s.curtain.state;
   $('motornote').style.display=s.curtain.motor_connected?'none':'block';
   $('gesture').textContent=s.engine.gesture?(KR[s.engine.gesture]||s.engine.gesture):'—';
   $('model').textContent=s.engine.profile+' / '+s.engine.imgsz;
   $('perf').textContent=s.engine.infer_ms+'ms';
   $('count').textContent=s.engine.count;
   $('sys').textContent=s.system.load+' / '+s.system.mem_used_mb+'·'+s.system.mem_total_mb+'MB / '+(s.system.temp_c??'?')+'°C';
   renderSched(s.schedules||[]);
   if(!profilesLoaded){
     const sel=$('profile'); sel.innerHTML='';
     s.profiles.forEach(p=>{const o=document.createElement('option');o.value=p.name;o.textContent=p.name+' ('+p.num_keypoints+'kp '+p.imgsz+')';sel.appendChild(o);});
     sel.value=s.engine.profile;
     $('conf').value=s.engine.conf; $('confv').textContent=(+s.engine.conf).toFixed(2);
     $('gest').checked=s.engine.gesture_enabled;
     if(s.location){$('lat').value=s.location.lat||''; $('lon').value=s.location.lon||'';}
     profilesLoaded=true;
   }
   $('profdesc').textContent=s.engine.profile_desc;
 }catch(e){ $('conn').textContent='오프라인'; $('conn').className='chip off'; }
}
setInterval(poll,1000); poll();
if('serviceWorker' in navigator){navigator.serviceWorker.register('/sw.js').catch(()=>{});}
</script>
</body></html>
"""

MANIFEST = json.dumps({
    "name": "AI 커튼 제어", "short_name": "AI커튼", "start_url": "/",
    "display": "standalone", "background_color": "#0e0e10",
    "theme_color": "#111", "icons": [],
})
SW_JS = "self.addEventListener('fetch',()=>{});"   # minimal (enables install)


def make_handler(proc, engine, controller):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def _send(self, code, ctype, body):
            if isinstance(body, str):
                body = body.encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code=200):
            self._send(code, "application/json", json.dumps(obj))

        def _read_json(self):
            n = int(self.headers.get("Content-Length", 0) or 0)
            if not n:
                return {}
            try:
                return json.loads(self.rfile.read(n) or b"{}")
            except Exception:
                return {}

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self._send(200, "text/html; charset=utf-8", DASHBOARD_HTML)
            elif self.path == "/manifest.json":
                self._send(200, "application/manifest+json", MANIFEST)
            elif self.path == "/sw.js":
                self._send(200, "application/javascript", SW_JS)
            elif self.path == "/api/state":
                self._json({
                    "engine": engine.state(),
                    "curtain": controller.snapshot(),
                    "system": system_info(),
                    "profiles": engine.available_profiles(),
                    "schedules": store.list_schedules(),
                    "location": {"lat": store.get_setting("lat"),
                                 "lon": store.get_setting("lon")},
                })
            elif self.path == "/snapshot.jpg":
                jpeg = proc.jpeg_slot.get()
                if jpeg is None:
                    self.send_error(503); return
                self._send(200, "image/jpeg", jpeg)
            elif self.path == "/stream.mjpg":
                self.send_response(200)
                self.send_header("Cache-Control", "no-cache, private")
                self.send_header("Content-Type",
                                 "multipart/x-mixed-replace; boundary=FRAME")
                self.end_headers()
                last = 0
                try:
                    while True:
                        jpeg, last = proc.jpeg_slot.wait_newer(last, timeout=2.0)
                        if jpeg is None:
                            continue
                        self.wfile.write(b"--FRAME\r\nContent-Type: image/jpeg\r\n"
                                         b"Content-Length: " + str(len(jpeg)).encode()
                                         + b"\r\n\r\n" + jpeg + b"\r\n")
                except (BrokenPipeError, ConnectionResetError):
                    pass
            else:
                self.send_error(404)

        def do_POST(self):
            data = self._read_json()
            if self.path == "/api/control":
                ok = controller.command(str(data.get("action", "")), "dashboard")
                self._json({"ok": ok, "curtain": controller.snapshot()})
            elif self.path == "/api/model":
                try:
                    prof = engine.set_profile(str(data.get("profile", "")))
                    self._json({"ok": True, "profile": prof.name})
                except Exception as e:
                    self._json({"ok": False, "error": str(e)}, 400)
            elif self.path == "/api/settings":
                if "conf" in data:
                    engine.set_conf(float(data["conf"]))
                if "gesture_enabled" in data:
                    engine.set_gesture_enabled(bool(data["gesture_enabled"]))
                if "lat" in data:
                    store.set_setting("lat", float(data["lat"]))
                if "lon" in data:
                    store.set_setting("lon", float(data["lon"]))
                self._json({"ok": True})
            elif self.path == "/api/schedules":
                store.add_schedule(data)
                self._json({"ok": True, "schedules": store.list_schedules()})
            elif self.path == "/api/schedules/delete":
                store.delete_schedule(int(data.get("id")))
                self._json({"ok": True, "schedules": store.list_schedules()})
            elif self.path == "/api/schedules/toggle":
                store.set_enabled(int(data.get("id")), bool(data.get("enabled")))
                self._json({"ok": True, "schedules": store.list_schedules()})
            else:
                self.send_error(404)
    return H


def parse_args():
    p = argparse.ArgumentParser(description="AI Curtain Control dashboard")
    p.add_argument("--profile", default="hand_near", help="hand_near | body_far")
    p.add_argument("--conf", type=float, default=0.3)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--quality", type=int, default=75)
    return p.parse_args()


def main():
    args = parse_args()
    store.init()
    controller = CurtainController()
    engine = PoseEngine(args.profile, conf=args.conf, controller=controller)
    scheduler = SchedulerThread(controller)
    scheduler.start()

    pipeline = isp_gst_pipeline(width=args.width, height=args.height, fps=args.fps)
    cam = CameraThread(pipeline)
    proc = ProcessThread(cam.slot, jpeg_quality=args.quality,
                         process_fn=engine.process, fps_fn=draw_fps)
    cam.start()
    proc.start()

    server = ThreadingHTTPServer((args.host, args.port),
                                 make_handler(proc, engine, controller))
    print(f"\n  ▶ 대시보드 →  http://<board-ip>:{args.port}   (Ctrl+C 종료)\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n종료 중...")
    finally:
        scheduler.stop(); proc.stop(); cam.stop(); server.shutdown(); engine.release()


if __name__ == "__main__":
    main()
