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
<meta name="theme-color" content="#111">
<link rel="manifest" href="/manifest.json">
<title>AI 커튼 제어</title>
<style>
 *{box-sizing:border-box} body{margin:0;background:#0e0e10;color:#eee;font-family:system-ui,sans-serif}
 header{background:#16161a;padding:12px 16px;font-size:18px;font-weight:700;color:#4CAF50;display:flex;justify-content:space-between;align-items:center}
 .badge{font-size:12px;padding:3px 8px;border-radius:10px;background:#333}
 .badge.on{background:#1b5e20}.badge.off{background:#7f1d1d}
 main{max-width:760px;margin:0 auto;padding:12px}
 .vid{position:relative} img#cam{width:100%;border:1px solid #333;border-radius:8px;display:block;background:#000;min-height:200px}
 .row{display:flex;gap:8px;margin:10px 0;flex-wrap:wrap}
 button{flex:1;min-width:90px;padding:14px;border:0;border-radius:8px;font-size:16px;font-weight:700;color:#fff;cursor:pointer}
 .open{background:#2e7d32}.close{background:#c62828}.stop{background:#ef6c00}
 .card{background:#16161a;border-radius:8px;padding:12px;margin:10px 0}
 .card h3{margin:0 0 8px;font-size:14px;color:#9aa}
 .kv{display:flex;justify-content:space-between;font-size:14px;padding:3px 0;border-bottom:1px solid #222}
 select,input[type=range]{width:100%} label{font-size:13px;color:#aab}
 .state{font-size:22px;font-weight:800;text-align:center;padding:8px}
 .note{font-size:12px;color:#888}
 details summary{cursor:pointer;color:#9aa;font-size:14px}
</style></head>
<body>
<header><span>🪟 AI 커튼 제어</span><span id="conn" class="badge off">연결 확인…</span></header>
<main>
 <div class="vid"><img id="cam" src="/stream.mjpg" alt="live"></div>

 <div class="card">
   <div class="state">커튼: <span id="curtain">—</span></div>
   <div class="row">
     <button class="open"  onclick="ctl('OPEN')">열기</button>
     <button class="stop"  onclick="ctl('STOP')">정지</button>
     <button class="close" onclick="ctl('CLOSE')">닫기</button>
   </div>
   <div class="note" id="motornote">⚠ 모터 미연결(Phase M) — 버튼은 상태 표시용 placeholder</div>
 </div>

 <div class="card">
   <h3>상태</h3>
   <div class="kv"><span>현재 제스처</span><b id="gesture">—</b></div>
   <div class="kv"><span>모델 / 입력</span><b id="model">—</b></div>
   <div class="kv"><span>추론 / FPS</span><b id="perf">—</b></div>
   <div class="kv"><span>검출 수</span><b id="count">—</b></div>
   <div class="kv"><span>CPU load / 메모리 / 온도</span><b id="sys">—</b></div>
 </div>

 <details class="card"><summary>⚙️ 관리자 설정</summary>
   <p><label>모델 / 프로파일 (런타임 전환)</label>
      <select id="profile" onchange="setModel()"></select>
      <span class="note" id="profdesc"></span></p>
   <p><label>신뢰도(conf): <span id="confv"></span></label>
      <input type="range" id="conf" min="0.1" max="0.8" step="0.05" onchange="setConf()"></p>
   <p><label><input type="checkbox" id="gest" onchange="setGest()"> 제스처 인식 사용</label></p>
 </details>
 <p class="note">사용자 스케줄/원격 접속은 다음 단계(Phase 3~4)에서 추가됩니다.</p>
</main>
<script>
const $=id=>document.getElementById(id);
let profilesLoaded=false;
async function ctl(a){ try{await fetch('/api/control',{method:'POST',body:JSON.stringify({action:a})});}catch(e){} }
async function setModel(){ const p=$('profile').value; $('conn').textContent='모델 전환중…';
  try{ await fetch('/api/model',{method:'POST',body:JSON.stringify({profile:p})}); }catch(e){} }
async function setConf(){ const v=parseFloat($('conf').value); $('confv').textContent=v.toFixed(2);
  fetch('/api/settings',{method:'POST',body:JSON.stringify({conf:v})}); }
async function setGest(){ fetch('/api/settings',{method:'POST',body:JSON.stringify({gesture_enabled:$('gest').checked})}); }

async function poll(){
 try{
   const r=await fetch('/api/state'); const s=await r.json();
   $('conn').textContent='온라인'; $('conn').className='badge on';
   const KR={OPEN:'열림',CLOSE:'닫힘',STOP:'정지',OPEN_:'열림'};
   const stMap={OPEN:'열림',CLOSED:'닫힘',STOPPED:'정지',UNKNOWN:'—'};
   $('curtain').textContent=stMap[s.curtain.state]||s.curtain.state;
   $('motornote').style.display=s.curtain.motor_connected?'none':'block';
   $('gesture').textContent=s.engine.gesture?(KR[s.engine.gesture]||s.engine.gesture):'—';
   $('model').textContent=s.engine.profile+' / '+s.engine.imgsz;
   $('perf').textContent=s.engine.infer_ms+'ms';
   $('count').textContent=s.engine.count;
   $('sys').textContent=s.system.load+' / '+s.system.mem_used_mb+'·'+s.system.mem_total_mb+'MB / '+(s.system.temp_c??'?')+'°C';
   if(!profilesLoaded){
     const sel=$('profile'); sel.innerHTML='';
     s.profiles.forEach(p=>{const o=document.createElement('option');o.value=p.name;o.textContent=p.name+' ('+p.num_keypoints+'kp '+p.imgsz+')';sel.appendChild(o);});
     sel.value=s.engine.profile; $('profdesc').textContent=s.engine.profile_desc;
     $('conf').value=s.engine.conf; $('confv').textContent=(+s.engine.conf).toFixed(2);
     $('gest').checked=s.engine.gesture_enabled; profilesLoaded=true;
   } else { $('profdesc').textContent=s.engine.profile_desc; }
 }catch(e){ $('conn').textContent='오프라인'; $('conn').className='badge off'; }
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
                self._json({"ok": True})
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
    controller = CurtainController()
    engine = PoseEngine(args.profile, conf=args.conf, controller=controller)

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
        proc.stop(); cam.stop(); server.shutdown(); engine.release()


if __name__ == "__main__":
    main()
