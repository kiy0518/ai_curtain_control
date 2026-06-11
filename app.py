#!/usr/bin/env python3
"""AI Curtain Control — dashboard server (Phase 2).

Low-latency MJPEG stream + JSON API + web/mobile (PWA) dashboard, built on the
standard library (no FastAPI dependency). Features:
  * live video + curtain state + current gesture + system stats
  * curtain control buttons (placeholder until motor — Phase M)
  * admin: runtime model/profile switch, confidence, gesture on/off
Open http://<board-ip>:8080

    python3 app.py --profile hand_near        # 근거리 손 (기본)
    python3 app.py --profile body_far         # 원거리 전신 팔 포즈
    python3 app.py --profile body_motion      # 원거리 손목 움직임(쓸기/멈춤)
"""

import argparse
import json
import logging
import logging.handlers
import os
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, "src")

_START = time.time()
log = logging.getLogger("curtain")


def setup_logging():
    here = os.path.dirname(os.path.abspath(__file__))
    logdir = os.path.join(here, "logs")
    os.makedirs(logdir, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    fh = logging.handlers.RotatingFileHandler(
        os.path.join(logdir, "curtain.log"), maxBytes=1_000_000, backupCount=3)
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers[:] = [fh, sh]

from camera import isp_gst_pipeline                # noqa: E402
from draw import draw_fps                          # noqa: E402
from streaming import CameraThread, ProcessThread  # noqa: E402
from engine import PoseEngine                      # noqa: E402
from controller import CurtainController           # noqa: E402
from constants import GESTURE_KR                   # noqa: E402
import store                                        # noqa: E402
from scheduler import SchedulerThread              # noqa: E402
from remote import RemoteManager                   # noqa: E402
import auth                                         # noqa: E402
from http.cookies import SimpleCookie              # noqa: E402


def geoip():
    """Approximate location from the board's public IP (ip-api, no API key)."""
    try:
        url = "http://ip-api.com/json/?fields=status,lat,lon,city"
        with urllib.request.urlopen(url, timeout=5) as r:
            d = json.load(r)
        if d.get("status") == "success":
            return {"lat": d["lat"], "lon": d["lon"], "city": d.get("city")}
    except Exception:
        pass
    return None


def autodetect_location():
    """If no location saved yet, set it from IP geolocation (background)."""
    if store.get_setting("lat"):
        return
    g = geoip()
    if g:
        store.set_setting("lat", g["lat"])
        store.set_setting("lon", g["lon"])
        log.info("위치 자동감지(IP): %s (%.4f, %.4f)",
                 g.get("city"), g["lat"], g["lon"])


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
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
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
 #fsbtn{position:absolute;top:18px;right:18px;width:42px;height:42px;border:0;border-radius:50%;
   background:rgba(0,0,0,.55);color:#fff;font-size:20px;cursor:pointer;line-height:1}
 #vidwrap:fullscreen{background:#000;display:flex;align-items:center;justify-content:center;padding:0}
 #vidwrap:fullscreen img#cam{width:auto;height:auto;max-width:100%;max-height:100%;border-radius:0}
 #vidwrap:-webkit-full-screen{background:#000;display:flex;align-items:center;justify-content:center;padding:0}
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
 #map{height:260px;border-radius:12px;margin:8px 0;background:#2B2930}
 .leaflet-container{background:#2B2930}
 #toast{position:fixed;left:50%;bottom:24px;transform:translateX(-50%) translateY(20px);
   background:var(--sec-c);color:var(--on-sec-c);padding:12px 20px;border-radius:100px;
   font-size:14px;opacity:0;transition:.25s;pointer-events:none;z-index:99}
 #toast.show{opacity:1;transform:translateX(-50%) translateY(0)}
</style></head>
<body>
<div id="toast"></div>
<div class="appbar"><span class="t">🪟 AI 커튼 제어</span>
  <span><span id="conn" class="chip off">연결 확인…</span>
  <button id="sndbtn" class="chip" style="border:0;cursor:pointer;margin-left:6px" onclick="toggleSound()">🔊</button>
  <button class="chip" style="border:0;cursor:pointer;margin-left:6px" onclick="logout()">로그아웃</button></span></div>
<main>
 <div id="pwwarn" class="card" style="display:none;background:#4a2024;color:#F2B8B5">
   ⚠ 기본 비밀번호(admin) 사용 중입니다. 아래 관리자 설정에서 변경하세요.</div>
 <div class="card" id="vidwrap" style="padding:8px;position:relative">
   <img id="cam" src="/stream.mjpg" alt="live">
   <button id="fsbtn" onclick="toggleFull()" title="전체화면">⛶</button>
 </div>

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
   <h3>원격 접속 (고유 주소)</h3>
   <label style="display:flex;align-items:center;gap:12px;margin:0">원격 터널 사용
     <span class="switch"><input type="checkbox" id="rem" onchange="setRemote()"><span class="tr"></span><span class="kn"></span></span>
   </label>
   <div id="remurl" class="note" style="margin-top:10px;word-break:break-all"></div>
 </div>

 <div class="card">
   <h3>상태</h3>
   <div class="kv"><span>현재 제스처</span><b id="gesture">—</b></div>
   <div class="kv"><span>모델 / 입력</span><b id="model">—</b></div>
   <div class="kv"><span>추론</span><b id="perf">—</b></div>
   <div class="kv"><span>검출 수</span><b id="count">—</b></div>
   <div class="kv"><span>load / 메모리 / 온도</span><b id="sys">—</b></div>
 </div>

 <details class="card" id="adminbox"><summary>⚙️ 관리자 설정</summary>
   <label>모델 / 프로파일 (런타임 전환)</label>
   <select id="profile" onchange="setModel()"></select>
   <div class="note" id="profdesc"></div>
   <label>신뢰도(conf) — 0.05 ~ 0.95 (숫자 입력)</label>
   <input type="number" id="conf" min="0.05" max="0.95" step="0.05" inputmode="decimal"
          onchange="setConf()">
   <label>제스처 확정 카운트 — 연속 N회 (1~30, 손/팔 정적 제스처)</label>
   <input type="number" id="hold" min="1" max="30" step="1" inputmode="numeric"
          onchange="setHold()">
   <label>정지(STOP) 인식 시간 — 손 들고 멈춤 유지 초 (원거리 움직임, 0.5~4)</label>
   <input type="number" id="hold_sec" min="0.5" max="4" step="0.1" inputmode="decimal"
          onchange="setMotion()">
   <label>명령 후 대기(불응) 시간 — 다음 명령까지 무시할 초 (0.3~5)</label>
   <input type="number" id="refr_sec" min="0.3" max="5" step="0.1" inputmode="decimal"
          onchange="setMotion()">
   <label style="display:flex;align-items:center;gap:12px;margin-top:16px">제스처 인식 사용
     <span class="switch"><input type="checkbox" id="gest" onchange="setGest()"><span class="tr"></span><span class="kn"></span></span>
   </label>
   <label style="display:flex;align-items:center;gap:12px;margin-top:16px">영상 좌우반전 (거울)
     <span class="switch"><input type="checkbox" id="flip" onchange="setFlip()"><span class="tr"></span><span class="kn"></span></span>
   </label>
   <label>위치(일출/일몰 계산용) — 지도를 클릭/드래그하면 자동 저장</label>
   <div id="map"></div>
   <div class="grid2">
     <input type="number" id="lat" step="0.0001" placeholder="위도" onchange="onLatLon()">
     <input type="number" id="lon" step="0.0001" placeholder="경도" onchange="onLatLon()">
   </div>
   <div style="margin-top:10px"><button class="btn tonal" onclick="useGPS()">📍 내 위치 (GPS)</button></div>
   <label>비밀번호 변경</label>
   <input type="password" id="pw_old" placeholder="현재 비밀번호">
   <input type="password" id="pw_new" placeholder="새 비밀번호(4자 이상)" style="margin-top:8px">
   <div class="note" id="pwmsg"></div>
   <div style="margin-top:10px"><button class="btn" onclick="changePw()">비밀번호 변경</button></div>
 </details>
</main>
<script>
const $=id=>document.getElementById(id);
// --- 비프음 (Web Audio, 음원파일 불필요) ---
let _ac=null, _snd=(localStorage.getItem('snd')!=='0'), _lastEvt=null;
function _ensureAudio(){ if(!_ac){try{_ac=new (window.AudioContext||window.webkitAudioContext)();}catch(e){}}
  if(_ac&&_ac.state==='suspended')_ac.resume(); }
document.addEventListener('click',_ensureAudio);
function beep(freqs,dur=0.13){ if(!_snd||!_ac)return; let t=_ac.currentTime;
  freqs.forEach((f,i)=>{ const o=_ac.createOscillator(),g=_ac.createGain();
    o.type='sine'; o.frequency.value=f; o.connect(g); g.connect(_ac.destination);
    const st=t+i*(dur+0.04);
    g.gain.setValueAtTime(0.0001,st); g.gain.exponentialRampToValueAtTime(0.3,st+0.01);
    g.gain.exponentialRampToValueAtTime(0.0001,st+dur); o.start(st); o.stop(st+dur); }); }
const BEEP={OPEN:[660,990],CLOSE:[660,440],STOP:[520,520,520]}; // 열림=상승,닫힘=하강,정지=3연
function gestureBeep(l){ if(BEEP[l])beep(BEEP[l]); }
function toggleSound(){ const first=!_ac; _ensureAudio();
  // 첫 누름은 무조건 켜기(오디오 잠금해제+테스트음) — 그 뒤부터 정상 토글
  _snd=first?true:!_snd; localStorage.setItem('snd',_snd?'1':'0');
  $('sndbtn').textContent=_snd?'🔊':'🔇';
  if(_snd){ if(_ac&&_ac.state==='suspended')_ac.resume().then(()=>beep([880])); else beep([880]); } }
function toggleFull(){ const el=$('vidwrap');
  const fs=document.fullscreenElement||document.webkitFullscreenElement;
  if(!fs){ (el.requestFullscreen||el.webkitRequestFullscreen||(()=>{})).call(el); }
  else { (document.exitFullscreen||document.webkitExitFullscreen||(()=>{})).call(document); } }
let _tt; function toast(m){const t=$('toast');t.textContent=m;t.classList.add('show');
  clearTimeout(_tt);_tt=setTimeout(()=>t.classList.remove('show'),1500);}
const KR={OPEN:'열림',CLOSE:'닫힘',STOP:'정지'};
const stMap={OPEN:'열림',CLOSED:'닫힘',STOPPED:'정지',UNKNOWN:'—'};
const DOW=['월','화','수','목','금','토','일'];
let profilesLoaded=false, daysSel=new Set();

// build weekday chips
DOW.forEach((d,i)=>{const e=document.createElement('div');e.className='day';e.textContent=d;
  e.onclick=()=>{e.classList.toggle('sel'); e.classList.contains('sel')?daysSel.add(i):daysSel.delete(i);};
  $('s_days').appendChild(e);});
function kindUI(){const sun=$('s_kind').value==='sun';$('s_sun_box').style.display=sun?'block':'none';$('s_time_box').style.display=sun?'none':'block';}

async function ctl(a){ gestureBeep(a); try{await fetch('/api/control',{method:'POST',body:JSON.stringify({action:a})});}catch(e){} }
async function setModel(){ $('conn').textContent='모델 전환중…';
  try{ await fetch('/api/model',{method:'POST',body:JSON.stringify({profile:$('profile').value})}); toast('모델 저장됨'); }catch(e){} }
async function setConf(){ let v=parseFloat($('conf').value);
  if(isNaN(v)) return; v=Math.min(0.95,Math.max(0.05,v)); $('conf').value=v;
  await fetch('/api/settings',{method:'POST',body:JSON.stringify({conf:v})}); toast('신뢰도 저장됨 '+v.toFixed(2)); }
async function setGest(){ await fetch('/api/settings',{method:'POST',body:JSON.stringify({gesture_enabled:$('gest').checked})}); toast('저장됨'); }
async function setFlip(){ await fetch('/api/settings',{method:'POST',body:JSON.stringify({flip:$('flip').checked})}); toast($('flip').checked?'좌우반전 켜짐':'좌우반전 꺼짐'); }
async function setHold(){ let h=parseInt($('hold').value); if(isNaN(h))return; h=Math.min(30,Math.max(1,h)); $('hold').value=h;
  await fetch('/api/settings',{method:'POST',body:JSON.stringify({hold:h})}); toast('확정 카운트 '+h+'회 저장됨'); }
async function setMotion(){ const b={};
  let hs=parseFloat($('hold_sec').value), rs=parseFloat($('refr_sec').value);
  if(!isNaN(hs)){hs=Math.min(4,Math.max(0.5,hs)); $('hold_sec').value=hs; b.motion_hold_sec=hs;}
  if(!isNaN(rs)){rs=Math.min(5,Math.max(0.3,rs)); $('refr_sec').value=rs; b.motion_refractory_sec=rs;}
  if(!Object.keys(b).length)return;
  await fetch('/api/settings',{method:'POST',body:JSON.stringify(b)});
  toast('움직임 타이밍 저장됨 (정지 '+(b.motion_hold_sec??$('hold_sec').value)+'s / 대기 '+(b.motion_refractory_sec??$('refr_sec').value)+'s)'); }
async function saveLoc(){ const lat=parseFloat($('lat').value),lon=parseFloat($('lon').value);
  if(isNaN(lat)||isNaN(lon))return;
  await fetch('/api/settings',{method:'POST',body:JSON.stringify({lat,lon})}); toast('위치 저장됨'); }
let _map,_marker;
function _pick(ll){ $('lat').value=ll.lat.toFixed(4); $('lon').value=ll.lng.toFixed(4); saveLoc(); }
async function _center(){   // 저장된 위치 → IP 지오로케이션 → 기본
  let lat=parseFloat($('lat').value), lon=parseFloat($('lon').value), saved=true;
  if(isNaN(lat)||isNaN(lon)){ saved=false;
    try{const g=await (await fetch('/api/geoip')).json(); if(g&&g.lat){lat=g.lat;lon=g.lon;}}catch(e){}
  }
  if(isNaN(lat)||isNaN(lon)){lat=37.5;lon=127.0;}
  return {lat,lon,saved};
}
async function initMap(){
  if(_map||typeof L==='undefined'||!$('map'))return;
  const c=await _center();
  _map=L.map('map').setView([c.lat,c.lon],13);
  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',{maxZoom:19,attribution:'© OpenStreetMap'}).addTo(_map);
  _marker=L.marker([c.lat,c.lon],{draggable:true}).addTo(_map);
  _map.on('click',e=>{_marker.setLatLng(e.latlng);_pick(e.latlng);});
  _marker.on('dragend',()=>_pick(_marker.getLatLng()));
  if(!c.saved) _pick({lat:c.lat,lng:c.lon});   // IP로 잡은 초기 위치 저장
  setTimeout(()=>_map&&_map.invalidateSize(),120);
}
function onLatLon(){ const lat=parseFloat($('lat').value),lon=parseFloat($('lon').value);
  if(_map&&!isNaN(lat)&&!isNaN(lon)){_marker.setLatLng([lat,lon]);_map.setView([lat,lon]);} saveLoc(); }
function useGPS(){
  if(!navigator.geolocation){toast('이 기기는 GPS 미지원');return;}
  toast('GPS 위치 확인중…');
  navigator.geolocation.getCurrentPosition(p=>{
    const lat=p.coords.latitude, lon=p.coords.longitude;
    if(_map){_marker.setLatLng([lat,lon]);_map.setView([lat,lon],16);}
    _pick({lat,lng:lon}); toast('내 위치로 저장됨');
  }, e=>toast('GPS 실패: '+e.message), {enableHighAccuracy:true,timeout:8000});
}
const _ab=$('adminbox'); if(_ab) _ab.addEventListener('toggle',e=>{ if(e.target.open) initMap(); });
async function setRemote(){ const en=$('rem').checked; $('remurl').textContent=en?'터널 생성중… (몇 초)':'';
  await fetch('/api/remote',{method:'POST',body:JSON.stringify({enable:en})}); }
async function logout(){ await fetch('/api/logout',{method:'POST'}); location.href='/login'; }
async function changePw(){
  const r=await fetch('/api/password',{method:'POST',body:JSON.stringify({old:$('pw_old').value,new:$('pw_new').value})});
  const d=await r.json(); $('pwmsg').textContent=d.ok?'변경됨 — 다시 로그인하세요':(d.error||'실패');
  if(d.ok){setTimeout(()=>location.href='/login',1200);} }

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
   const r=await fetch('/api/state');
   if(r.status===401){location.href='/login';return;}
   const s=await r.json();
   $('conn').textContent='온라인'; $('conn').className='chip on';
   $('pwwarn').style.display=(s.auth&&s.auth.default_pw)?'block':'none';
   $('curtain').textContent=stMap[s.curtain.state]||s.curtain.state;
   $('motornote').style.display=s.curtain.motor_connected?'none':'block';
   if(typeof s.engine.event_seq==='number'){
     if(_lastEvt!==null && s.engine.event_seq>_lastEvt && s.engine.event_label) gestureBeep(s.engine.event_label);
     _lastEvt=s.engine.event_seq;
   }
   $('gesture').textContent=s.engine.gesture?(KR[s.engine.gesture]||s.engine.gesture):'—';
   $('model').textContent=s.engine.profile+' / '+s.engine.imgsz;
   $('perf').textContent=s.engine.infer_ms+'ms';
   $('count').textContent=s.engine.count;
   $('sys').textContent=s.system.load+' / '+s.system.mem_used_mb+'·'+s.system.mem_total_mb+'MB / '+(s.system.temp_c??'?')+'°C';
   renderSched(s.schedules||[]);
   if(s.remote){
     if(!$('rem').matches(':focus')) $('rem').checked=s.remote.active;
     $('remurl').innerHTML = s.remote.url
       ? '🔗 <a href="'+s.remote.url+'" target="_blank" style="color:var(--primary)">'+s.remote.url+'</a>'
       : (s.remote.active?'터널 생성중…':(s.remote.error||(s.remote.available?'꺼짐':'cloudflared 미설치')));
   }
   if(!profilesLoaded){
     const sel=$('profile'); sel.innerHTML='';
     s.profiles.forEach(p=>{const o=document.createElement('option');o.value=p.name;o.textContent=p.name+' ('+p.num_keypoints+'kp '+p.imgsz+')';sel.appendChild(o);});
     sel.value=s.engine.profile;
     $('conf').value=(+s.engine.conf).toFixed(2);
     $('hold').value=s.engine.hold;
     if(s.engine.motion_hold_sec!=null) $('hold_sec').value=s.engine.motion_hold_sec;
     if(s.engine.motion_refractory_sec!=null) $('refr_sec').value=s.engine.motion_refractory_sec;
     $('gest').checked=s.engine.gesture_enabled;
     $('flip').checked=!!s.engine.flip;
     if(s.location){$('lat').value=s.location.lat||''; $('lon').value=s.location.lon||'';}
     profilesLoaded=true;
   }
   $('profdesc').textContent=s.engine.profile_desc;
 }catch(e){ $('conn').textContent='오프라인'; $('conn').className='chip off'; }
}
$('sndbtn').textContent=_snd?'🔊':'🔇';
setInterval(poll,1000); poll();
if('serviceWorker' in navigator){navigator.serviceWorker.register('/sw.js').catch(()=>{});}
</script>
</body></html>
"""

LOGIN_HTML = """<!doctype html>
<html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="theme-color" content="#141218"><title>로그인 · AI 커튼</title>
<style>
 *{box-sizing:border-box}
 body{margin:0;height:100vh;display:flex;align-items:center;justify-content:center;
   background:#141218;color:#E6E1E9;font-family:Roboto,system-ui,"Noto Sans KR",sans-serif}
 .box{background:#211F26;padding:28px;border-radius:28px;width:320px;
   display:flex;flex-direction:column;gap:14px;text-align:center}
 h1{font-size:20px;margin:0}
 input{width:100%;height:52px;padding:0 16px;border-radius:12px;border:1px solid #49454F;
   background:#2B2930;color:#E6E1E9;font-size:16px}
 button{width:100%;height:52px;border:0;border-radius:100px;background:#D0BCFF;
   color:#381E72;font-weight:600;font-size:16px;cursor:pointer}
 .err{color:#F2B8B5;font-size:13px;min-height:16px;margin:0}
</style></head>
<body><div class="box">
 <h1>🪟 AI 커튼 제어</h1>
 <div class="err" id="err"></div>
 <input type="password" id="pw" placeholder="비밀번호" autofocus
   onkeydown="if(event.key==='Enter')go()">
 <button onclick="go()">로그인</button>
</div>
<script>
async function go(){
 const r=await fetch('/api/login',{method:'POST',body:JSON.stringify({password:document.getElementById('pw').value})});
 const d=await r.json();
 if(d.ok){location.href='/';} else {document.getElementById('err').textContent=d.error||'로그인 실패';}
}
</script></body></html>
"""

MANIFEST = json.dumps({
    "name": "AI 커튼 제어", "short_name": "AI커튼", "start_url": "/",
    "display": "standalone", "background_color": "#0e0e10",
    "theme_color": "#111", "icons": [],
})
SW_JS = "self.addEventListener('fetch',()=>{});"   # minimal (enables install)


def make_handler(proc, engine, controller, remote, auth_enabled=True):
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

        def _token(self):
            c = SimpleCookie(self.headers.get("Cookie", ""))
            return c["session"].value if "session" in c else None

        def _authed(self):
            return (not auth_enabled) or auth.valid(self._token())

        def do_GET(self):
            # public (no auth): login page, PWA assets
            if self.path in ("/login", "/login.html"):
                self._send(200, "text/html; charset=utf-8", LOGIN_HTML)
                return
            if self.path == "/manifest.json":
                self._send(200, "application/manifest+json", MANIFEST)
                return
            if self.path == "/sw.js":
                self._send(200, "application/javascript", SW_JS)
                return
            if self.path == "/healthz":
                self._json({"status": "ok",
                            "uptime_s": int(time.time() - _START),
                            "camera": proc.jpeg_slot.get() is not None,
                            "fps": round(proc.process_fps, 1),
                            "profile": engine.state()["profile"]})
                return
            if not self._authed():
                if self.path in ("/", "/index.html"):
                    self._send(200, "text/html; charset=utf-8", LOGIN_HTML)
                else:
                    self.send_error(401)
                return
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
                    "remote": remote.status(),
                    "auth": {"default_pw": auth.is_default()},
                })
            elif self.path == "/api/geoip":
                self._json(geoip() or {})
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
            # public: login
            if self.path == "/api/login":
                tok, err = auth.login(str(data.get("password", "")))
                if tok:
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Set-Cookie",
                                     f"session={tok}; HttpOnly; Path=/; SameSite=Lax; Max-Age=604800")
                    body = json.dumps({"ok": True}).encode()
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                else:
                    self._json({"ok": False, "error": err}, 401)
                return
            # everything else requires a valid session
            if not self._authed():
                self._json({"ok": False, "error": "unauthorized"}, 401)
                return
            if self.path == "/api/logout":
                auth.logout(self._token())
                self._json({"ok": True})
            elif self.path == "/api/password":
                ok, err = auth.change_password(str(data.get("old", "")),
                                               str(data.get("new", "")))
                self._json({"ok": ok, "error": err}, 200 if ok else 400)
            elif self.path == "/api/control":
                ok = controller.command(str(data.get("action", "")), "dashboard")
                self._json({"ok": ok, "curtain": controller.snapshot()})
            elif self.path == "/api/model":
                try:
                    prof = engine.set_profile(str(data.get("profile", "")))
                    store.set_setting("profile", prof.name)      # 영구 저장
                    self._json({"ok": True, "profile": prof.name})
                except Exception as e:
                    self._json({"ok": False, "error": str(e)}, 400)
            elif self.path == "/api/settings":
                if "conf" in data:
                    engine.set_conf(float(data["conf"]))
                    store.set_setting("conf", float(data["conf"]))
                if "gesture_enabled" in data:
                    g = bool(data["gesture_enabled"])
                    engine.set_gesture_enabled(g)
                    store.set_setting("gesture_enabled", "1" if g else "0")
                if "hold" in data:
                    h = max(1, int(data["hold"]))
                    engine.set_hold(h)
                    store.set_setting("hold", h)
                if "flip" in data:
                    f = bool(data["flip"])
                    engine.set_flip(f)
                    store.set_setting("flip", "1" if f else "0")
                if "motion_hold_sec" in data:
                    v = min(4.0, max(0.5, float(data["motion_hold_sec"])))
                    engine.set_motion_timing(hold_sec=v)
                    store.set_setting("motion_hold_sec", v)
                if "motion_refractory_sec" in data:
                    v = min(5.0, max(0.3, float(data["motion_refractory_sec"])))
                    engine.set_motion_timing(refractory_sec=v)
                    store.set_setting("motion_refractory_sec", v)
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
            elif self.path == "/api/remote":
                if data.get("enable"):
                    remote.start()
                else:
                    remote.stop()
                self._json({"ok": True, "remote": remote.status()})
            else:
                self.send_error(404)
    return H


def _env(k, d):
    return os.environ.get(k, d)


def parse_args():
    # defaults from config.env (CURTAIN_*) when present; flags override
    p = argparse.ArgumentParser(description="AI Curtain Control dashboard")
    p.add_argument("--profile", default=_env("CURTAIN_PROFILE", "hand_near"),
                   help="hand_near | body_far | body_motion")
    p.add_argument("--conf", type=float, default=float(_env("CURTAIN_CONF", "0.3")))
    p.add_argument("--width", type=int, default=int(_env("CURTAIN_WIDTH", "1280")))
    p.add_argument("--height", type=int, default=int(_env("CURTAIN_HEIGHT", "720")))
    p.add_argument("--fps", type=int, default=int(_env("CURTAIN_FPS", "30")))
    p.add_argument("--port", type=int, default=int(_env("CURTAIN_PORT", "8080")))
    p.add_argument("--host", default=_env("CURTAIN_HOST", "0.0.0.0"))
    p.add_argument("--quality", type=int, default=int(_env("CURTAIN_QUALITY", "75")))
    p.add_argument("--remote", action="store_true",
                   default=_env("CURTAIN_REMOTE", "") == "1",
                   help="start a Cloudflare quick tunnel (public unique URL) at launch")
    p.add_argument("--ble", action="store_true",
                   default=_env("CURTAIN_BLE", "") == "1",
                   help="start the BLE remote peripheral (Flutter app control)")
    p.add_argument("--no-auth", action="store_true",
                   default=_env("CURTAIN_AUTH", "1") == "0",
                   help="disable login (development convenience)")
    return p.parse_args()


def main():
    setup_logging()
    args = parse_args()
    store.init()
    auth.init()
    threading.Thread(target=autodetect_location, daemon=True).start()  # IP geo
    log.info("starting profile=%s port=%d remote=%s", args.profile, args.port, args.remote)
    if auth.is_default():
        log.warning("기본 비밀번호 'admin' 사용 중 — 대시보드에서 변경하세요.")
    controller = CurtainController()
    # saved settings (SQLite) take precedence over CLI/env defaults
    saved_profile = store.get_setting("profile") or args.profile
    try:
        saved_conf = float(store.get_setting("conf"))
    except (TypeError, ValueError):
        saved_conf = args.conf
    try:
        saved_hold = int(store.get_setting("hold"))
    except (TypeError, ValueError):
        saved_hold = 3

    def _fget(key):
        try:
            return float(store.get_setting(key))
        except (TypeError, ValueError):
            return None
    engine = PoseEngine(saved_profile, conf=saved_conf, controller=controller,
                        hold=saved_hold, flip=(store.get_setting("flip") == "1"),
                        motion_hold_sec=_fget("motion_hold_sec"),
                        motion_refractory_sec=_fget("motion_refractory_sec"))
    g = store.get_setting("gesture_enabled")
    if g is not None:
        engine.set_gesture_enabled(g == "1")
    log.info("settings: profile=%s conf=%.2f gesture=%s hold=%d",
             saved_profile, saved_conf, engine.gesture_enabled, engine.hold)
    scheduler = SchedulerThread(controller)
    scheduler.start()
    remote = RemoteManager(args.port)
    if args.remote:
        remote.start()
    ble = None
    if args.ble:
        try:
            from ble_server import BleServerThread
            ble = BleServerThread(controller)
            ble.start()
            log.info("BLE 리모컨 주변장치 시작 (AI-Curtain)")
        except Exception as e:
            log.warning("BLE 시작 실패: %s", e)

    pipeline = isp_gst_pipeline(width=args.width, height=args.height, fps=args.fps)
    cam = CameraThread(pipeline)
    proc = ProcessThread(cam.slot, jpeg_quality=args.quality,
                         process_fn=engine.process, fps_fn=draw_fps)
    cam.start()
    proc.start()

    if args.no_auth:
        log.warning("로그인 비활성화 (개발용) — 인증 없이 접근 가능")
    server = ThreadingHTTPServer((args.host, args.port),
                                 make_handler(proc, engine, controller, remote,
                                              auth_enabled=not args.no_auth))
    log.info("dashboard ready: http://<board-ip>:%d", args.port)
    print(f"\n  ▶ 대시보드 →  http://<board-ip>:{args.port}   (Ctrl+C 종료)\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("shutting down")
    finally:
        if ble:
            ble.stop()
        remote.stop(); scheduler.stop(); proc.stop(); cam.stop()
        server.shutdown(); engine.release()


if __name__ == "__main__":
    main()
