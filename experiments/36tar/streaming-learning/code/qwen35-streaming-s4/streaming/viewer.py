"""Loopback-only live view. Bounded state; no retained frames or tensors."""
from collections import deque
import base64
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import threading

from PIL import Image

HTML = '''<!doctype html><html lang="zh"><meta charset="utf-8">
<title>Qwen3.5 连续视频问答</title><style>
body{background:#101821;color:#e8eef5;font:16px system-ui;max-width:1000px;margin:32px auto;padding:16px}
img{width:100%;height:340px;object-fit:contain;background:#080d13;border-radius:12px}
.row{display:grid;grid-template-columns:1fr 1fr;gap:24px}pre{white-space:pre-wrap}li{margin:16px 0}
.note{color:#ffd391} @media(max-width:650px){.row{display:block}}
</style><h1>连续视频问答 · FP32 参考版本</h1>
<p id="scope"></p><p class="note">容量重置会清空历史。模型忙时可能跳过画面；回答仅依据已接入模型的帧。按时间送帧不保证回答实时。</p>
<div class="row"><div><img id="frame" alt="当前已到达画面"><pre id="status"></pre></div>
<div><h2>最近回答</h2><ol id="answers"></ol></div></div>
<script>function render(s){document.getElementById('scope').textContent=s.scope||'';
document.getElementById('frame').src=s.image||'';
let t=s.status||{},p=s.source||{};let sec=x=>x==null?'暂无':x.toFixed(2)+' 秒';
document.getElementById('status').textContent=`正在播放：${sec(p.timestamp)}\n最近接入模型：${sec(t.video_seconds)}\n当前记忆从：${sec(t.memory_from_seconds)}\n重置：${t.reset_count||0} 次\n丢帧：${p.dropped_samples||0} / 已采样 ${p.produced_samples||0}\n模型已收到：${t.received_frames||0} 帧\n${t.finished?'本次运行已结束':''}`;
let a=document.getElementById('answers');a.replaceChildren();for(let x of s.answers||[]){let li=document.createElement('li');
li.textContent=`依据视频 ${x.timestamp.toFixed(2)} 秒 / 记忆从 ${x.memory_from_pts.toFixed(2)} 秒开始${x.answer_lag_seconds==null?'':` / 回答滞后 ${x.answer_lag_seconds.toFixed(2)} 秒`}\n${x.question}\n${x.answer}`;a.append(li)}}
if(window.RECORDED){render(window.RECORDED)}else{setInterval(async()=>{try{render(await(await fetch('/state',{cache:'no-store'})).json())}catch(e){}},500)}
</script></html>'''


class Viewer:
    def __init__(self, scope):
        self.lock = threading.Lock()
        self.answers = deque(maxlen=8)
        self.image, self.status, self.scope = "", {}, scope
        self.source = {}
        self.server = self.thread = None

    def frame(self, rgb, timestamp=None, timing=None):
        stream = io.BytesIO()
        Image.fromarray(rgb).save(stream, format="JPEG", quality=80)
        with self.lock:
            self.image = "data:image/jpeg;base64," + base64.b64encode(stream.getvalue()).decode("ascii")
            self.source = {"timestamp":timestamp,**(timing or {})}

    def event(self, event):
        s = event["status"]
        state = {"event":event["kind"], "video_seconds":s["last_received_pts"],
                 "memory_from_seconds":s["memory_from_pts"], "reset_count":s["resets"],
                 "received_frames":s["received_real_frames"], "encoded_frames":s["encoded_real_frames"],
                 "history_tokens":s["session"]["seen_tokens"], "finished":s["finished"]}
        with self.lock:
            self.status = state
            if event["kind"] == "answer":
                self.answers.append({**event["answer"], "memory_from_pts":s["memory_from_pts"]})

    def snapshot(self):
        with self.lock:
            return {"scope":self.scope,"image":self.image,"status":dict(self.status),
                    "source":dict(self.source),"answers":list(self.answers)}

    def start(self, port):
        owner = self
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path not in ("/", "/state"):
                    self.send_error(404)
                    return
                data = HTML.encode("utf-8") if self.path == "/" else json.dumps(owner.snapshot(),ensure_ascii=False).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8" if self.path == "/" else "application/json; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            def log_message(self, *_):
                pass
        self.server = ThreadingHTTPServer(("127.0.0.1",port),Handler)
        self.thread = threading.Thread(target=self.server.serve_forever,daemon=True)
        self.thread.start()
        return f"http://127.0.0.1:{self.server.server_port}/"

    def save(self, path):
        # Escape '<' to prevent user questions/answers ending the script element.
        payload = json.dumps(self.snapshot(),ensure_ascii=False).replace("<", "\\u003c")
        path.write_text(HTML.replace("<script>", "<script>window.RECORDED="+payload+";</script><script>",1),encoding="utf-8")

    def close(self):
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()
            self.thread.join(timeout=2)
            self.server = None
