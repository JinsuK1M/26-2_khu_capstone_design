"""Loopback-only browser preview, accessed through SSH port forwarding."""
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading

HTML = '''<!doctype html><html lang="ko"><meta charset="utf-8">
<title>Orange Pi Camera + Backbone</title>
<style>body{background:#14181f;color:#eef;font:16px system-ui;max-width:1000px;margin:24px auto;padding:12px}
img{width:100%;background:#222}pre{white-space:pre-wrap}small{color:#abc}</style>
<h2>Orange Pi 카메라 · 전체 backbone</h2>
<p>특징 추출을 완료한 프레임입니다. 탐지 박스와 FCM은 포함하지 않습니다.</p>
<img id="view" alt="첫 프레임 수신 대기"><pre id="stats">연결 중…</pre>
<small>표시 속도에는 추론·JPEG 변환·네트워크·브라우저 지연이 포함됩니다.</small>
<script>
let previous;
async function refresh(){try {
 const s=await fetch('/status.json',{cache:'no-store'}).then(r=>r.json());
 document.getElementById('stats').textContent=JSON.stringify(s,null,2);
 const r=await fetch('/frame.jpg',{cache:'no-store'});
 if(r.ok){const url=URL.createObjectURL(await r.blob());
 document.getElementById('view').src=url;if(previous)URL.revokeObjectURL(previous);previous=url;}
} catch(e){document.getElementById('stats').textContent='연결 끊김: '+e.message;}
setTimeout(refresh,250);}refresh();
</script></html>'''


class Preview:
    def __init__(self, port):
        self.lock = threading.Lock()
        self.jpeg = None
        self.status = {'state': '카메라 프레임 대기'}
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                with owner.lock:
                    jpeg, status = owner.jpeg, dict(owner.status)
                if self.path == '/':
                    data, kind, code = HTML.encode(), 'text/html; charset=utf-8', 200
                elif self.path == '/status.json':
                    data, kind, code = json.dumps(status, ensure_ascii=False).encode(), 'application/json; charset=utf-8', 200
                elif self.path == '/frame.jpg':
                    data, kind, code = (jpeg or b''), 'image/jpeg', (200 if jpeg else 503)
                else:
                    data, kind, code = b'Not found', 'text/plain', 404
                self.send_response(code)
                self.send_header('Content-Type', kind)
                self.send_header('Cache-Control', 'no-store')
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                try:
                    self.wfile.write(data)
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def publish(self, jpeg, status):
        with self.lock:
            self.jpeg, self.status = jpeg, status

    def close(self):
        self.server.shutdown()
        self.server.server_close()
