"""Standalone SwingTrace review and export app."""
from __future__ import annotations

from base64 import b64decode, b64encode
from hashlib import sha256
from pathlib import Path
import re
from shutil import copyfileobj
import tempfile
from uuid import uuid4

import cv2
import streamlit as st
import streamlit.components.v1 as components

from tracking import Detector, estimate_path
from trajectory import estimate_timing, render_trajectory
from video_io import (
    DEFAULT_TRAIL_STYLE, MAX_FRAMES, MAX_SECONDS, TrailStyle,
    VideoInfo, analyze, export_video, normalize, open_video, preview_jpeg, run_ffmpeg,
)

st.set_page_config(page_title="SwingTrace クラブヘッド追跡", page_icon="⛳", layout="wide")
st.markdown("""
<style>
html, [data-testid="stMain"] { scrollbar-width: auto; }
html::-webkit-scrollbar:vertical,
[data-testid="stMain"]::-webkit-scrollbar:vertical { width: 22px; }
html::-webkit-scrollbar-track:vertical,
[data-testid="stMain"]::-webkit-scrollbar-track:vertical { background: #e8eeec; }
html::-webkit-scrollbar-thumb:vertical,
[data-testid="stMain"]::-webkit-scrollbar-thumb:vertical {
    background: #648478;
    border: 2px solid #e8eeec;
    border-radius: 12px;
}
@media (pointer: coarse) {
    [data-testid="stMainBlockContainer"] { padding-right: max(2.5rem, env(safe-area-inset-right)); }
}
</style>
""", unsafe_allow_html=True)

mobile_scrollbar = st.components.v2.component(
    "swingtrace_mobile_scrollbar",
    html="""<div id="rail" role="scrollbar" aria-label="ページの縦スクロール"
         aria-orientation="vertical" aria-valuemin="0" aria-valuemax="100"
         aria-valuenow="0" tabindex="0" hidden><div id="thumb"></div></div>""",
    css="""
    #rail {
        position: fixed;
        top: 4rem;
        bottom: max(8px, env(safe-area-inset-bottom));
        right: max(3px, env(safe-area-inset-right));
        width: 28px;
        z-index: 10;
        background: #e8eeec;
        border-radius: 14px;
        touch-action: none;
        user-select: none;
    }
    #rail[hidden] { display: none; }
    #thumb {
        position: absolute;
        left: 3px;
        right: 3px;
        background: #648478;
        border-radius: 12px;
    }
    """,
    js=r"""
    export default function ({parentElement}) {
        const rail = parentElement.querySelector("#rail");
        const thumb = parentElement.querySelector("#thumb");
        const main = document.querySelector('[data-testid="stMain"]');
        const page = document.scrollingElement;
        const touch = matchMedia("(pointer: coarse)");
        const listeners = new AbortController();
        const options = {signal: listeners.signal};
        let scroller = main || page;
        let maxScroll = 0;
        let travel = 0;
        let drag = null;

        function sync() {
            scroller = main && main.scrollHeight > main.clientHeight + 1 ? main : page;
            maxScroll = Math.max(0, scroller.scrollHeight - scroller.clientHeight);
            rail.hidden = !touch.matches || maxScroll < 2;
            if (rail.hidden) return;
            const height = rail.clientHeight;
            const thumbHeight = Math.min(height, Math.max(44, height * scroller.clientHeight / scroller.scrollHeight));
            travel = height - thumbHeight;
            const fraction = Math.max(0, Math.min(1, scroller.scrollTop / maxScroll));
            thumb.style.height = thumbHeight + "px";
            thumb.style.top = fraction * travel + "px";
            rail.setAttribute("aria-valuenow", Math.round(fraction * 100));
        }

        function scrollAt(clientY) {
            if (!travel) return;
            const top = clientY - rail.getBoundingClientRect().top - drag.offset;
            scroller.scrollTop = maxScroll * Math.max(0, Math.min(1, top / travel));
            sync();
        }

        rail.addEventListener("pointerdown", event => {
            if (event.button !== 0 || drag || !travel) return;
            event.preventDefault();
            const box = thumb.getBoundingClientRect();
            const onThumb = event.clientY >= box.top && event.clientY <= box.bottom;
            drag = {id: event.pointerId, offset: onThumb ? event.clientY - box.top : box.height / 2};
            rail.setPointerCapture(event.pointerId);
            if (!onThumb) scrollAt(event.clientY);
        }, options);
        rail.addEventListener("pointermove", event => {
            if (drag?.id === event.pointerId) scrollAt(event.clientY);
        }, options);
        rail.addEventListener("lostpointercapture", event => {
            if (drag?.id === event.pointerId) drag = null;
        }, options);
        rail.addEventListener("keydown", event => {
            const steps = {
                ArrowUp: -40, ArrowDown: 40, PageUp: -scroller.clientHeight,
                PageDown: scroller.clientHeight, Home: -maxScroll, End: maxScroll
            };
            if (!Object.hasOwn(steps, event.key)) return;
            event.preventDefault();
            scroller.scrollTop += steps[event.key];
            sync();
        }, options);
        document.addEventListener("scroll", sync, {...options, capture: true});
        window.addEventListener("resize", sync, options);
        touch.addEventListener("change", sync, options);
        const observer = new ResizeObserver(sync);
        observer.observe(main || page);
        const content = document.querySelector('[data-testid="stMainBlockContainer"]');
        if (content) observer.observe(content);
        sync();
        return () => {
            observer.disconnect();
            listeners.abort();
            rail.hidden = true;
        };
    }
    """,
)
mobile_scrollbar(key="mobile_page_scrollbar")

# The component is embedded so deployment only needs this Python file for the editor.
CLICK_FRAME_HTML = r'''<!doctype html>
<html lang="ja">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>修正するクラブヘッドの位置</title>
<style>
  html, body { margin: 0; }
  #frame { position: relative; margin: 0 auto; width: 100%; }
  img { display: block; width: 100%; cursor: crosshair; touch-action: manipulation;
        user-select: none; border-radius: 7px; }
  #review-controls { position: absolute; inset: 6px 6px auto; box-sizing: border-box;
                     color: #fff; background: rgba(16, 25, 24, .86); border-radius: 7px;
                     padding: 2px 6px 4px; font: 14px system-ui, sans-serif; }
  #review-controls.bottom { inset: auto 6px 6px; }
  #review-controls[hidden] { display: none; }
  .review-row { display: flex; align-items: center; gap: 6px; }
  .review-status { justify-content: space-between; min-height: 30px; padding-left: 4px; }
  button { min-width: 44px; min-height: 38px; border: 1px solid #889d94;
           border-radius: 5px; color: inherit; background: #293c35; font: inherit; cursor: pointer; }
  button:disabled { opacity: .45; cursor: default; }
  #move-controls { min-height: 30px; border: 0; background: transparent; }
  #position { flex: 1; min-width: 0; margin: 0; height: 38px;
              accent-color: #9aefc9; cursor: pointer; touch-action: none; }
  @media (pointer: coarse) {
    button, #position { min-height: 44px; }
  }
</style>
<div id="frame">
  <img id="photo" alt="スイング動画の確認コマ" draggable="false">
  <div id="review-controls" role="group" aria-label="コマの移動" hidden>
    <div class="review-row review-status">
      <span id="frame-number"></span>
      <button id="move-controls" type="button" aria-label="操作バーを下へ移動">下へ</button>
    </div>
    <div class="review-row">
      <button id="previous" type="button" aria-label="前のコマ">◀</button>
      <input id="position" type="range" min="1" step="1" aria-label="コマ番号">
      <button id="next" type="button" aria-label="次のコマ">▶</button>
    </div>
  </div>
</div>
<script>
  const photo = document.getElementById("photo");
  const frame = document.getElementById("frame");
  const controls = document.getElementById("review-controls");
  const position = document.getElementById("position");
  const number = document.getElementById("frame-number");
  const previous = document.getElementById("previous");
  const next = document.getElementById("next");
  const moveControls = document.getElementById("move-controls");
  let token = null;
  let ready = false;
  let currentFrame = 1;
  let pendingSeek = null;
  const send = (type, fields = {}) =>
    window.parent.postMessage({ isStreamlitMessage: true, type, ...fields }, "*");
  const sendValue = (value) =>
    send("streamlit:setComponentValue", { dataType: "json", value });
  const eventId = () => Date.now().toString(36) + Math.random().toString(36).slice(2);
  const resize = () => send("streamlit:setFrameHeight", {
    height: Math.ceil(frame.getBoundingClientRect().height) + 1
  });
  function updateControls() {
    number.textContent = position.value + " / " + position.max + " コマ";
    previous.disabled = Number(position.value) <= 1;
    next.disabled = Number(position.value) >= Number(position.max);
  }
  function seek(target) {
    if (token === null) return;
    position.value = target;
    updateControls();
    if (pendingSeek === null && Number(position.value) === currentFrame) return;
    pendingSeek = eventId();
    sendValue({ kind: "seek", id: pendingSeek, frame: Number(position.value) });
  }
  position.addEventListener("input", updateControls);
  position.addEventListener("change", () => seek(position.value));
  previous.addEventListener("click", () => seek(Number(position.value) - 1));
  next.addEventListener("click", () => seek(Number(position.value) + 1));
  moveControls.addEventListener("click", () => {
    const direction = controls.classList.toggle("bottom") ? "上へ" : "下へ";
    moveControls.textContent = direction;
    moveControls.setAttribute("aria-label", "操作バーを" + direction + "移動");
  });

  window.addEventListener("message", ({ source, data }) => {
    if (source !== window.parent || data?.type !== "streamlit:render") return;
    const args = data.args;
    currentFrame = args.frame;
    position.max = args.total_frames;
    controls.hidden = args.total_frames < 2;
    if (args.ack === pendingSeek) pendingSeek = null;
    if (pendingSeek === null) position.value = currentFrame;
    updateControls();
    const url = "data:image/jpeg;base64," + args.image;
    if (token !== args.token || photo.getAttribute("src") !== url) {
      ready = false;
      token = args.token;
      frame.style.maxWidth = Math.round(Math.min(960, 720 * args.aspect)) + "px";
      photo.style.aspectRatio = args.aspect;
      photo.src = url;
      ready = photo.complete && photo.naturalWidth > 0;
    }
    resize();
  });
  photo.addEventListener("load", () => { ready = true; resize(); });
  photo.addEventListener("click", event => {
    if (!ready || token === null || pendingSeek !== null) return;
    const bounds = photo.getBoundingClientRect();
    if (!bounds.width || !bounds.height) return;
    const unit = value => Math.max(0, Math.min(1, value));
    sendValue({
      kind: "mark", token, id: eventId(),
      x: unit((event.clientX - bounds.left) / bounds.width),
      y: unit((event.clientY - bounds.top) / bounds.height)
    });
  });
  new ResizeObserver(resize).observe(frame);
  send("streamlit:componentReady", { apiVersion: 1 });
  resize();
</script>
</html>
'''


@st.cache_resource(show_spinner=False)
def click_frame_assets(html: str) -> tempfile.TemporaryDirectory:
    directory = tempfile.TemporaryDirectory(prefix="swingtrace-click-")
    (Path(directory.name) / "index.html").write_text(html, encoding="utf-8")
    return directory


# Change the iframe URL when its HTML changes, including after a hot reload.
review_version = sha256(CLICK_FRAME_HTML.encode("utf-8")).hexdigest()[:12]
click_frame = components.declare_component(
    f"swingtrace_click_frame_{review_version}", path=click_frame_assets(CLICK_FRAME_HTML).name
)
MAX_UPLOAD_MB = 200
MOBILE_CHUNK_BYTES = 256 * 1024
MOBILE_MAX_SECONDS = 60
MOBILE_MAX_FPS = 60

MOBILE_UPLOAD_HTML = r"""<!doctype html><html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>
body{margin:0;font-family:Arial,'Yu Gothic',sans-serif;color:#19382c}*{box-sizing:border-box}.box{padding:12px;background:#f0f4ef;border:1px solid #cbd5c6;border-radius:8px}button{font:inherit;padding:12px;border:1px solid #9caf98;border-radius:6px;background:white;color:#19382c;cursor:pointer;margin:4px 4px 4px 0}button:disabled{opacity:.45;cursor:default}#send{background:#247052;color:white}#name{margin:10px 0;overflow-wrap:anywhere;font-weight:bold}#status{font-size:14px;line-height:1.6;white-space:pre-wrap}progress{width:100%;height:18px}small{display:block;margin:8px 0;color:#52644c;line-height:1.6}
</style><div class="box" id="root"><button id="choose">① ファイルから__MEDIA__を選ぶ</button><button id="gallery">__GALLERY__</button><input type="file" id="files" hidden><input type="file" id="videos" accept="__ACCEPT__" hidden><div id="name">__MEDIA__はまだ選ばれていません</div><button id="send" disabled>② __MEDIA__を送信</button><progress id="progress" value="0" max="100"></progress><div id="status" role="status">端末内に保存した__MEDIA__を選んでください（__LIMIT__MBまで）</div><small>__MEDIA__名が出たら「② __MEDIA__を送信」を押してください__AFTER_SEND__</small></div><script>
const $=id=>document.getElementById(id);let file=null,busy=false,args={},pending=null,picker=false;
function post(type,extra={}){window.parent.postMessage({isStreamlitMessage:true,type,...extra},'*');}
function height(){post('streamlit:setFrameHeight',{height:Math.ceil($('root').getBoundingClientRect().height)+8});}
function status(t){$('status').textContent=t;height();}
function controls(){for(const id of ['choose','gallery','files','videos'])$(id).disabled=busy;$('send').disabled=busy||!file;}
for(const [button,input] of [['choose','files'],['gallery','videos']]){$(button).onclick=()=>{picker=true;$(input).value='';$(input).click();};$(input).onchange=()=>{picker=false;const f=$(input).files[0];if(!f)return;file=f;$('name').textContent=f.name+'（'+(f.size/1048576).toFixed(1)+' MB）';$('progress').value=0;if(!f.size||f.size>__LIMIT__*1048576){file=null;status('空の__MEDIA__、または__LIMIT__MBを超える__MEDIA__です。別の__MEDIA__を選んでください。');}else status('__MEDIA__を選択しました。「② __MEDIA__を送信」を押してください。');controls();height();};}
window.addEventListener('focus',()=>{if(picker)setTimeout(()=>{if(picker){picker=false;status('__MEDIA__名がまだ返っていません。「① ファイルから__MEDIA__を選ぶ」で端末内の__MEDIA__を選んでください。');}},2000);});
function send(value,expected){return new Promise((resolve,reject)=>{let attempts=0;const transmit=()=>{post('streamlit:setComponentValue',{dataType:'json',value});};const timer=setInterval(()=>{if(++attempts>=3){clearInterval(timer);pending=null;reject(Error('応答がありません。アプリを開き直してから、もう一度送信してください。'));}else{status('接続を確認して再送しています…');transmit();}},15000);pending={id:value.id,expected,resolve:()=>{clearInterval(timer);pending=null;resolve();},reject:e=>{clearInterval(timer);pending=null;reject(Error(e));}};transmit();});}
function read(slice){return new Promise((resolve,reject)=>{const r=new FileReader();r.onload=()=>resolve(String(r.result).split(',')[1]);r.onerror=()=>reject(Error('この__MEDIA__を端末から読み出せません。端末内へ保存した__MEDIA__で試してください。'));r.readAsDataURL(slice);});}
$('send').onclick=async()=>{if(!file||busy)return;const chosen=file,id=Date.now().toString(36)+'-'+Math.random().toString(36).slice(2);busy=true;controls();try{status('送信を開始しています…');await send({kind:'begin',id,name:chosen.name,size:chosen.size},0);let seq=0;for(let offset=0;offset<chosen.size;offset+=262144){const end=Math.min(chosen.size,offset+262144),data=await read(chosen.slice(offset,end));status('__MEDIA__を送信中… '+Math.floor(offset/chosen.size*100)+'%');await send({kind:'chunk',id,seq,data},seq+1);seq++;$('progress').value=Math.round(end/chosen.size*100);}status('受信完了__AFTER_SEND__');}catch(e){status(e.message);}finally{busy=false;controls();}};
window.addEventListener('message',e=>{if(e.source!==window.parent||e.data.type!=='streamlit:render')return;args=e.data.args||{};if(pending&&args.transfer_id===pending.id){if(args.error)pending.reject(args.error);else if(args.next_seq>=pending.expected)pending.resolve();}if(!file&&args.complete){$('name').textContent='受信済み：'+args.name;$('progress').value=100;status('受信完了__AFTER_SEND__');}height();});
new ResizeObserver(height).observe($('root'));post('streamlit:componentReady',{apiVersion:1});height();
</script></html>"""

st.title("SwingTrace クラブヘッド追跡")
st.caption("スイング動画を選択 → 自動追跡 → 外れたコマをクリックで修正 → MP4を保存")

STATUS = {
    "head": "AI検出",
    "shaft": "シャフトから推定",
    "interpolated": "前後のコマから補間",
    "missing": "未検出",
    "manual": "手動修正済み",
    "excluded": "軌道から除外",
}


@st.cache_resource(show_spinner=False)
def get_detector() -> Detector:
    return Detector()


def upload_error(upload) -> str | None:
    if upload.size == 0:
        return "動画が空です。端末に保存した動画を選び直してください。"
    if upload.size > MAX_UPLOAD_MB * 1024 * 1024:
        return f"動画の容量が上限を超えています。{MAX_UPLOAD_MB} MB以下に短くして選び直してください。"
    return None


def accept_mobile_upload(state: dict, event) -> bool:
    if not isinstance(event, dict):
        return False
    transfer_id = event.get("id")
    if not isinstance(transfer_id, str) or not re.fullmatch(r"[A-Za-z0-9-]{1,80}", transfer_id):
        return False
    kind = event.get("kind")
    if kind == "begin":
        if state.get("id") == transfer_id:
            return False
        size, name = event.get("size"), event.get("name")
        if type(size) is not int or not 0 < size <= MAX_UPLOAD_MB * 1024 * 1024:
            return False
        if not isinstance(name, str) or not 0 < len(name) <= 512:
            return False
        directory = tempfile.TemporaryDirectory(prefix="swingtrace-upload-")
        state.clear()
        state.update(id=transfer_id, name=name, size=size, directory=directory,
                     path=Path(directory.name) / "video", received=0, next_seq=0)
        return True
    if kind != "chunk" or state.get("id") != transfer_id or state.get("error"):
        return False
    seq = event.get("seq")
    if type(seq) is not int or seq != state["next_seq"]:
        return False
    try:
        encoded = event.get("data")
        if not isinstance(encoded, str) or len(encoded) > 4 * ((MOBILE_CHUNK_BYTES + 2) // 3):
            raise ValueError("送信データのサイズが不正です。")
        chunk = b64decode(encoded, validate=True)
        expected = min(MOBILE_CHUNK_BYTES, state["size"] - state["received"])
        if not expected or len(chunk) != expected:
            raise ValueError("動画の一部を受信できませんでした。もう一度送信してください。")
        with state["path"].open("ab") as output:
            output.write(chunk)
        state["next_seq"] += 1
        state["received"] += len(chunk)
    except ValueError as error:
        state["error"] = str(error)
    except OSError:
        state["error"] = "サーバーに動画を保存できませんでした。"
    return True


class MobileUpload:
    def __init__(self, state: dict):
        self.name = state["name"]
        self.size = state["size"]
        self.path = state["path"]


@st.cache_resource(show_spinner=False)
def mobile_upload_assets() -> tempfile.TemporaryDirectory:
    html = (MOBILE_UPLOAD_HTML.replace("__MEDIA__", "動画")
            .replace("__LIMIT__", str(MAX_UPLOAD_MB))
            .replace("__GALLERY__", "写真・動画から選ぶ")
            .replace("__ACCEPT__", "video/*")
            .replace("__AFTER_SEND__", "。下の「自動追跡する」を押してください。"))
    directory = tempfile.TemporaryDirectory(prefix="swingtrace-mobile-picker-")
    (Path(directory.name) / "index.html").write_text(html, encoding="utf-8")
    return directory


mobile_picker = components.declare_component(
    "swingtrace_mobile_import", path=mobile_upload_assets().name
)


def mobile_upload() -> MobileUpload | None:
    state = st.session_state.setdefault("swingtrace_mobile_import", {})
    complete = "size" in state and state["received"] == state["size"]
    event = mobile_picker(
        transfer_id=state.get("id"), next_seq=state.get("next_seq", -1),
        complete=complete, error=state.get("error"), name=state.get("name", ""),
        key="swingtrace_mobile_upload_widget", default=None,
    )
    if accept_mobile_upload(state, event):
        st.rerun()
    if state.get("error"):
        st.error(state["error"])
    return MobileUpload(state) if complete and not state.get("error") else None


def normalize_mobile(source: Path, output: Path) -> VideoInfo:
    # Match the working golf-swing-python import profile while retaining audio for export.
    scale = ("scale=w='min(960,iw)':h='min(960,ih)':force_original_aspect_ratio=decrease:"
             f"force_divisible_by=2,setsar=1,fps='min(source_fps,{MOBILE_MAX_FPS})'")
    run_ffmpeg([
        "-nostdin", "-threads", "2", "-i", str(source), "-map", "0:v:0", "-map", "0:a:0?",
        "-t", str(MOBILE_MAX_SECONDS + 1), "-filter_threads", "2", "-vf", scale,
        "-c:v", "libx264", "-threads:v", "2", "-preset", "veryfast", "-crf", "20",
        "-pix_fmt", "yuv420p", "-c:a", "aac", "-movflags", "+faststart", str(output),
    ])
    with open_video(output) as capture:
        info = VideoInfo(
            capture.get(cv2.CAP_PROP_FPS), int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        )
        if info.frames < 1 or not 0 < info.fps <= MOBILE_MAX_FPS or not capture.grab():
            raise RuntimeError("動画の映像やフレーム情報を取得できませんでした。")
    if info.frames / info.fps > MOBILE_MAX_SECONDS + 0.05:
        raise RuntimeError(f"動画を{MOBILE_MAX_SECONDS}秒以内に切り出してください。")
    return info


def start_job(upload, image_size: int):
    job_dir = tempfile.TemporaryDirectory(prefix="swingtrace-")
    root = Path(job_dir.name)
    suffix = Path(upload.name).suffix.lower()
    input_path = root / ("original" + (suffix if re.fullmatch(r"\.[a-z0-9]{1,8}", suffix) else ".mp4"))
    source = root / "normalized.mp4"
    try:
        mobile = isinstance(upload, MobileUpload)
        with input_path.open("wb") as file:
            if mobile:
                with upload.path.open("rb") as data:
                    copyfileobj(data, file)
            else:
                with upload.getbuffer() as data:
                    file.write(data)
        with st.spinner("動画を読み込んでいます…"):
            try:
                info = (normalize_mobile if mobile else normalize)(input_path, source)
            except RuntimeError as error:
                raise RuntimeError(f"動画は受信済みですが、読み込みに失敗しました。{error}") from error
        input_path.unlink()
        with st.spinner("SwingTraceモデルを準備中です。初回は約40MBを取得します…"):
            detector = get_detector()
        progress = st.progress(0, text="クラブヘッドを追跡しています…")
        try:
            detections = analyze(
                source, info, detector, image_size,
                lambda done, total: progress.progress(done / total, text=f"自動追跡中 · {done:,} / {total:,} コマ"),
            )
        finally:
            progress.empty()
    except Exception:
        job_dir.cleanup()
        raise
    return {
        "id": uuid4().hex, "temp": job_dir, "source": source, "info": info,
        "detections": detections, "image_size": image_size,
        "corrections": {}, "split": None, "end": None, "export": None,
        "filename": f"{Path(upload.name).stem}_swingtrace.mp4",
    }


def invalidate_export(job) -> None:
    if job["export"] is not None:
        job["export"].unlink(missing_ok=True)
        job["export"] = None


def edit(job, index, point) -> None:
    job["corrections"][index] = point
    invalidate_export(job)


def move_to(key: str, position: int) -> None:
    st.session_state[key] = position


def apply_review_event() -> None:
    job = st.session_state["job"]
    job_id = job["id"]
    event = st.session_state.get(f"click_{job_id}")
    if not isinstance(event, dict) or not isinstance(event.get("id"), str):
        return
    job["review_ack"] = event["id"]
    info = job["info"]
    frame_key = f"frame_{job_id}"
    if event.get("kind") == "seek":
        position = event.get("frame")
        if type(position) is int and 1 <= position <= info.frames:
            move_to(frame_key, position)
        return
    index = st.session_state.get(frame_key, 1) - 1
    if event.get("kind") != "mark" or event.get("token") != f"{job_id}:{index}":
        return
    x, y = event.get("x"), event.get("y")
    if all(type(value) in (int, float) and 0 <= value <= 1 for value in (x, y)):
        edit(job, index, (round(x * (info.width - 1)), round(y * (info.height - 1))))


with st.container():
    st.subheader("動画を選択")
    android = "android" in st.context.headers.get("User-Agent", "").lower()
    method = st.radio("動画の取り込み方法", ("通常（PCなど）", "Android向け（分割送信）"),
                      index=int(android), horizontal=True)
    if method == "Android向け（分割送信）":
        upload = mobile_upload()
        st.caption(f"上限{MAX_UPLOAD_MB} MB・{MOBILE_MAX_SECONDS}秒。"
                   f"解析用の動画は長辺960 px・最大{MOBILE_MAX_FPS} fpsに調整します。")
    else:
        st.session_state.pop("swingtrace_mobile_import", None)
        upload = st.file_uploader("動画を選択", type=None)
        st.caption(f"上限{MAX_UPLOAD_MB} MB・{MAX_SECONDS}秒・{MAX_FRAMES:,}コマ。"
                   "ファイル名が出ない場合は「Android向け（分割送信）」を選んでください。")
    upload_ready = False
    if upload is not None:
        error = upload_error(upload)
        upload_ready = error is None
        if error:
            st.error(error)
        else:
            st.success(f"動画を受け取りました（{upload.size / (1024 * 1024):.1f} MB）。「自動追跡する」を押してください。")
    image_size = st.radio("追跡モード", [896, 640, 480], format_func=lambda size: {
        896: "高精度（未検出のコマを再解析）", 640: "標準", 480: "高速",
    }[size], horizontal=True)
    st.caption("高精度モードは処理に時間がかかります。まずは短いスイング動画でお試しください。")
    scope = st.radio(
        "軌道の描画範囲", ("stop", "follow"),
        format_func=lambda value: {
            "stop": "体に隠れるまで",
            "follow": "隠れた区間も補間してフォローまで",
        }[value], horizontal=True, key="drawing_scope",
    )
    with st.expander("軌道線の色と太さ"):
        colors = st.columns(2)
        before_color = colors[0].color_picker("切り替え前の色", DEFAULT_TRAIL_STYLE.before)
        after_color = colors[1].color_picker("切り替え後の色", DEFAULT_TRAIL_STYLE.after)
        thickness = st.slider("線の太さ（短辺1080px換算）", 1, 16, DEFAULT_TRAIL_STYLE.thickness,
                              format="%d px")
    style = TrailStyle(before_color, after_color, thickness)
    style_key = (style.before, style.after, style.thickness)
    auto_export = st.checkbox("解析後に軌道付きMP4も自動で作成", value=True)
    loaded = st.session_state.get("job")
    if st.button("自動追跡する", disabled=not upload_ready, type="primary"):
        try:
            new_job = start_job(upload, image_size)
        except Exception as error:
            st.error(f"自動追跡を開始または完了できませんでした。{error}")
        else:
            if loaded is not None:
                loaded["temp"].cleanup()
                st.session_state.pop(f"frame_{loaded['id']}", None)
            new_job["auto_export_pending"] = auto_export
            new_job["drawing_scope"] = scope
            new_job["trail_style"] = style_key
            st.session_state.job = new_job

job = st.session_state.get("job")
if job is None:
    st.stop()

info = job["info"]
path = estimate_path(job["detections"], job["corrections"], info.fps, info.width, info.height)
suggestion = estimate_timing(path, info.fps, info.height)
if job.get("drawing_scope", "stop") != scope:
    invalidate_export(job)
    job["drawing_scope"] = scope
    job["auto_export_pending"] = auto_export

if job.get("trail_style") != style_key:
    old_style = job.get("trail_style")
    job["trail_style"] = style_key
    if old_style is not None:
        invalidate_export(job)
        if not job.get("auto_export_pending"):
            job["style_changed"] = True

include_follow = scope == "follow"
end_key = "follow_end" if include_follow else "end"
last_seen = next((i for i in reversed(range(len(path))) if path[i].point is not None), suggestion.end)
default_end = last_seen
if not include_follow:
    default_end = suggestion.end if suggestion.occlusion_found else -1
manual_end = job.get(end_key)
end = default_end if manual_end is None else manual_end
render_path = render_trajectory(
    path, info.fps, info.width, info.height, end,
    occlusion_end=suggestion.end if include_follow and suggestion.occlusion_found else None,
)
split = job.get("split")
split = suggestion.split if split is None else split
split = split if split is not None and split <= end else None
suspects = [i for i, sample in enumerate(path) if sample.suspect]
manual_count = sum(sample.source == "manual" for sample in path)
frame_key = f"frame_{job['id']}"
index = st.session_state.get(frame_key, 1) - 1
current = path[index]
review_sample = render_path[index] if current.source == "missing" else current
st.divider()
st.subheader("コマを確認・修正")
st.caption(f"{job['filename']} · {info.frames:,}コマ · {info.fps:g} fps · 要確認 {len(suspects):,} · 手動修正 {manual_count:,}")
direct = sum(sample.source == "head" for sample in path)
estimated = sum(sample.source in ("shaft", "interpolated") for sample in path)
missing = sum(sample.source == "missing" for sample in render_path[:end + 1])
auto_filled = sum(before.point is None and after.point is not None
                  for before, after in zip(path[:end + 1], render_path[:end + 1]))
st.caption(f"ヘッド直接検出 {direct:,} · シャフト推定・補間 {estimated:,} · 曲線で補間 {auto_filled:,} · 未検出 {missing:,}")
if all(sample.point is None for sample in path):
    hint = ("高精度モードで再解析するか、" if job.get("image_size", 640) < 896 else "")
    st.warning(f"クラブヘッドの位置を特定できませんでした。{hint}見えるコマをクリックして位置を追加できます。")

show_trail = st.session_state.get("review_show_trail", True)
try:
    jpeg = preview_jpeg(job["source"], render_path, index, show_trail,
                        split=split, end=end, review_sample=review_sample, style=style)
except RuntimeError as error:
    st.error(str(error))
else:
    click_frame(
        image=b64encode(jpeg).decode("ascii"), token=f"{job['id']}:{index}",
        frame=index + 1, total_frames=info.frames, ack=job.get("review_ack"),
        aspect=info.width / info.height, key=f"click_{job['id']}", on_change=apply_review_event,
    )

if info.frames > 1:
    slider_key = f"review_slider_{job['id']}"
    if st.session_state.get(slider_key) != index + 1:
        st.session_state[slider_key] = index + 1
    with st.expander("プレビューの下でコマを調整"):
        st.slider(
            "コマ番号", 1, info.frames, key=slider_key,
            on_change=lambda: move_to(frame_key, st.session_state[slider_key]),
        )

previous = next((i for i in reversed(suspects) if i < index), None)
following = next((i for i in suspects if i > index), None)
nav = st.columns(2)
for column, label, target in zip(nav, ("前の要確認", "次の要確認"), (previous, following)):
    column.button(label, disabled=target is None, on_click=move_to if target is not None else None,
                  args=(frame_key, target + 1) if target is not None else None, width="stretch")
st.caption(f"{index + 1:,} / {info.frames:,} コマ · {index / info.fps:.3f}秒 · {STATUS[review_sample.source]}"
           + ("（要確認）" if current.suspect else ""))
st.write("画像上のクラブヘッドの中心をクリックして修正します。描画範囲内で検出が途切れた部分は自動で曲線をつなぎます。")

controls = st.columns(3)
controls[0].checkbox("確認画像に軌道を表示", value=True, key="review_show_trail")
if controls[1].button("このコマを除外", disabled=current.source == "excluded", width="stretch"):
    edit(job, index, None)
    st.rerun()
if controls[2].button("このコマの修正を戻す", disabled=index not in job["corrections"], width="stretch"):
    del job["corrections"][index]
    invalidate_export(job)
    st.rerun()

st.subheader("軌道の色と描画範囲")
if split is None:
    st.caption("切り替え点を指定すると、そこから後半の色に変わります。")
else:
    st.caption(f"色の切り替え　{split + 1:,}コマ目（{split / info.fps:.2f}秒）"
               f"{' ・ 自動候補' if job.get('split') is None else ''}")
if include_follow and suggestion.occlusion_found and last_seen <= suggestion.end:
    st.caption("ヘッドが再び見えるコマを検出できた場合に、フォローまで描画します。")
if end < 0:
    st.warning("ヘッドが隠れる位置を自動で特定できませんでした。コマ番号でヘッドが隠れる直前に移動し、「このコマで描画終了」を押してください。")
else:
    end_label = " ・ 手動指定" if manual_end is not None else (
        " ・ 最後の検出点" if include_follow else " ・ 隠れる直前の自動候補"
    )
    st.caption(f"描画終了　{end + 1:,}コマ目（{end / info.fps:.2f}秒）{end_label}")
timing = st.columns(3)
if timing[0].button("このコマで色を切り替える", disabled=index >= end, width="stretch"):
    job["split"] = index
    invalidate_export(job)
    st.rerun()
if timing[1].button("このコマで描画終了", width="stretch"):
    job[end_key] = index
    invalidate_export(job)
    job["auto_export_pending"] = auto_export
    st.rerun()
if timing[2].button("色と終了を自動に戻す",
                    disabled=job.get("split") is None and manual_end is None, width="stretch"):
    job["split"] = None
    job[end_key] = None
    invalidate_export(job)
    job["auto_export_pending"] = auto_export
    st.rerun()
seek_controls = st.columns(2)
if split is not None:
    seek_controls[0].button("色の切り替えを確認", on_click=move_to,
                            args=(frame_key, split + 1), width="stretch")
if end >= 0:
    seek_controls[1].button("終了点を確認", on_click=move_to,
                            args=(frame_key, end + 1), width="stretch")

st.divider()
st.subheader("軌道付き動画")
st.caption("描画範囲内の未検出区間は前後の動きから推定します。描画終了後も軌道を表示したまま再生します。")
if missing:
    st.caption(f"描画範囲内に補間できなかったコマが {missing:,} 枚あります。")
can_export = end >= 0 and any(p.point is not None for p in render_path[:end + 1])
requested = st.button("軌道付きMP4を作成", type="primary", disabled=not can_export)
auto_requested = can_export and job.pop("auto_export_pending", False)
if can_export and (auto_requested or requested):
    target = job["source"].with_name("swingtrace.mp4")
    progress = st.progress(0, text="軌道付き動画を書き出しています…")
    try:
        export_video(
            job["source"], info, render_path, target,
            lambda done, total: progress.progress(done / total, text=f"MP4作成中 · {done:,} / {total:,} コマ"),
            split=split, end=end, style=style,
        )
        job["export"] = target
        job.pop("style_changed", None)
    except Exception as error:
        st.error(f"MP4を作成できませんでした。{error}")
    finally:
        progress.empty()
if job.get("style_changed") and job["export"] is None:
    st.caption("現在の色と太さで保存するには「軌道付きMP4を作成」を押してください。")
if job["export"] is not None:
    st.video(str(job["export"]), format="video/mp4")
    with job["export"].open("rb") as finished:
        st.download_button("軌道付きMP4を保存", finished, file_name=job["filename"],
                           mime="video/mp4", on_click="ignore")
