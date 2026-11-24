"""Standalone SwingTrace review and export app."""
from __future__ import annotations

from base64 import b64encode
from pathlib import Path
import tempfile
from uuid import uuid4

import streamlit as st
import streamlit.components.v1 as components

from tracking import Detector, estimate_path
from trajectory import estimate_timing, smooth_path
from video_io import MAX_FRAMES, MAX_SECONDS, analyze, export_video, normalize, preview_jpeg

st.set_page_config(page_title="SwingTrace クラブヘッド追跡", page_icon="⛳", layout="wide")

# The component is embedded so deployment only needs this Python file for the editor.
CLICK_FRAME_HTML = r'''<!doctype html>
<html lang="ja">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>修正するクラブヘッドの位置</title>
<style>
  html, body { margin: 0; }
  #frame { position: relative; margin: 0 auto; width: 100%; line-height: 0; }
  img { display: block; width: 100%; cursor: crosshair; touch-action: manipulation;
        user-select: none; border-radius: 7px; }
</style>
<div id="frame"><img id="photo" alt="スイング動画の確認コマ" draggable="false"></div>
<script>
  const photo = document.getElementById("photo");
  const frame = document.getElementById("frame");
  let token = null;
  let ready = false;
  const send = (type, fields = {}) =>
    window.parent.postMessage({ isStreamlitMessage: true, type, ...fields }, "*");
  const resize = () => send("streamlit:setFrameHeight", {
    height: Math.ceil(frame.getBoundingClientRect().height) + 1
  });

  window.addEventListener("message", ({ source, data }) => {
    if (source !== window.parent || data?.type !== "streamlit:render") return;
    const args = data.args;
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
    if (!ready || token === null) return;
    const bounds = photo.getBoundingClientRect();
    if (!bounds.width || !bounds.height) return;
    const unit = value => Math.max(0, Math.min(1, value));
    send("streamlit:setComponentValue", {
      dataType: "json",
      value: {
        token, id: Date.now().toString(36) + Math.random().toString(36).slice(2),
        x: unit((event.clientX - bounds.left) / bounds.width),
        y: unit((event.clientY - bounds.top) / bounds.height)
      }
    });
  });
  new ResizeObserver(resize).observe(frame);
  send("streamlit:componentReady", { apiVersion: 1 });
  resize();
</script>
</html>
'''


@st.cache_resource(show_spinner=False)
def click_frame_assets() -> tempfile.TemporaryDirectory:
    directory = tempfile.TemporaryDirectory(prefix="swingtrace-click-")
    (Path(directory.name) / "index.html").write_text(CLICK_FRAME_HTML, encoding="utf-8")
    return directory


click_frame = components.declare_component(
    "swingtrace_click_frame", path=click_frame_assets().name
)
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


def start_job(upload, image_size: int):
    if upload.size > 100 * 1024 * 1024:
        raise RuntimeError("100 MB以下の動画に切り詰めてください。")
    job_dir = tempfile.TemporaryDirectory(prefix="swingtrace-")
    root = Path(job_dir.name)
    input_path = root / ("original" + Path(upload.name).suffix.lower())
    source = root / "normalized.mp4"
    try:
        with input_path.open("wb") as file, upload.getbuffer() as data:
            file.write(data)
        with st.spinner("動画を読み込んでいます…"):
            info = normalize(input_path, source)
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
        "corrections": {}, "split": None, "end": None, "export": None, "click": None,
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


with st.container():
    st.subheader("動画を選択")
    upload = st.file_uploader("スイング動画", type=["mp4", "mov", "m4v"], label_visibility="collapsed")
    st.caption(f"1人・1スイング。上限100 MB、{MAX_SECONDS}秒、{MAX_FRAMES:,}コマ。")
    image_size = st.radio("追跡モード", [896, 640, 480], format_func=lambda size: {
        896: "高精度（未検出のコマを再解析）", 640: "標準", 480: "高速",
    }[size], horizontal=True)
    st.caption("高精度モードは処理に時間がかかります。まずは短いスイング動画でお試しください。")
    auto_export = st.checkbox("解析後に軌道付きMP4も自動で作成", value=True)
    loaded = st.session_state.get("job")
    if st.button("自動追跡する", disabled=upload is None, type="primary"):
        try:
            new_job = start_job(upload, image_size)
        except Exception as error:
            st.error(f"自動追跡を開始または完了できませんでした。{error}")
        else:
            if loaded is not None:
                loaded["temp"].cleanup()
                st.session_state.pop(f"frame_{loaded['id']}", None)
            new_job["auto_export_pending"] = auto_export
            st.session_state.job = new_job

job = st.session_state.get("job")
if job is None:
    st.stop()

info = job["info"]
path = estimate_path(job["detections"], job["corrections"], info.fps, info.width, info.height)
suggestion = estimate_timing(path, info.fps, info.height)
end = job.get("end")
end = suggestion.end if end is None else end
render_path = smooth_path(path, info.fps, info.width, info.height, end)
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

previous = next((i for i in reversed(suspects) if i < index), None)
following = next((i for i in suspects if i > index), None)
nav = st.columns(4)
for column, label, target in zip(nav, ("前の要確認", "前のコマ", "次のコマ", "次の要確認"),
                                 (previous, index - 1, index + 1, following)):
    enabled = target is not None and 0 <= target < info.frames
    column.button(label, disabled=not enabled, on_click=move_to if enabled else None,
                  args=(frame_key, target + 1) if enabled else None, width="stretch")

if info.frames > 1:
    st.slider("コマ番号", 1, info.frames, key=frame_key)
st.caption(f"{index + 1:,} / {info.frames:,} コマ · {index / info.fps:.3f}秒 · {STATUS[review_sample.source]}"
           + ("（要確認）" if current.suspect else ""))
st.write("画像上のクラブヘッドの中心をクリックして修正します。描画範囲内で検出が途切れた部分は自動で曲線をつなぎます。")

controls = st.columns(3)
show_trail = controls[0].checkbox("確認画像に軌道を表示", value=True)
if controls[1].button("このコマを除外", disabled=current.source == "excluded", width="stretch"):
    edit(job, index, None)
    st.rerun()
if controls[2].button("このコマの修正を戻す", disabled=index not in job["corrections"], width="stretch"):
    del job["corrections"][index]
    invalidate_export(job)
    st.rerun()

st.subheader("軌道の色と描画範囲")
if split is None:
    st.caption("切り替え点を指定すると、そこまでが青、その後が赤になります。")
else:
    st.caption(f"青 → 赤　{split + 1:,}コマ目（{split / info.fps:.2f}秒）"
               f"{' ・ 自動候補' if job.get('split') is None else ''}")
end_label = ""
if job.get("end") is None:
    end_label = " ・ 自動候補" if suggestion.occlusion_found else " ・ 最後の検出点"
st.caption(f"描画終了　{end + 1:,}コマ目（{end / info.fps:.2f}秒）{end_label}")
timing = st.columns(3)
if timing[0].button("このコマで青 → 赤", disabled=index >= end, width="stretch"):
    job["split"] = index
    invalidate_export(job)
    st.rerun()
if timing[1].button("このコマで描画終了", disabled=current.point is None, width="stretch"):
    job["end"] = index
    invalidate_export(job)
    st.rerun()
if timing[2].button("色と終了を自動に戻す",
                    disabled=job.get("split") is None and job.get("end") is None, width="stretch"):
    job["split"] = None
    job["end"] = None
    invalidate_export(job)
    st.rerun()
seek_controls = st.columns(2)
if split is not None:
    seek_controls[0].button("色の切り替えを確認", on_click=move_to,
                            args=(frame_key, split + 1), width="stretch")
seek_controls[1].button("終了点を確認", on_click=move_to,
                        args=(frame_key, end + 1), width="stretch")

try:
    jpeg = preview_jpeg(job["source"], render_path, index, show_trail,
                        split=split, end=end, review_sample=review_sample)
except RuntimeError as error:
    st.error(str(error))
    st.stop()

token = f"{job['id']}:{index}"
event = click_frame(
    image=b64encode(jpeg).decode("ascii"), token=token,
    aspect=info.width / info.height, key=f"click_{job['id']}", default=None,
)
if event and event.get("token") == token and event.get("id") != job["click"]:
    job["click"] = event["id"]
    x, y = event.get("x"), event.get("y")
    if isinstance(x, (int, float)) and isinstance(y, (int, float)) and 0 <= x <= 1 and 0 <= y <= 1:
        edit(job, index, (round(x * (info.width - 1)), round(y * (info.height - 1))))
        st.rerun()

st.divider()
st.subheader("軌道付き動画")
st.caption("描画範囲内の未検出区間は前後の動きから推定します。描画終了後も軌道を表示したまま再生します。")
if missing:
    st.caption(f"描画範囲内に補間できなかったコマが {missing:,} 枚あります。")
can_export = any(p.point is not None for p in render_path[:end + 1])
requested = st.button("軌道付きMP4を作成", type="primary", disabled=not can_export)
auto_requested = job.pop("auto_export_pending", False)
if can_export and (auto_requested or requested):
    target = job["source"].with_name("swingtrace.mp4")
    progress = st.progress(0, text="軌道付き動画を書き出しています…")
    try:
        export_video(
            job["source"], info, render_path, target,
            lambda done, total: progress.progress(done / total, text=f"MP4作成中 · {done:,} / {total:,} コマ"),
            split=split, end=end,
        )
        job["export"] = target
    except Exception as error:
        st.error(f"MP4を作成できませんでした。{error}")
    finally:
        progress.empty()
if job["export"] is not None:
    st.video(str(job["export"]), format="video/mp4")
    with job["export"].open("rb") as finished:
        st.download_button("軌道付きMP4を保存", finished, file_name=job["filename"],
                           mime="video/mp4", on_click="ignore")
