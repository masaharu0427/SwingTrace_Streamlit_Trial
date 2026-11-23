# SwingTrace クラブヘッド追跡 試験版

学習済みの SwingTrace（YOLO11m）を使い、1本の動画からクラブヘッドの2D軌道を作る Streamlit アプリです。動画を選んで「自動追跡する」を押すだけで解析が始まります。P1〜P10の指定や最初の位置合わせは不要です。

## GitHub と Streamlit Community Cloud

1. ZIPを展開し、専用のGitHubリポジトリに中身をアップロードします。リポジトリ直下に `app.py`、`tracking.py`、`video_io.py`、`requirements.txt`、`packages.txt`、`click_frame/index.html` が来るようにフォルダー構成を維持してください。表示される場合は `.streamlit/config.toml` と `.gitignore` も入れてください。
2. [Streamlit Community Cloud](https://share.streamlit.io/) でアプリを作成し、そのリポジトリとブランチを選び、Main file path に `app.py` を指定します。
3. Advanced settings で **Python 3.12** を選んで Deploy します。Secrets や API キーの登録は不要です。
4. 初回の自動追跡時に約40 MBの学習済み重みを公開GitHubから取得します。実行プロセスが保持されている間は、読み込んだモデルを再利用します。

## 使い方

1. MP4・MOV・M4Vを選択して「自動追跡する」を押します。屋内外ともカメラを固定し、クラブ全体が入るように撮影した短い動画でお試しください。
2. 「前の要確認」「次の要確認」で低信頼の検出、推定、補間、未検出のコマに移動できます。最初に近い候補を開くには、コマ1で「次の要確認」を押してください。コマ1自体が候補の場合は状態欄に表示されます。スライダーですべてのコマに移動できます。
3. ずれたコマの画像上で、実際のクラブヘッドの中心をクリックします。水色の印が修正後の位置です。前後の短い欠損は修正点も使って再計算されます。ヘッドが画面外にあるコマは「このコマを除外」、やり直すときは「このコマの修正を戻す」を使います。
4. 「軌道付きMP4を作成」を押し、プレビューを確認して「軌道付きMP4を保存」を押します。作成後に修正した場合は、MP4を作り直してください。編集中の状態は開いているセッション内で保持されます。

## Windows 11で起動する場合

Python 3.12（64ビット）を入れ、展開したフォルダーで PowerShell を開きます。

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

ブラウザーで `http://localhost:8501` が開きます。WindowsでもFFmpegはPython依存関係から利用します。初回の重みの自動取得に失敗した場合は、[公開重み best.pt](https://raw.githubusercontent.com/nishsm/golf-club-detection-swingtrace/609af10e19bae6741e016c41caa966c6e6124faf/weights/best.pt) をダウンロードし、`app.py` と同じ階層に `weights` フォルダーを作って `weights/best.pt` として配置してください。

## 動画と軌道について

- 動画は100 MB、30秒、1,800コマまでです。240 fpsなら約7.5秒分です。CPUでコマごとに推論するため、最初は短いクリップでお試しください。高速モードはAIの入力画像を小さくします。
- 動画は回転情報を反映して読み込み、長辺が1,920 pxを超える場合は縮小します。可変フレームレートの動画は固定フレームレートに変換して、確認画面と保存動画のコマ番号をそろえます。
- ヘッドの直接検出を優先し、短い欠損はシャフト検出や前後の位置で補います。位置を埋められない区間は線が途切れます。要確認の印がないコマも、画像で位置を確認して修正できます。
- 保存形式はH.264のMP4です。元動画の最初の音声トラックがあればAACで保持します。クラウド版では推論と書き出しをアプリのサーバー内で実行します。ローカル版ではPC内で処理します。

## 出典とライセンス

- 学習済みモデルは Nishanth Madthila の [SwingTrace](https://github.com/nishsm/golf-club-detection-swingtrace/tree/609af10e19bae6741e016c41caa966c6e6124faf) の公開重みです。この試験版は元の重みを使い、軌道の補間・修正画面・MP4出力を実装しています。
- 元のモデルは [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics) を [golf-club-tracking データセット](https://universe.roboflow.com/club-head-tracking/golf-club-tracking)（CC BY 4.0）で追加学習したものです。
- 本試験版とモデルのライセンスは AGPL-3.0 です。ライセンス本文は `LICENSE` を参照してください。
