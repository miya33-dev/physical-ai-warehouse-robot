# 自然言語指示による倉庫内ピッキング・配送ロボット

赤・青の箱を文章で指定し、画像から認識して2指グリッパで持ち上げ、障害物を避けて指定場所へ置くMuJoCoシミュレーションです。

## ライセンス
本プロジェクトの独自ソースコードは **AGPL-3.0-only** で提供します。[LICENSE](LICENSE) と [第三者ソフト・モデルの表示](THIRD_PARTY_NOTICES.md) を参照してください。元の依存ライブラリとモデルは、それぞれのライセンスを保持します。

## 処理が見える説明動画
最新の説明動画は `outputs/explanatory_demo_large/demo.mp4` です。3D映像、真上の全体画像に重ねたA*の探索・計画経路・実走行、LLMの実際のJSON応答、YOLOの検出枠または指先を同時に表示します。

```sh
./run.sh --demo --layout-seed 0 --distractors --video --video-speed 2 --explain-video --output outputs/explanatory_demo_large
```

物理動作は約2倍速です。A*の探索は実際のログを見やすい速度で再生し、その間は物理時間を停止します。探索に実際にかかった時間の表示ではありません。[表示の意味・再現方法](docs/説明動画.md) を参照してください。

## CNNによるキューブ検出
学習済みYOLO26nを合成画像で追加学習し、全RGB画像から赤・青のキューブの枠・クラス・スコアを推定します。球・円柱・薄い板を含む画像も学習・評価に使います。推論時にシミュレータの正解IDは渡しません。

```sh
./run.sh --demo --layout-seed 0 --distractors --video --video-speed 2 --output outputs/yolo_llm_demo
```

[YOLOの生成・学習・実行手順](docs/YOLO導入.md) と [モデルカード](docs/YOLOモデルカード.md) を参照。従来の色分類器は `--vision gaussian` を明示した場合だけ使います。未学習・認識失敗時の自動切り替えはありません。

## 言語理解はローカルLLMへ
- 使用モデルは **Qwen3-4B-Instruct-2507 / Q4_K_M** の1種類。
- Ollamaタグ: `qwen3:4b-instruct-2507-q4_K_M`。ダウンロード約2.5GB。
- MacではMetal経由でGPU推論。実際の配置は `./tools/ollama.sh ps` で確認。
- 日本語・英語の単発指示を、対象色・配送先のJSONへ変換。
- JSON構造と値を検証し、曖昧な指示、未対応の作業、否定・条件付き指示は停止。
- 接続・推論に失敗した場合、別モデルや従来方式へ自動で切り替えません。

画像認識の既定は追加学習済みYOLO26nです。枠の中心を既知のカメラ・高さから座標へ変換します。経路はA*、アームは逆運動学と位置サーボ、把持は有限の駆動力を持つ2本の指の接触と摩擦を使い、吸着・固定拘束は使いません。台車は左右2輪の速度サーボと床との摩擦で走行します。位置・姿勢のフィードバックはシミュレータから取得します。詳細は [車輪駆動](docs/車輪駆動.md) を参照。

## このMacで動かす
ターミナルで本フォルダを開き、まずローカル推論サーバーを起動します。すでに起動中なら再起動は不要です。

```sh
./tools/start_llm.sh
```

別のターミナルで実行します。

```sh
# まず文章の解釈のみ確認
./run.sh --parse-only --command '赤い荷物を発送エリアまで持っていって'

# LLMを使う2件の配送デモ
./run.sh --demo --distractors --video --video-speed 2 --output outputs/yolo_llm_demo

# 単発の配送
./run.sh --command '青の荷物を倉庫にしまって' --video --output outputs/llm_single

# GPU利用とモデルの読み込み状況
./tools/ollama.sh ps
```

同じ出力先では生成物を上書きします。起動したサーバーは、そのターミナルでCtrl+Cすると停止します。モデルを再取得する場合はサーバー起動中に以下を実行します。

```sh
./tools/ollama.sh pull qwen3:4b-instruct-2507-q4_K_M
```

このMacでは `.runtime/ollama` にOllama公式実行ファイル、`models/` に重み、`.packages/` にPython依存を置いています。いずれもGit管理外です。推論は `127.0.0.1:11435` で動きます。クラウド推論は無効です。Ollamaの標準設定はホームの `.ollama` に作成されます。

## 荷物の配置を変える
```sh
# 同じ番号で同じ配置を再現。別の番号にすると配置が変わる
./run.sh --demo --layout-seed 0 --video --video-speed 2 --output outputs/layout_seed_0
# JSONで座標を指定
./run.sh --demo --layout-file configs/layout_example.json --video --video-speed 2 --output outputs/custom_layout
# 認識器を固定した36ケースの検証
./run.sh --layout-evaluate
```

台の上の対応範囲と実測結果は [配置変更](docs/配置変更.md) を参照。初期位置を変えたLLM版動画は `outputs/random_layout_demo/demo.mp4` に保存しています。配置オプションを付けなければ従来の固定配置です。

## 従来版を明示的に使う
導入前のNaive Bayes分類器も残しています。

```sh
./run.sh --language naive_bayes --demo --video --output outputs/baseline_demo
```

導入前のGitコミットは `494db94`、ブランチは `codex/warehouse-baseline`。LLM導入は `codex/llm-integration`、配置変更は `codex/randomized-layouts`、車輪駆動は `codex/wheel-drive`、指把持は `codex/finger-gripper`、検出器導入は `codex/yolo-cube-detection` です。

## 指でつかむ
指把持導入時の動画は `outputs/finger_llm_demo/demo.mp4`（約2分19秒・2倍速を明記）。指先の拡大表示付きです。通常・指開放・低摩擦の比較は `./run.sh --gripper-evaluate` で実行できます。仕組みと制約は [指把持](docs/指把持.md) を参照。

## 車輪での走行
車輪導入時の動画は `outputs/wheel_llm_demo/demo.mp4`（約1分46秒・2倍速を明記）。車輪の前進・旋回・摩擦の検証は `./run.sh --drive-evaluate` で実行できます。駆動方式と限界は [車輪駆動](docs/車輪駆動.md) を参照。

## 検出器の評価
```sh
./run.sh --detector-evaluate      # 最終テスト画像の検出評価
./run.sh --yolo-robot-evaluate    # YOLOから把持・配送まで
```

## 従来方式・制御の評価
```sh
./run.sh --test          # 単体テスト（LLM APIはモック）
./run.sh --llm-evaluate  # 実際のLLMで24件の指示を評価
./run.sh --evaluate      # 従来の分類器を使う6ケースの物理検証
```

LLM評価は言い換え、日英の指示、情報不足、未知の対象・場所、否定、複数作業、条件付き、質問、規則変更要求を含みます。モデルを複数比較する評価ではありません。実行結果は `outputs/llm_evaluation/results.json` に保存します。固定した少数ケースの結果を、未知の指示に対する安全保証とは扱いません。

## 別環境での再現
検出器の重みとデータはソースに含めず、[YOLO導入手順](docs/YOLO導入.md) から生成・追加学習します。学習の出発点の重み、学習用ソース、条件、モデルのハッシュはモデルカードに記録します。
Python 3.12と、公式のOllamaを用意します。

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt -r requirements-vision.txt
./tools/start_llm.sh
# 別ターミナルで同じ仮想環境を有効化してから
./tools/ollama.sh pull qwen3:4b-instruct-2507-q4_K_M
python run.py --demo --video --output outputs/llm_demo
```

ローカル依存フォルダを別のマシンへコピーせず、各環境で依存を導入してください。LinuxではMuJoCo描画用OpenGL/EGL等の設定が別途必要な場合があります。

## ファイル
- `assets/warehouse.xml`: 独自の倉庫・台車・SCARAアーム。
- `src/warehouse_robot/llm_language.py`: LLMプロンプト・JSONスキーマ・出力検証。
- `src/warehouse_robot/vision.py`: 学習した画素分類器と座標変換。
- `src/warehouse_robot/planning.py`: 既知地図のA*。
- `src/warehouse_robot/simulation.py`: 物理・制御・動画出力。
- `tools/evaluate_language.py`: 1モデルの指示理解評価。
- `docs/LLM導入.md`: 選定理由と確認結果。

最新の説明資料は [提出用PDF（5ページ）](submission/説明資料_PAI最終課題_99691.pdf) です。LLM・YOLO・車輪・指把持・最新動画と評価結果を反映しています。生成用ソースは `tools/make_report.py`。この生成には最終動画と個別評価ログが必要です。以前のドラフトと `outputs/demo/` は導入前の版です。
講義資料、モデル重み、生成動画、認証情報はGit管理から除外します。ソースは https://github.com/miya33-dev/physical-ai-warehouse-robot で公開しています。YouTubeアップロードとOmnicampusへの課題提出は未実施です。

参考: https://ollama.com/library/qwen3:4b-instruct-2507-q4_K_M
GPU: https://docs.ollama.com/gpu
構造化出力: https://docs.ollama.com/capabilities/structured-outputs
