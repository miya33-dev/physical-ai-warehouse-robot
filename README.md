# 自然言語指示による倉庫内ピッキング・配送ロボット

赤・青の箱を文章で指定し、画像から認識して吸着アームで持ち上げ、障害物を避けて指定場所へ置くMuJoCoシミュレーションです。

## 言語理解はローカルLLMへ
- 使用モデルは **Qwen3-4B-Instruct-2507 / Q4_K_M** の1種類。
- Ollamaタグ: `qwen3:4b-instruct-2507-q4_K_M`。ダウンロード約2.5GB。
- MacではMetal経由でGPU推論。実際の配置は `./tools/ollama.sh ps` で確認。
- 日本語・英語の単発指示を、対象色・配送先のJSONへ変換。
- JSON構造と値を検証し、曖昧な指示、未対応の作業、否定・条件付き指示は停止。
- 接続・推論に失敗した場合、別モデルや従来方式へ自動で切り替えません。

画像認識は従来どおり、合成画像で学習する小型Gaussian分類器を使います。経路はA*、アームは逆運動学と位置サーボ、吸着は固定拘束です。台車はXY位置サーボで動かし、車輪の駆動はモデル化していません。

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
./run.sh --demo --video --output outputs/llm_demo

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
./run.sh --demo --layout-seed 0 --video --output outputs/layout_seed_0
# JSONで座標を指定
./run.sh --demo --layout-file configs/layout_example.json --video --output outputs/custom_layout
# 認識器を固定した36ケースの検証
./run.sh --layout-evaluate
```

台の上の対応範囲と実測結果は [配置変更](docs/配置変更.md) を参照。初期位置を変えたLLM版動画は `outputs/random_layout_demo/demo.mp4` に保存しています。配置オプションを付けなければ従来の固定配置です。

## 従来版を明示的に使う
導入前のNaive Bayes分類器も残しています。

```sh
./run.sh --language naive_bayes --demo --video --output outputs/baseline_demo
```

導入前のGitコミットは `494db94`、ブランチは `codex/warehouse-baseline`。LLM導入は `codex/llm-integration`、配置変更は `codex/randomized-layouts` です。

## 評価
```sh
./run.sh --test          # 単体テスト（LLM APIはモック）
./run.sh --llm-evaluate  # 実際のLLMで24件の指示を評価
./run.sh --evaluate      # 従来の分類器を使う6ケースの物理検証
```

LLM評価は言い換え、日英の指示、情報不足、未知の対象・場所、否定、複数作業、条件付き、質問、規則変更要求を含みます。モデルを複数比較する評価ではありません。実行結果は `outputs/llm_evaluation/results.json` に保存します。固定した少数ケースの結果を、未知の指示に対する安全保証とは扱いません。

## 別環境での再現
Python 3.12と、公式のOllamaを用意します。

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
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

`outputs/demo/` と既存の説明資料PDFは導入前の版です。LLM版の動画は `outputs/llm_demo/` に区別します。説明資料PDFはLLM版へ未更新です。
講義資料、モデル重み、生成動画、認証情報はGit管理から除外します。公開・アップロード・課題提出はまだ行っていません。

参考: https://ollama.com/library/qwen3:4b-instruct-2507-q4_K_M
GPU: https://docs.ollama.com/gpu
構造化出力: https://docs.ollama.com/capabilities/structured-outputs
