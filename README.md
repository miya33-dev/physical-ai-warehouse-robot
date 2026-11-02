# 自然言語指示による倉庫内ピッキング・配送ロボット

赤・青の箱を、文章による指示に応じて選び、吸着アームで持ち上げ、障害物を避けて指定の台に置くMuJoCoシミュレーションです。

## 今できること
- 指示をローカル学習の文字n-gram分類器で「対象色・配送先」に変換。
- 合成画像で学習したRGB分類器でカメラ画像から箱を選択。
- SCARAアームの逆運動学、吸着拘束、既知地図のA*で配送。
- 存在しない対象や未対応・曖昧な指示は停止。
- MP4動画、カメラ画像、学習モデル、実行ログを出力。

初版では車輪の動力学を扱わずXY位置サーボで台車を移動し、指による把持の代わりに吸着を固定拘束で表現します。実機・SLAM・LLM・強化学習は使用していません。

## このMacで動かす
最終課題フォルダをターミナルで開き、以下を実行します。`.packages`に必要ライブラリを導入済みです。

```sh
./run.sh --command '赤い箱を出荷場所へ運んで' --video
./run.sh --demo --video --output outputs/demo
```

`--demo` は赤→出荷、青→保管を連続実行します。既定の単発実行では `outputs/first_run/`、デモでは上記の指定先に保存します。同じ出力先では生成物を上書きします。画面描画可能なログインセッションで実行してください。

## 別環境での再現
Python 3.12を想定します。依存関係の固定バージョンは `requirements.txt` にあります。

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run.py --demo --video --output outputs/demo
```

`.packages` はこのMac向けのローカル依存ファイルです。別環境にはコピーせず、仮想環境にインストールしてください。LinuxではMuJoCo描画用のOpenGL/EGL等の設定が別途必要になることがあります。

## 検証
通常の依存環境では以下で実行できます。

```sh
PYTHONPATH=src python -m unittest discover -s tests -v
python evaluate.py
```

このMacのローカル依存を使う場合は `./run.sh --evaluate` または `./run.sh --test` を利用できます。

2026-11-02に4通りの配送、色の入れ替え、対象なしの6ケースが通過。配置誤差は約2.5cm、障害物接触は0でした。固定した合成シーンでの確認であり、未知環境の成功率を示しません。

## 主なファイル
- `assets/warehouse.xml`: 独自の倉庫・移動台車・SCARAアームモデル。
- `src/warehouse_robot/language.py`: 指示分類器と入力検証。
- `src/warehouse_robot/vision.py`: 学習した画素分類器と固定カメラの座標変換。
- `src/warehouse_robot/planning.py`: 障害物を膨張させた4近傍A*。
- `src/warehouse_robot/simulation.py`: 物理・制御・状態遷移・動画生成。
- `outputs/demo/demo.mp4`: 約2分30秒の動作動画。
- `outputs/evaluation/summary.json`: 統合検証結果。
- `docs/実装と検証.md`: AIの構成と実測結果、制約。
- `submission/説明資料_PAI最終課題_ドラフト.pdf`: 説明資料のドラフト。

講義配布PDF、受講者専用資料、ローカル依存、出力物はコード公開用リポジトリに混ぜません。公開・アップロード・課題提出はまだ行っていません。
