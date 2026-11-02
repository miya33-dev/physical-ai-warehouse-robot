# YOLO26nによるキューブ検出

## 現在の構成
Ultralytics YOLO26nの学習済み重みを、red_cube / blue_cubeの2クラスへ追加学習します。全体のRGB画像を入力し、物体ごとにクラス、バウンディングボックス、検出スコアを出します。従来の画素ごとのGaussian色分類器は --vision gaussian で明示指定できます。YOLOの失敗時に自動で切り替える処理はありません。

既知の荷物置き場にある、指定色のキューブが1個だけ検出された場合に実行します。対象が0個、同色の候補が複数、スコアがしきい値未満の場合は停止します。スコアはモデルの内部値で、実際の正答確率を保証するものではありません。

枠の中心を固定カメラの既知パラメータと既知の箱上面高さ0.68mから世界座標へ変換し、既存の車輪・指グリッパへ接続します。枠の中心が任意の物体の把持点になるとは主張しません。箱の寸法・向き・高さは現在の設定に限定しています。

## 学習データ
画像、正解枠、クラスはプロジェクト独自のMuJoCoシーンから生成します。講義資料・外部写真・受講者専用資料は使いません。MuJoCoのセグメンテーションIDは教師ラベル作成と評価のみに使い、推論入力には渡しません。

- 各版で512枚の学習画像、96枚の検証画像、96枚のテスト画像。
- 0〜複数個の赤/青キューブと、同色の球・円柱・薄い板を配置。
- 箱のサイズ、位置、明るさ、台・床の色、ロボットの位置を変化。
- v1は主に格子配置。v2は連続配置と実行時に近い見え方を混ぜ、配置の偏りを改善。
- v1とv2、学習/検証/テストの乱数範囲を分離。

最初のモデルはテスト画像では良好でも、ロボットの連続配置で一部の赤い箱を見落としました。v2はこの分布差を改善するために作成しています。既に確認した配置は開発用として扱い、最終のロボット評価には新しい番号200〜203を使用します。

## 再現手順
Python 3.12を想定。別環境では仮想環境等で以下の依存を導入してください。

```sh
python -m pip install -r requirements.txt -r requirements-vision.txt
```

このMacでは独立した .vision-packages を使います。run.pyと学習・評価スクリプトが読み込みます。重み、データ、追加ライブラリはGit管理外です。

元の重みを公式配布元から取得します。

```sh
curl -fL https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt -o models/yolo26n.pt
./run.sh --detector-generate --variant v1
./run.sh --detector-train --dataset cubes_v1 --name cubes_yolo26n --epochs 30
./run.sh --detector-generate --variant v2
./run.sh --detector-train --dataset cubes_v2 --name cubes_yolo26n_v2 --initial runs/detect/cubes_yolo26n/weights/best.pt --epochs 20
./run.sh --detector-package
./run.sh --detector-evaluate
./run.sh --yolo-robot-evaluate
```

学習はMacのMPS GPUを指定します。学習中に外部の実験管理サービスへ同期しません。MPSが利用できなければ学習は停止します。モデルの推論でCPUを使う場合は --vision-device cpu を明示してください。

乱数とバージョンを固定していますが、MPSには完全に決定的ではない演算があるため、別環境で重みや数値がビット単位で一致する保証はありません。学習済みの配布用重みは models/cubes_yolo26n.pt、元の学習記録は runs/detect/ に保存します。学習記録にはローカルパスが含まれるため、そのまま公開せず、package処理で整えた推論用重みと必要な再現情報を使用してください。

## 実行と動画
LLMサーバーを起動してから:

```sh
./run.sh --demo --layout-seed 0 --distractors --video --video-speed 2 --output outputs/yolo_llm_demo
```

--distractors は指の接近経路と重ならない台の空き位置に、同色の別形状を静的に配置します。空きに応じて配置数が変わります。これらは今回運搬する対象ではありません。

動画では検出枠・クラス・スコア・推定座標を表示し、読みやすさのため認識画像の台周辺を拡大します。推論入力は拡大前の全RGB画像です。把持・配置中は指先のライブ拡大へ切り替わります。認識画像は対象を選んだ時点の静止画像で、走行中に再推論し続ける構成ではありません。

従来方式の再現:

```sh
./run.sh --vision gaussian --language naive_bayes --demo --layout-seed 0 --video --video-speed 2 --output outputs/gaussian_baseline
```

## 評価と公開
最終モデルの実測値は YOLOモデルカード.md と configs/detector_model.json に記録します。少数の合成場面での結果を、実世界や任意の未知物体での精度と説明しません。

ソースのライセンスはAGPL-3.0-onlyです。第三者コード・モデルの条件は THIRD_PARTY_NOTICES.md を参照。モデルを公開する場合は、対応するソース・設定・生成/学習/評価手順・ライセンス・モデルカードを一緒に案内します。GitHub等への公開・アップロードはまだ行っていません。
