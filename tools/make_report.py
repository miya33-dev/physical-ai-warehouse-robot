from pathlib import Path
import os
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.utils import ImageReader
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'submission/説明資料_PAI最終課題_ドラフト.pdf'
pdfmetrics.registerFont(TTFont('HeiseiKakuGo-W5',os.environ.get('PAI_REPORT_FONT','/System/Library/Fonts/Supplemental/Arial Unicode.ttf')))
FONT='HeiseiKakuGo-W5'; W,H=595.28,841.89
c=canvas.Canvas(str(OUT),pagesize=(W,H)); c.setTitle('自然言語指示による倉庫内ピッキング・配送ロボット')
c.setAuthor(''); c.setSubject('PAI最終課題 説明資料ドラフト')

def text(s,x,y,size=10,color='#263547'):
    c.setFillColor(HexColor(color)); c.setFont(FONT,size);c.drawString(x,y,s)

def para(s,y,size=10,width=499,leading=17):
    line=''
    for ch in s:
        if ch=='\n' or pdfmetrics.stringWidth(line+ch,FONT,size)>width:
            text(line,48,y,size);y-=leading;line='' if ch=='\n' else ch
        else:line+=ch
    if line:text(line,48,y,size);y-=leading
    return y

def heading(s,y):
    c.setFillColor(HexColor('#16a094')); c.rect(35,y-3,4,15,fill=1,stroke=0)
    text(s,48,y,13,'#153848');return y-25

def footer(page):
    c.setStrokeColor(HexColor('#d9e1e8'));c.line(48,42,W-48,42)
    text('PAI最終課題 / 説明資料ドラフト / 2026-11-02',48,27,8,'#708294')
    text(str(page),W-58,27,8,'#708294')

text('自然言語指示による倉庫内',48,790,21,'#153848')
text('ピッキング・配送ロボット',48,760,21,'#153848')
text('指示理解 × 画像認識 × アーム操作 × 障害物回避',48,730,11,'#16877e')
y=heading('1. 概要と目的',692)
y=para('「赤い箱を出荷場所へ運んで」という文章から対象と配送先を判断し、カメラで箱を選び、吸着アームで持ち上げて障害物を迂回し、指定の台へ配置する。認識と行動を一連の作業に統合することを目的とした。',y)
y=para('MuJoCo 3.3.7上に、移動台車、SCARAアーム、赤と青の箱、2か所の配送台、固定障害物を独自に構成した。実機を使用せず、ローカル環境で実行する。',y-3)
image=ROOT/'outputs/demo/preview.png'
c.drawImage(ImageReader(str(image)),48,289,width=499,height=285,preserveAspectRatio=True,mask='auto')
text('図1  動作デモ画面（右側は指示解析時のカメラ画像と認識位置）',48,267,8,'#708294')
y=heading('2. 処理の流れと工夫',237)
y=para('文章入力 → 指示分類 → RGB画像認識 → 荷物へ接近 → 吸着・持ち上げ → A*経路で移動 → 配置・成否判定。各段階を状態として表示し、失敗理由をログに残す。',y)
y=para('視覚推論には物体IDを渡さず、画像から位置を求める。色を交換する検証で、固定の箱IDではなく外観に従って選択することを確認した。荷物が見つからない場合は、移動前に停止する。',y-4)
y=para('動画：149.6秒 / MP4 / 960×640 / 15fps。赤→出荷、青→保管の2件を連続実行。シミュレーション時間にほぼ対応する速度（約1倍）。',y-4,9)
footer(1);c.showPage()
y=heading('3. AIと制御の担当範囲',790)
y=para('指示理解：生成した144例で文字1-gram/2-gramの多項Naive Bayes分類器を学習。赤/青×出荷/保管の4クラスを推論し、許可語彙・動作語の検証と照合する。LLMではなく、対応表現を限定した教師ありモデル。',y)
y=para('画像認識：12条件の合成RGB画像と教師マスクから18,096画素を収集。RGBと色差の特徴で背景・赤・青のガウス分類器を学習する。推論時はRGBのみを使用。既知カメラと箱上面の高さから座標へ変換する。',y-4)
y=para('制御：既知地図を膨張処理したA*で経路を生成。アームは解析的な逆運動学と位置サーボで制御。移動・把持の強化学習は行っていない。',y-4)
y=heading('4. 実行結果',y-15)
rows=[('検証内容','結果'),('赤/青 × 出荷/保管の4通り','4件とも配送成功'),('学習後に箱の色を交換','別の箱を選び、配送成功'),('対象の赤い箱を除去','移動前に停止'),('配送時の配置誤差','約2.5 cm'),('障害物との接触','0（下記の判定条件による）')]
for i,(a,b) in enumerate(rows):
    c.setFillColor(HexColor('#e7f3f1' if i==0 else ('#f2f5f7' if i%2 else '#ffffff')))
    c.rect(48,y-20,499,26,fill=1,stroke=0)
    text(a,57,y-11,9);text(b,309,y-11,9);y-=26
y=para('固定シーンで各ケースを1回ずつ確認した限定的な結果。接触は台車・アーム・箱と中央障害物の1mm超の貫入を計測。標準色の別描画1画像で画素分類精度は背景100%、赤・青99.44%。未知環境や実世界の性能を示す値ではない。',y-9,9,leading=15)
y=heading('5. 簡略化・制約と今後',y-14)
y=para('台車はXY位置サーボで移動し、車輪の駆動をモデル化していない。把持は接近確認後の固定拘束による吸着表現であり、指の接触把持ではない。箱の運動は重力・接触・拘束を含む物理計算で扱う。',y,9,leading=15)
y=para('既知地図・固定カメラ・既知高さを使用。同色の複数物体、未知形状、SLAM、複雑な否定指示、実機には未対応。今後は配置を変えた評価、言語理解の拡張、車輪・把持の詳細化を進める。',y-3,9,leading=15)
y=heading('6. 再現方法と参考',y-14)
y=para('READMEの手順で依存関係を導入し、run.py --demo --video を実行する。evaluate.pyで6ケースを再検証できる。実装・資料作成に生成AIを補助利用。実行時はローカル分類器を使用する。',y,9,leading=15)
y=para('参考：MuJoCo公式ドキュメント（Python / Simulation）\nhttps://mujoco.readthedocs.io/en/stable/python.html\nhttps://mujoco.readthedocs.io/en/stable/programming/simulation.html',y-3,8,leading=13)
if y<57:raise RuntimeError(f'content overflow {y}')
footer(2);c.save()
print(OUT,'last y',y)
