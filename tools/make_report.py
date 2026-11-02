# SPDX-License-Identifier: AGPL-3.0-only
"""Build the final Japanese report from recorded demo/evaluation artifacts."""
import argparse,json,os
from pathlib import Path
from io import BytesIO
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.utils import ImageReader
from PIL import Image
import imageio.v2 as iio

ROOT=Path(__file__).resolve().parents[1]
a=argparse.ArgumentParser();a.add_argument('--account-id',default='99691');a.add_argument('--output');args=a.parse_args()
if not args.account_id.isalnum():raise ValueError('account ID must be alphanumeric')
OUT=Path(args.output) if args.output else ROOT/'submission'/f'説明資料_PAI最終課題_{args.account_id}.pdf'
OUT.parent.mkdir(parents=True,exist_ok=True)
DEMO=ROOT/'outputs/explanatory_demo_large'
r=json.loads((DEMO/'results.json').read_text());det=json.loads((ROOT/'configs/detector_model.json').read_text())
llm=json.loads((ROOT/'outputs/llm_evaluation/results.json').read_text())
robot=json.loads((ROOT/'outputs/yolo_robot_evaluation/summary.json').read_text())
v=iio.get_reader(DEMO/'demo.mp4');meta=v.get_meta_data()
def frame(second):return Image.fromarray(v.get_data(round(second*meta['fps'])))
hero=frame(70);vision=frame(7).crop((984,714,1896,1056));route=frame(70).crop((984,110,1896,660));grip=frame(16).crop((1004,758,1404,1046));v.close()
pdfmetrics.registerFont(TTFont('JP',os.environ.get('PAI_REPORT_FONT','/System/Library/Fonts/Supplemental/Arial Unicode.ttf')))
W,H=595.28,841.89;M=42;CW=W-2*M;C=canvas.Canvas(str(OUT),pagesize=(W,H),pageCompression=1)
C.setTitle('自然言語指示による倉庫内ピッキング・配送ロボット');C.setAuthor('');C.setSubject('PAI最終課題 説明資料');C.setCreator('PAI report generator');C.setKeywords('MuJoCo, Qwen3, YOLO26n, A*, mobile manipulator')
INK='#183144';MUTED='#5b7180';TEAL='#087f83';LIGHT='#eef5f6'
def rect(x,t,w,h,color):C.setFillColor(HexColor(color));C.rect(x,H-t-h,w,h,fill=1,stroke=0)
def text(s,x,t,size=10,color=INK):C.setFillColor(HexColor(color));C.setFont('JP',size);C.drawString(x,H-t-size,s)
def lines(s,width,size):
 out=[]
 for raw in s.split('\n'):
  row=''
  for ch in raw:
   if pdfmetrics.stringWidth(row+ch,'JP',size)>width:out.append(row);row=ch
   else:row+=ch
  out.append(row)
 return out
def para(s,t,x=M,width=CW,size=10,leading=16,color=INK):
 for row in lines(s,width,size):text(row,x,t,size,color);t+=leading
 if t>794:raise RuntimeError(f'page content overflow at {t}: {s[:35]}')
 return t

def title(n,kicker,heading):
 rect(0,0,W,7,TEAL);text('PHYSICAL AI  /  FINAL PROJECT',M,23,9,TEAL);text(kicker,M,51,10,MUTED);text(heading,M,72,21)
 C.setStrokeColor(HexColor('#cfdee3'));C.line(M,H-111,W-M,H-111)
 text(f'PAI最終課題  |  account ID {args.account_id}  |  2026-11-03',M,811,8,MUTED);text(f'{n} / 5',W-M-23,811,8,MUTED)
def heading(s,t):rect(M,t+2,3,16,TEAL);text(s,M+12,t,13);return t+27
def image(im,x,t,w,h):C.drawImage(ImageReader(im),x,H-t-h,width=w,height=h,mask='auto')
def table(rows,t,widths,header=True,size=9.2):
 for i,row in enumerate(rows):
  chunks=[lines(str(s),widths[j]-18,size) for j,s in enumerate(row)];rh=max(len(z) for z in chunks)*15+14
  rect(M,t,sum(widths),rh,'#deeff0' if i==0 and header else ('#f1f5f7' if i%2 else '#ffffff'))
  x=M
  for j,part in enumerate(chunks):
   for k,line in enumerate(part):text(line,x+9,t+7+k*15,size)
   x+=widths[j]
  t+=rh
 return t

title(1,'OVERVIEW','自然言語で指示する倉庫配送ロボット')
t=para('文章から荷物と配送先を理解し、画像で見つけ、指でつかみ、車輪で運ぶ。\nローカルAIと物理シミュレーションを一連の作業として統合した。',128,size=11,leading=18)
image(hero,M,179,CW,CW*1080/1920)
text('図1  最終デモの画面。LLMのJSON、YOLO、A*の探索と走行を同時に表示。',M,474,8.5,MUTED)
t=heading('目的とタスク',506)
t=para('倉庫の配送作業を題材に、指示理解・視覚認識・経路計画・把持をつなぐ。MuJoCo上に台車、SCARAアーム、2指グリッパ、荷物台、2つの配送先、中央障害物を独自に構成した。',t)
t=para('赤いキューブを発送エリアへ運び、続いて青いキューブを保管場所へ運ぶ。2件目は最初の配送先から荷物台へ戻り、再度認識・把持して配送する。同色の球・円柱・薄い板も置き、色が同じでも形の異なる物体を区別してキューブを選ぶ。',t+6)
t=heading('独自の工夫',t+16)
t=para('① 構造化したLLM出力を検証してから実行する。\n② 自作の合成画像でYOLOを追加学習し、キューブの枠を検出する。\n③ 接触・摩擦による指把持と車輪駆動を物理計算する。\n④ 実際の探索ログと走行軌跡を重ね、AIと制御の役割を可視化する。',t,leading=18)
C.showPage()

title(2,'AI / LANGUAGE & VISION','AIの担当：指示理解とキューブ検出')
t=heading('1. ローカルLLMで文章をJSONにする',130)
t=para('Qwen3-4B-Instruct-2507（約40億パラメータ、Q4_K_M量子化）をOllama経由で使用。Apple M5・統合メモリ32GBのMacでMetal GPU推論を行う。LLMは対象色と配送先の判断を担当し、モーターを直接操作しない。',t)
rect(M,219,CW,114,LIGHT)
text('実際の応答例：赤い荷物を発送エリアまで持っていって',M+12,228,9,TEAL)
for i,line in enumerate(json.dumps(r['records'][0]['llm_response'],ensure_ascii=False,indent=2).splitlines()):text(line,M+15,247+i*11,9)
t=para('PythonでJSONの型・許可値・文章中の色と配送先を照合。情報不足、複数候補、否定・条件付き指示、clarify / rejectでは停止して再入力を求める。接続・推論失敗時も停止し、別方式へ自動で切り替えない。',345,size=9.5,leading=15)
t=heading('2. YOLOで画像から枠・クラス・スコアを出す',t+15)
image(vision,M,t,CW,CW*342/912)
text('図2  RGB画像の予測枠を拡大表示。全体画像がモデルへの入力。',M,t+198,8.5,MUTED)
t+=225
t=para('ベースはUltralytics YOLO26n。赤・青のキューブ2クラスへ追加学習した。MuJoCoで位置・色・照明・サイズを変え、同色の別形状や対象なしを含む画像を生成。最初の画像セットで30エポック学習し、その重みを引き継いで、配置と実行時の見え方を増やした画像セットでさらに20エポック学習した。各セットは学習512枚・検証96枚・テスト96枚。実行時は、この追加学習後の最終モデル1つだけを使う。',t,size=9.5,leading=15)
t=para('RGBだけを入力し、物体IDは教師枠生成・評価に限定する。予測枠中心を、既知カメラと箱上面の高さからXYへ変換する。荷物台内の同色候補が複数、または対象なしなら停止。動画の認識枠は認識時点の静止画像で、走行中の再推論は行わない。',t+6,size=9.5,leading=15)
C.showPage()

title(3,'PLANNING & PHYSICS','制御の担当：経路計画・車輪・指把持')
image(route,M,129,CW,CW*550/912)
text('図3  既知の地図によるA*。黄色が計画経路、緑が台車の実走行。',M,447,8.5,MUTED)
t=heading('A*と車輪で障害物を迂回する',475)
t=para('10cm格子・上下左右4近傍でA*探索。中央障害物を0.53m拡張し、f=g+h（移動コスト＋マンハッタン距離）で候補を選ぶ。曲がり角を残した経路を台車が追従し、距離と向きの誤差から左右輪の目標速度を更新する。',t,size=9.5,leading=15)
t=para('車輪の速度サーボと床との接触・摩擦で前進・旋回する。車体位置を直接書き換えない。自己位置と姿勢はMuJoCoから取得する。上空映像への重ね描画は可視化であり、障害物を画像から地図化する処理ではない。',t+5,size=9.5,leading=15)
t=heading('逆運動学と接触・摩擦でつかむ',t+14)
t=para('YOLOの推定XYへアーム先端を運び、指を開いて下降し、左右から閉じる。解析的な逆運動学で関節目標を求め、位置サーボで動かす。両指の接触と持ち上げ後の高さを確認してから配送し、目的地では下げて指を開く。',t,size=9.5,leading=15)
t=para('対象は一辺0.18m・質量0.15kg。指の駆動力は各±15Nに制限。把持は接触と摩擦で支え、固定拘束は0。箱の座標を運搬中に書き換えない。接触力・高さによる把持確認と落下停止にはシミュレータ情報を使う。',t+5,size=9.5,leading=15)
C.showPage()

title(4,'VALIDATION','実行結果と評価の範囲')
t=heading('最終デモ：2件の連続配送に成功',130)
rows=[['指示','結果','箱の配置誤差'],['赤 → 発送エリア','成功',f"{r['records'][0]['placement_error_m']*100:.2f} cm"],['青 → 保管場所','成功',f"{r['records'][1]['placement_error_m']*100:.2f} cm"]]
t=table(rows,t,[235,112,CW-347])
t=para('中央障害物とロボット・荷物の1mm超の貫入、ロボットと台の2mm超の貫入の検出はともに0。配置成功はXY誤差15cm未満、箱中心高さ0.59±0.05m、上記の接触検出0で判定した。',t+8,size=9,leading=14)
t=para(f"動画は{meta['duration']:.2f}秒、1920×1080、15fps。物理動作は約2倍速。A*は実際の探索ログを各約2.53秒で再生し、その間は物理時間を止める。探索計算の実測所要時間ではない。LLM待ち時間も物理時間に含めない。",t+5,size=9,leading=14)
t=heading('個別評価：条件を分けて検証',t+15)
rows=[['対象・条件','結果'],['言語：固定24指示を各1回\n言い換え、日英、曖昧・否定・条件付き等','検証込み 24 / 24\nLLM単体 23 / 24'],['YOLO：未学習のテスト96画像\nスコア0.5、IoU 0.5以上','適合率 92.64% / 再現率 94.38%\n正検出151・誤検出12・見落とし9'],['YOLO＋制御：新規4配置×2色×2配送先\n＋同色別形状を含む2件','18 / 18 配送成功\n言語はNaive Bayes分類器を使用'],['指把持：通常・指開放・低摩擦\n既知座標で機構を比較','通常のみ持ち上げ・10秒保持\n指開放・低摩擦では持ち上がらない'],['単体テスト：言語検証・経路・制御等','14 / 14 通過']]
t=table(rows,t,[300,CW-300],size=9)
t=para('LLM単体では条件付き指示を配送として解釈した1件を、独立した検証ルールが停止させた。言語評価はプロンプト調整に使った例を含む開発中の少数試験で、未知指示に対する正答率ではない。',t+9,size=9,leading=14)
t=para('18件はLLMを使わず、文字の出現傾向から色・配送先を選ぶNaive Bayes分類器で定型指示を解釈し、画像認識・制御を評価した。LLM込みの連続配送2件とは別の試験。画像評価は合成画像・固定カメラの条件で、実世界の精度を示さない。',t+5,size=9,leading=14)
image(grip,M,682,150,108)
text('図4  2本の指で箱の側面を挟む',M+170,688,10,TEAL)
para('最終デモからの拡大。固定拘束で接着せず、接触と摩擦で持ち上げる。把持の確認には接触力と箱の高さを使う。',711,x=M+170,width=CW-170,size=9,leading=15)
C.showPage()

title(5,'REPRODUCTION & LIMITATIONS','再現方法・制約・ライセンス')
t=heading('再現方法と成果物',130)
t=para('実行環境：Python 3.12、MuJoCo 3.3.7、Ultralytics 8.4.198、PyTorch 2.15.0、Ollama。MacではMPS/Metalを利用。READMEとdocs/YOLO導入.mdに従い、依存・LLM・合成データ生成・検出器学習を準備する。',t,size=9.5,leading=15)
rect(M,214,CW,94,LIGHT)
cmds=['./tools/start_llm.sh  # 別ターミナルでデモ実行','./run.sh --demo --layout-seed 0 --distractors --video','  --video-speed 2 --explain-video','  --output outputs/explanatory_demo_large']
for i,s in enumerate(cmds):text(s,M+12,225+i*18,9)
t=para('上のデモコマンドは2〜4行目を1行につなげて実行する。./run.sh --test で単体テスト。--llm-evaluate、--detector-evaluate、--yolo-robot-evaluate、--gripper-evaluate で個別評価を行う。',320,size=9,leading=14)
t=para('動画・results.json・planning.jsonには、実際のLLM応答、認識結果、探索ログ、走行、成否を保存する。生成動画、ローカル環境、モデル重み、データ、講義資料はソースGit管理外。学習・評価用ソースと条件は含める。',t+5,size=9,leading=14)
t=heading('制約と発展内容',t+14)
t=para('固定した真上カメラ、既知の箱・台の高さ、軸にそろった箱の向き、中央障害物の既知地図を前提とする。自己位置・接触力・箱高さはシミュレータから取得する。SLAM、未知障害物の認識、6D姿勢推定、実機検証は未実装。LLMの確認質問へ会話で回答して再開する機能もない。',t,size=9.5,leading=15)
t=para('今後は、深度画像による位置・姿勢推定、画像由来の占有地図、自己位置推定、回転した箱や材質・重量の変化、遮蔽を含む追加評価へ発展できる。',t+5,size=9.5,leading=15)
t=heading('ライセンスと開発支援',t+14)
t=para('独自ソースはAGPL-3.0-only。Ultralyticsのコード・ベース重み・派生モデルは上流のAGPL-3.0条件を保持する。Qwen3はApache-2.0の別モデル。上流のモデル構造・事前学習を独自成果とは扱わない。LICENSE、THIRD_PARTY_NOTICES.md、モデルカードに出典と条件を記載した。',t,size=9,leading=14)
t=para('モデル重みを配布する際は、対応するソース・設定・生成／学習／評価手順・ライセンス・モデルカードを併せて案内する。実装・資料作成には生成AI（Codex）を補助利用し、結果はローカルで実行・検証した。',t+4,size=9,leading=14)
t=heading('参考・出典',t+12)
refs=[('MuJoCo公式ドキュメント','https://mujoco.readthedocs.io/en/stable/'),('Qwen3-4B-Instruct-2507 モデルカード','https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507'),('Ollama 構造化出力','https://docs.ollama.com/capabilities/structured-outputs'),('Ultralytics YOLO26 / ライセンス','https://docs.ultralytics.com/models/yolo26/')]
for name,url in refs:
 text(name,M,t,8);text(url,M+5,t+12,7.5,TEAL);C.linkURL(url,(M,H-t-25,W-M,H-t),relative=0);t+=29
if t>796:raise RuntimeError(f'references overflow {t}')
C.save();print(OUT)
