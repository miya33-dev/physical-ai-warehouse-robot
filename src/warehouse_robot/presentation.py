# SPDX-License-Identifier: AGPL-3.0-only
"""Full-HD explanatory frames using real LLM outputs and recorded A* events."""
import json,math
from functools import lru_cache
from PIL import Image,ImageDraw,ImageFont,ImageOps
import numpy as np

@lru_cache(maxsize=24)
def font(size):
    try:return ImageFont.truetype('/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc',size)
    except OSError:return ImageFont.load_default(size=size)

def pixel(x,y):
    scale=2*7*math.tan(math.radians(45)/2)/480
    return (320+x/scale,240-y/scale)

def wrapped(text,width,size):
    lines=[]
    for original in text.splitlines():
        line=''
        for ch in original:
            if font(size).getlength(line+ch)>width:
                lines.append(line);line=ch
            else:line+=ch
        lines.append(line)
    return lines

def block(draw,text,xy,width,height,size=22,color='#dce7f3'):
    while size>14 and len(wrapped(text,width,size))*(size+3)>height:size-=1
    for i,line in enumerate(wrapped(text,width,size)):
        draw.text((xy[0],xy[1]+i*(size+3)),line,font=font(size),fill=color)

def search_overlay(s):
    trace=s.current_plan
    count=min(s.search_progress,len(trace['events'])) if trace else 0
    key=(id(trace),count)
    if getattr(s,'_overlay_key',None)==key:return s._overlay_image
    layer=Image.new('RGBA',(640,480),(0,0,0,0));d=ImageDraw.Draw(layer)
    # Known central-obstacle map, the same one used by planning.py.
    if trace:
        a,b,c,e=trace['obstacle'];margin=trace['radius']
    else:a,b,c,e,margin=-.3,.3,-.68,.68,.53
    d.rectangle((*pixel(a-margin,e+margin),*pixel(b+margin,c-margin)),fill=(255,168,68,45),outline=(255,177,75,200),width=1)
    if trace:
        res=trace['resolution'];closed=set();frontier={tuple(round(v/res) for v in trace['start'])}
        for event in trace['events'][:count]:
            node=tuple(event['current']);frontier.discard(node);closed.add(node)
            for item in event['opened']:frontier.add(tuple(item))
        half=.5*res/(2*7*math.tan(math.radians(45)/2)/480)
        for nodes,color in [(closed,(74,176,255,65)),(frontier-closed,(196,134,255,145))]:
            for x,y in nodes:
                px,py=pixel(x*res,y*res);d.rectangle((px-half,py-half,px+half,py+half),fill=color)
        if count==len(trace['events']) and len(trace['path'])>1:
            d.line([pixel(*p) for p in trace['path']],fill=(255,226,90,255),width=3)
        for point,label,color in [(trace['start'],'S',(100,220,255,255)),(trace['goal'],'G',(255,230,110,255))]:
            x,y=pixel(*point);d.ellipse((x-6,y-6,x+6,y+6),fill=color,outline='white',width=1)
            d.text((x+8,y-12),label,font=font(14),fill='white',stroke_width=1,stroke_fill='black')
    s._overlay_key=key;s._overlay_image=layer
    return layer

def overhead(s):
    image=Image.fromarray(s.rgb()).convert('RGBA')
    image=Image.alpha_composite(image,search_overlay(s));d=ImageDraw.Draw(image)
    if len(s.current_actual)>1:d.line([pixel(*p) for p in s.current_actual],fill='#58f09d',width=2)
    x,y=s.base_xy();px,py=pixel(x,y);yaw=s.yaw()
    d.ellipse((px-5,py-5,px+5,py+5),fill='#58f09d',outline='black',width=1)
    ex,ey=pixel(x+.20*math.cos(yaw),y+.20*math.sin(yaw));d.line((px,py,ex,ey),fill='white',width=3)
    return image.convert('RGB')

def detail(s):
    if s.stage in ['PICK','PLACE']:return Image.fromarray(s.gripper_rgb()),'指先：接触と摩擦'
    if s.vision_frame is None:return Image.new('RGB',(208,156),'#0b1320'),'カメラ画像'
    image=Image.fromarray(s.vision_frame.copy());d=ImageDraw.Draw(image)
    if s.detection:
        for item in s.detection.get('detections',[]):
            color='#ff9292' if item['class']=='red_cube' else '#77bbff'
            d.rectangle(item['bbox_xyxy'],outline=color,width=2)
            x,y=item['bbox_xyxy'][:2];d.text((x,max(0,y-13)),item['class']+f" {item['score']:.2f}",fill=color)
    return image.crop((95,145,260,335)),'YOLO：予測枠の拡大'

def render_explanation(s):
    s.r.disable_segmentation_rendering();s.r.update_scene(s.d,camera=s.cam)
    scene=Image.fromarray(s.r.render().copy());top=overhead(s)
    canvas=Image.new('RGB',(1920,1080),'#0b1320');d=ImageDraw.Draw(canvas)
    d.text((28,16),'AI WAREHOUSE  |  LLM → YOLO → A* → 車輪 → 指把持',font=font(32),fill='#72e7d4')
    d.text((28,64),s.command or '配送の準備中',font=font(27),fill='white')
    stages={'INITIALIZE':'準備','PARSE':'指示を解釈','PERCEIVE':'画像から対象を検出','APPROACH':'荷物へ接近','PICK':'指でつかむ','TRANSPORT':'配送','PLACE':'配置','DONE':'完了','ERROR':'停止'}
    d.text((1440,24),'状態：'+stages.get(s.stage,s.stage),font=font(24),fill='#72e7d4')
    d.text((1440,62),f'物理時間 {s.d.time:6.1f}s  /  接触検出 {s.collisions}',font=font(22),fill='#a6b8cc')
    for box in [(24,110,936,660),(984,110,1896,660),(24,714,960,1056),(984,714,1896,1056)]:
        d.rounded_rectangle(box,radius=10,fill='#111e2d',outline='#314458',width=2)
    d.text((44,120),'3D物理シミュレーション',font=font(27),fill='white')
    d.text((1004,120),'真上の全体画像 ＋ A*探索',font=font(27),fill='white')
    d.text((1004,153),'中央障害物を0.53m拡張 / 10cm格子・4近傍 / f = g + h',font=font(16),fill='#a6b8cc')
    canvas.paste(scene.resize((624,468)),(168,184));canvas.paste(top.resize((624,468)),(1128,184))
    trace=s.current_plan
    if trace:
        n=min(s.search_progress,len(trace['events']));event=trace['events'][n-1] if n else None
        label='探索ログ再生' if s.search_animating else '探索完了'
        phase='接近' if trace.get('phase')=='APPROACH' else '配送'
        if trace.get('command')!=s.command:label='前回の経路（次の探索待ち）'
        d.rectangle((1100,178,1815,216),fill='#0b1320')
        text=f'経路 #{len(s.plan_records)}：{phase} | {label} {n}/{len(trace["events"])}'
        d.text((1110,182),text,font=font(20),fill='white')
        if event:
            g=event['g'];h=event['h'];d.text((1500,122),f'g {g} + h {h} = f {g+h}',font=font(21),fill='#ffe26a')
    else:d.text((1100,182),'台車の移動開始時に探索します',font=font(20),fill='white')
    d.text((32,675),'姿勢・接触・把持はMuJoCoの物理計算',font=font(21),fill='#a6b8cc')
    legend=[('#60baff','探索済み'),('#c68aff','探索候補'),('#ffb14b','安全余裕'),('#ffe25a','計画経路'),('#58f09d','実走行')]
    for i,(color,label) in enumerate(legend):
        x=990+181*i;d.rectangle((x,677,x+16,693),fill=color);d.text((x+23,671),label,font=font(19),fill='#dce7f3')
    d.text((44,726),'LLM出力：実際に返ったJSON',font=font(28),fill='#72e7d4')
    d.text((44,764),'Qwen3 4B / ローカルGPU',font=font(22),fill='#a6b8cc')
    output=json.dumps(s.llm_output,ensure_ascii=False,indent=2) if s.llm_output is not None else (('LLM応答を取得できませんでした' if s.stage=='ERROR' else 'LLMの応答待ち') if getattr(s.lang,'name','')=='local_llm' else '従来分類器を使用（LLMなし）')
    block(d,output,(44,804),896,240,28)
    image,label=detail(s);image=ImageOps.contain(image,(400,288))
    d.text((1004,726),label,font=font(28),fill='#72e7d4');canvas.paste(image,(1004+(400-image.width)//2,758))
    d.text((1424,786),'認識：'+getattr(getattr(s,'vision',None),'name','Gaussian'),font=font(23),fill='white')
    if s.detection:
        x,y=s.detection['xy'];d.text((1424,844),f'X {x:.2f}m / Y {y:.2f}m',font=font(22),fill='white')
        if 'score' in s.detection:d.text((1424,902),f'検出スコア {s.detection["score"]:.2f}',font=font(22),fill='white')
    if s.intent:d.text((1424,976),s.intent['color']+' → '+s.intent['destination'],font=font(22),fill='#72e7d4')
    footer=f'物理動作：約{s.video_speed}倍速 | A*は実際の探索ログを低速再生（再生中は物理時間停止） | 既知地図・シミュレータ自己位置'
    d.text((28,1058),footer,font=font(14),fill='#9db1c5')
    return canvas
