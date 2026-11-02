"""Small supervised character-ngram intent classifier, trained locally.
Not an LLM. Explicit safety rules reject unsupported or ambiguous requests.
"""
import numpy as np

COLORS={'red':['赤','レッド','red'], 'blue':['青','ブルー','blue']}
DESTS={'shipping':['出荷','発送','shipping'], 'storage':['保管','倉庫','storage']}

def tokens(s):
    s=s.lower().replace(' ','')
    return list(s)+[s[i:i+2] for i in range(len(s)-1)]

class IntentModel:
    def __init__(self):
        samples=[]
        for c,cs in COLORS.items():
            for dest,ds in DESTS.items():
                for a in cs:
                    for b in ds:
                        for fmt in ['{a}い箱を{b}場所へ運んで','{b}に{a}の荷物を届けて','{a}の箱を{b}へ配送して','take {a} box to {b}']:
                            samples.append((fmt.format(a=a,b=b),c+':'+dest))
        self.labels=sorted({y for _,y in samples})
        self.vocab={t:i for i,t in enumerate(sorted({t for x,_ in samples for t in tokens(x)}))}
        counts=np.ones((len(self.labels),len(self.vocab)))
        for text,label in samples:
            for t in tokens(text): counts[self.labels.index(label),self.vocab[t]]+=1
        self.logprob=np.log(counts/counts.sum(1,keepdims=True))
        self.n_train=len(samples)

    def predict(self,text):
        text=text.lower()
        if any(s in text for s in ['緑','黄色','黒','白','green','yellow','捨て','壊して','運ばない','運ばなく','運ぶな','しないで','やめ','禁止','以外','ではなく','じゃなく','以外','以外','not ','don’t',"don't"]):
            raise ValueError('unsupported color, negation, or action')
        cs=[k for k,vs in COLORS.items() if any(v in text for v in vs)]
        ds=[k for k,vs in DESTS.items() if any(v in text for v in vs)]
        if len(cs)!=1 or len(ds)!=1:
            raise ValueError('specify exactly one supported color and destination')
        if not any(v in text for v in ['運','届','配送','持って','移動','take','bring','deliver']):
            raise ValueError('explicit delivery action required')
        v=np.zeros(len(self.vocab))
        for t in tokens(text):
            if t in self.vocab: v[self.vocab[t]]+=1
        scores=self.logprob@v
        prob=np.exp(scores-scores.max()); prob/=prob.sum()
        label=self.labels[int(prob.argmax())]
        color,dest=label.split(':')
        if (color,dest)!=(cs[0],ds[0]): raise ValueError('model and safety validation disagree')
        return {'color':color,'destination':dest,'score':float(prob.max())}
