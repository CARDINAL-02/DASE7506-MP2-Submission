"""Small, deterministic local arithmetic and question/trace coverage features."""
import ast
from fractions import Fraction
import re
import numpy as np
from features import extract
from supplement_features import restricted

NUMBER=r'[-+]?\d[\d,]*(?:\.\d+)?'
EQUATION=re.compile(r'(?<![\w.])('+NUMBER+r'(?:\s*[-+*/]\s*'+NUMBER+r')+)\s*=\s*('+NUMBER+r'(?:/\d+)?)')
STOP={'the','a','an','is','are','was','were','to','of','and','in','for','he','she','it','they','his','her','how','many','much','does','do','has','have','with','at','on','by','that','this','as','from','their'}


def safe_number(text):
    if len(text)>100:
        return None
    text=text.replace(',','').replace('×','*').replace('÷','/').replace('−','-')
    try:
        tree=ast.parse(text.strip(),mode='eval')
        if sum(1 for _ in ast.walk(tree))>40:
            return None
        def visit(node):
            if isinstance(node,ast.Constant) and type(node.value) in (int,float):
                value=Fraction(str(node.value))
            elif isinstance(node,ast.UnaryOp) and isinstance(node.op,(ast.UAdd,ast.USub)):
                value=visit(node.operand)*(1 if isinstance(node.op,ast.UAdd) else -1)
            elif isinstance(node,ast.BinOp) and isinstance(node.op,(ast.Add,ast.Sub,ast.Mult,ast.Div)):
                left,right=visit(node.left),visit(node.right)
                if isinstance(node.op,ast.Add): value=left+right
                elif isinstance(node.op,ast.Sub): value=left-right
                elif isinstance(node.op,ast.Mult): value=left*right
                else: value=left/right
            else:
                raise ValueError('Unsupported arithmetic')
            if abs(value)>10**30 or value.denominator>10**20:
                raise ValueError('Arithmetic budget exceeded')
            return value
        return visit(tree.body)
    except (ValueError,TypeError,SyntaxError,ZeroDivisionError,OverflowError,RecursionError):
        return None


def arithmetic_stats(text):
    text=text.replace('×','*').replace('÷','/').replace('−','-')
    checked=correct=0
    errors=[]
    for match in list(EQUATION.finditer(text))[:40]:
        lhs,rhs=map(safe_number,match.groups())
        if lhs is None or rhs is None:
            continue
        checked+=1
        error=abs(float(lhs-rhs))/(1+abs(float(rhs)))
        correct+=int(error<=1e-6)
        errors.append(min(error,1e6))
    return {'checked':checked,'correct':correct,'wrong':checked-correct,
            'mean_error':float(np.mean(errors)) if errors else 0.,'max_error':max(errors,default=0.)}


def coverage(bank):
    rows=[]
    for row in bank:
        question=row['question'].lower()
        qw=set(re.findall(r'[a-z]+',question))-STOP
        qn=set(re.findall(NUMBER,question))
        group=[]
        for c in row['candidates']:
            t=c['text'].lower()
            tw=set(re.findall(r'[a-z]+',t))-STOP
            tn=set(re.findall(NUMBER,t))
            ar=arithmetic_stats(t)
            group.append([np.log1p(ar['checked']),np.log1p(ar['correct']),np.log1p(ar['wrong']),
                ar['correct']/max(ar['checked'],1),ar['wrong']/max(ar['checked'],1),
                np.log1p(ar['mean_error']),np.log1p(ar['max_error']),float(ar['checked']>0),
                len(qw&tw)/max(len(qw),1),len(qw&tw)/max(len(tw),1),
                len(qw&tw)/max(len(qw|tw),1),len(qn&tn)/max(len(qn),1),
                len(tn-qn)/max(len(tn),1),np.log1p(len(tw)),np.log1p(len(tn)),
                float('?' in t),float('not' in tw),float('remaining' in tw),float('total' in tw),
                float('each' in tw),float('per' in tw),float('twice' in tw)])
        group=np.asarray(group)
        # Group-relative evidence is reference-free and invariant to question IDs.
        rows.extend(np.concatenate([group,group-group.mean(axis=0,keepdims=True)],axis=1))
    return np.asarray(rows,dtype=np.float64)


def v3_features(bank,arrays,mode):
    compact=np.concatenate([extract(bank,arrays,'structure'),restricted(bank,'answer'),coverage(bank)],axis=1)
    if mode=='compact':
        return compact
    if mode=='dense':
        return np.concatenate([extract(bank,arrays,'embedding'),compact],axis=1)
    raise ValueError(mode)
