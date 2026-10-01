#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""重建专家意图数据，比较三分类与两阶段学习路由。"""

from __future__ import annotations

import argparse
import importlib
import json
import random
import sys
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

R = importlib.import_module("10_train_package_router_v2")
LABELS = R.BASE_LABEL_NAMES
C, V, U = R.CALCULATOR, R.VERTICAL, R.UNKNOWN
PKG = R.PACKAGE_MARKER

# 训练和测试按表达模板族分开；UNKNOWN 明确包括所有不由这两个专家执行的请求。
TRAIN = {
 C: ["计算{p}", "直接给出{p}的答案", "求出{p}的结果", "算一下{p}", "我只要{p}的最终数值", "请完成{p}", "evaluate {p}", "what is the result of {p}", "帮我求{p}的和", "给我答案：{p}", "把{p}算出来", "返回{p}的值"],
 V: ["用竖式计算{p}", "展示{p}每一列的进位", "从个位开始逐步算{p}", "把{p}按位展开并给出过程", "列出{p}的逐列计算步骤", "说明{p}每一位怎样相加", "work out {p} digit by digit", "show each carry for {p}", "给我看{p}的竖式过程", "按列演算{p}", "详细推导{p}的加法步骤", "逐位写出{p}的中间结果"],
 U: ["把{p}原样抄写出来", "只数一下{p}有几个字符", "把{p}翻译成英文但不要计算", "解释加号这个符号的含义，不要求结果：{p}", "将{p}作为文本放进引号里", "比较{p}和我稍后给的另一段文字", "检查{p}是否符合我给的文本格式", "不要运算，只把{p}改成大写文本", "为包含{p}的句子纠正标点", "把{p}按字符串拆成字符列表", "围绕{p}写一句俏皮话", "忽略算式含义，把{p}当作文件名", "describe the characters in {p} without solving", "translate the text {p}", "repeat {p} exactly", "compare {p} with another string", "count the symbols in {p}", "explain the plus sign in {p}, not its result", "format {p} as quoted text", "classify the text style of {p}"],
}
# 增补真实 package 控制视图；裸算式按默认求值请求标注为 CALCULATOR。
TRAIN[C] += ["{p}", "{p} 求值", "{p} 直接算", "{p} 给出结果", "{p} 算一下", "{p} 计算答案",
             "请算出{p}的答案", "告诉我{p}的计算结果", "帮我把{p}求和", "{p}的结果是多少",
             "calculate {p} and return the answer", "give only the numeric answer for {p}",
             "求一下{p}", "算出{p}", "计算结果：{p}", "求{p}的答案"]
TRAIN[V] += ["{p} 竖式计算", "{p} 做竖式", "{p} 展开步骤", "{p} 逐位计算", "{p} 竖式过程",
             "{p} 列竖式", "{p} 按列计算", "{p} 详细步骤", "请对{p}进行竖式运算",
             "一步一步列出{p}的进位", "显示{p}逐位相加的过程", "show the vertical method for {p}",
             "show the intermediate carries in {p}", "break {p} into column steps"]
TRAIN[U] += ["{p} 只复述", "{p} 不要计算，只检查格式", "{p} 抄写出来，不要求结果",
             "{p} 翻译成英文", "{p} 作为普通文本处理", "{p} 有几个字符",
             "{p} 是否包含加号，只回答是或否", "{p} 按字符拆开", "{p} 改成引号包裹的文本",
             "{p} 不要求计算，只描述格式", "repeat {p} verbatim, do not solve",
             "count characters in {p}", "translate {p} as text", "treat {p} as a string",
             "check whether {p} contains a plus sign", "quote the text {p}"]
HELDOUT = {
 C: ["麻烦告诉我{p}等于多少", "请计算出这个表达式的数值：{p}", "what does {p} evaluate to?", "最终答案是多少：{p}", "求和结果为多少，算式是{p}", "请返回计算结果{p}"],
 V: ["把{p}写成传统加法竖式", "从右往左逐列展示{p}的进位", "演示如何手算{p}", "show the column addition steps for {p}", "给出{p}的每列和值及进位", "逐个数位推演{p}的运算"],
 U: ["逐字复述{p}，保持完全不变", "把{p}中的标点和数字分别列出来", "将{p}翻译为中文，不要求计算", "把{p}用括号括起来作为普通字符串", "检查字符串{p}是否含有加号", "写一段介绍文本处理的说明，并引用{p}", "summarize the text {p} without calculating", "spell out the characters of {p}", "use {p} as a label, do not evaluate it", "is the string {p} valid UTF-8?", "请把{p}和另一个短语做字面比较", "只说明{p}包含哪些字符，不求值"],
}
HELDOUT[C] += ["{p} 是多少", "麻烦把{p}算出来", "what is the value of {p}?", "给出最终数值：{p}"]
HELDOUT[V] += ["{p} 从个位到最高位怎么进位", "将{p}按数位写出运算", "列出算式{p}的手算过程", "calculate {p} using long addition"]
HELDOUT[U] += ["把算式{p}逐字念一遍", "{p}只判断是否为文本中的加号表达式", "write the literal text {p} into a sentence", "extract punctuation from {p} without solving"]

def build(seed: int, n: int):
    rng = random.Random(seed)
    train, test = [], []
    for y, templates in TRAIN.items():
        for _ in range(n):
            train.append((rng.choice(templates).format(p=PKG), y))
    for y, templates in HELDOUT.items():
        test.extend((x.format(p=PKG), y) for x in templates)
    rng.shuffle(train)
    return train, test

def save_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for x,y in rows:
            f.write(json.dumps({"text":x,"label":LABELS[y]},ensure_ascii=False)+"\n")

def read_rows(path):
    rows=[]
    with path.open(encoding="utf-8") as f:
        for n,line in enumerate(f,1):
            r=json.loads(line)
            if set(r)!={"text","label"} or r["label"] not in LABELS or PKG not in r["text"]:
                raise ValueError(f"{path}:{n} invalid record")
            rows.append((r["text"],LABELS.index(r["label"])))
    return rows

class DS(Dataset):
    def __init__(self, rows, binary=None):
        codec=R.get_codec(); self.rows=[]
        for x,y in rows:
            if binary == "execute": y = int(y != U)
            elif binary == "expert":
                if y == U: continue
                y = int(y == V)
            self.rows.append((codec.encode_router_text(x), y))
    def __len__(self): return len(self.rows)
    def __getitem__(self,i): return self.rows[i]

def collate(items):
    codec=R.get_codec(); w=max(len(x) for x,_ in items)
    ids=torch.full((len(items),w),codec.PAD_ID,dtype=torch.long); ys=torch.tensor([y for _,y in items])
    for i,(x,_) in enumerate(items): ids[i,:len(x)]=torch.tensor(x)
    return ids,ids.eq(codec.PAD_ID),ys

def init_model(path, classes, device, binary=None):
    base=torch.load(path,map_location="cpu",weights_only=True)
    model=R.PackageRouterThreeWay(base["width"],base["layers"],base["heads"])
    model.load_state_dict(base["model"])
    if classes != 3:
        old=model.classifier
        original_weight=old.weight.detach().clone(); original_bias=old.bias.detach().clone()
        model.classifier=nn.Linear(old.in_features,classes)
        with torch.no_grad():
            if classes == 2 and binary == "execute":
                # execute binary rows: 0=UNKNOWN, 1=CALCULATOR or VERTICAL.
                model.classifier.weight[0].copy_(original_weight[U])
                model.classifier.bias[0].copy_(original_bias[U])
                model.classifier.weight[1].copy_(0.5*(original_weight[C]+original_weight[V]))
                model.classifier.bias[1].copy_(0.5*(original_bias[C]+original_bias[V]))
            else:
                # expert binary rows: 0=CALCULATOR, 1=VERTICAL.
                model.classifier.weight.copy_(original_weight[[C,V]])
                model.classifier.bias.copy_(original_bias[[C,V]])
    return model.to(device)

def fit(rows, init, classes, out, device, epochs, lr, binary=None):
    torch.manual_seed(404)
    model=init_model(init,classes,device,binary)
    loader=DataLoader(DS(rows,binary),batch_size=64,shuffle=True,collate_fn=collate)
    opt=torch.optim.AdamW(model.parameters(),lr=lr)
    for ep in range(epochs):
        model.train(); loss_sum=0
        for ids,mask,y in loader:
            ids,mask,y=ids.to(device),mask.to(device),y.to(device)
            opt.zero_grad(set_to_none=True); loss=nn.functional.cross_entropy(model(ids,mask),y);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); loss_sum+=loss.item()
        print(f"{out.stem} epoch={ep+1}/{epochs} loss={loss_sum/len(loader):.4f}",flush=True)
    out.parent.mkdir(parents=True,exist_ok=True)
    torch.save({"model":model.cpu().state_dict(),"width":128,"layers":4,"heads":4},out)
    return model.to(device).eval()

def pred(model,x,device):
    codec=R.get_codec(); ids=torch.tensor([codec.encode_router_text(x)],device=device); mask=torch.zeros_like(ids,dtype=torch.bool)
    with torch.inference_mode(): return model(ids,mask).softmax(-1)[0].cpu().tolist()

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prepare",action="store_true"); ap.add_argument("--train",action="store_true")
    ap.add_argument("--out",default="output_router_v4_reviewed"); ap.add_argument("--init-from",default="output_router_v3_three_way/router.pt")
    ap.add_argument("--device",default="cuda",choices=["cuda","cpu"]); ap.add_argument("--per-class",type=int,default=1200)
    ap.add_argument("--epochs",type=int,default=12); ap.add_argument("--seed",type=int,default=404)
    a=ap.parse_args(); root=Path(a.out)
    if a.prepare:
        train,test=build(a.seed,a.per_class)
        save_rows(root/"data"/"train.jsonl",train);save_rows(root/"data"/"heldout.jsonl",test)
        overlap=set(x for x,_ in train)&set(x for x,_ in test)
        if overlap: ap.error(f"训练/留出存在完全重复句：{len(overlap)}")
        print(f"train={len(train)} counts={[sum(y==i for _,y in train) for i in range(3)]}; unique={len(set(x for x,_ in train))}; heldout={len(test)}; overlap=0");return
    if not a.train: ap.error("请提供 --prepare 或 --train")
    if not (root/"data"/"train.jsonl").exists() or not (root/"data"/"heldout.jsonl").exists():
        ap.error("找不到已审核的数据；请先运行 --prepare")
    train=read_rows(root/"data"/"train.jsonl"); test=read_rows(root/"data"/"heldout.jsonl")
    if set(x for x,_ in train)&set(x for x,_ in test): ap.error("训练/留出数据有完全重复句")
    if a.device=="cuda" and not torch.cuda.is_available(): ap.error("CUDA 不可用")
    device=torch.device(a.device)
    print("标签: CALCULATOR=直接结果；VERTICAL=逐位过程；UNKNOWN=不属于这两个专家的任何请求。")
    start=time.perf_counter()
    models={}
    models["three"]=fit(train,a.init_from,3,root/"three_way.pt",device,a.epochs,1e-4)
    models["execute"]=fit(train,a.init_from,2,root/"execute_gate.pt",device,a.epochs,1e-4,"execute")
    models["expert"]=fit(train,a.init_from,2,root/"expert_choice.pt",device,a.epochs,1e-4,"expert")
    # Fixed heldout set: compare overall correctness and UNKNOWN false dispatches.
    results={"baseline_v3":{"correct":0},"three_way":{"correct":0,"unknown_to_expert":0},"hierarchical":{"correct":0,"unknown_to_expert":0},"rows":[]}
    baseline=R.load_router(a.init_from,a.device)
    for x,y in test:
        b,_=R.predict(x,baseline); by=LABELS.index(b); results["baseline_v3"]["correct"]+=by==y
        p3=pred(models["three"],x,device); y3=max(range(3),key=p3.__getitem__)
        pe=pred(models["execute"],x,device); px=pred(models["expert"],x,device)
        yh=(V if px[1]>px[0] else C) if pe[1]>pe[0] else U
        for name,p in (("three_way",y3),("hierarchical",yh)):
            results[name]["correct"]+=p==y
            results[name]["unknown_to_expert"]+= y==U and p!=U
        results["rows"].append({"text":x,"true":LABELS[y],"baseline_v3":b,"three_way":LABELS[y3],"three_confidence":max(p3),"hierarchical":LABELS[yh],"execute_prob":pe[1]})
    for name in ("baseline_v3","three_way","hierarchical"):
        key=name if name!="baseline_v3" else "baseline_v3"
        column=key
        results[name]["accuracy"]=results[name]["correct"]/len(test)
        results[name]["per_class"]={LABELS[y]:{"correct":sum(r["true"]==LABELS[y] and r[column]==LABELS[y] for r in results["rows"]),"total":sum(r["true"]==LABELS[y] for r in results["rows"])} for y in range(3)}
    results["elapsed_seconds"]=time.perf_counter()-start
    (root/"evaluation.json").write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({k:v for k,v in results.items() if k!="rows"},ensure_ascii=False,indent=2))
    print(f"逐条错分见 {root/'evaluation.json'}")

if __name__=="__main__":
    if sys.stdout.encoding and sys.stdout.encoding.lower()!="utf-8": sys.stdout.reconfigure(encoding="utf-8")
    main()
