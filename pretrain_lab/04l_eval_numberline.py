# -*- coding: utf-8 -*-
"""
04l_eval_numberline.py
数轴能力【分层】评估：把"背下来了"和"学会规则了"彻底分开。

============================ 为什么需要这个脚本 ============================
04h_eval_math_v7.py 只测训练范围内（-50~50），v7.2 拿了 100%，但
04i_probe_successor.py 一测范围外就全崩（51->'4'、357->'4'、-99->'1'）。
说明 04h 的 100% 只证明了"记忆完整"，完全没触及"数轴是否形成"。

本脚本按【三个层次】测同一个能力（后继 / +1 位移 / 比较），
一张表就能看出模型到底学到了什么：

  L1 in-dist  训练内      数据里有的转移           -> 记忆即可满分
  L2 hole     held-out 洞 01k 用 --holdout 挖掉的   -> 只有规则能做对
  L3 extrap   真外推      1101~9998，完全没训过     -> 规则 + 位值泛化

判据：
  L1 高、L2/L3 崩  => 查表记忆（v7.2 就是这个形态）
  L1/L2 高、L3 崩  => 学到了局部规则，但绑定在见过的位数上
  L1/L2/L3 全高    => 真正的数轴（位值递推电路形成）✅ 目标

============================ 头号陷阱：prompt 前缀一致性 ============================
本项目已踩过两次（v7.1 的 `=-` BPE 合并、v7 的 `-(-2)` 写法缺失）：
评估 prompt 的 token 序列若不是训练样本的严格前缀，模型看到的输入分布
就和训练时不一样，测出来的低分是【评估脚本的 bug】而不是模型的能力。
所以本脚本启动时先做 [S0] 自检，不通过就直接报错退出，不出假结果。

⚠️ op_num() / eq_res() 与 01j / 01k / 04h 中重复定义，改格式必须同步所有副本。
"""
import argparse
import random
import re
import sys

import torch
from transformers import Qwen2ForCausalLM, AutoTokenizer

try:
    # line_buffering=True：重定向到日志文件时也能实时看到进度。
    # 本脚本要跑 ~20 个子项 × N 次 generate，耗时数分钟；默认全缓冲会让
    # 日志文件在进程结束前一直是 0 字节，无法判断是卡死还是正常在跑。
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass

# ⚠️ MAX_NEW 必须够长：
#    · 计数链 `数到 944 945 946 947 948 949。` ≈ 25 token（首版设 10 全截断，假 0%）
#    · 六位数 CoT `后继 499999：个位9+1=10，写0进1；...（6 步）...，得500000。`
#      每步约 12 token × 6 步 + 答案 6 token ≈ 80 token
MAX_NEW = 96


# ==================== 格式 helper（与 01j / 01k / 04h 逐字一致）====================
def op_num(n):
    """加数/减数位置：负数加括号"""
    return f"({n})" if n < 0 else str(n)


def eq_res(c):
    """结果位置：负结果写 `= -X`（防 `=-` BPE 合并）；正结果写 `=X`"""
    return f"= {c}" if c < 0 else f"={c}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="output_math_v8/final_model")
    parser.add_argument("--line-n", type=int, default=999, help="训练数轴范围（须与 01k 一致）")
    parser.add_argument("--holdout", type=int, default=7,
                        help="01k 挖洞的末位数字（须与 01k --holdout 一致，否则 L2 无意义）")
    parser.add_argument("--extrap-lo", type=int, default=1101,
                        help="真外推下界（须 > 01k 的 --res-max，避开四位进位专项覆盖区）")
    parser.add_argument("--extrap-hi", type=int, default=9998, help="真外推上界")
    parser.add_argument("--n", type=int, default=40, help="每个子项抽样数")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--dense-n", type=int, default=50, help="回归项的稠密范围（对齐 04h）")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    m = Qwen2ForCausalLM.from_pretrained(args.model).to(torch.bfloat16).to(device)
    m.eval()
    m.generation_config.pad_token_id = tok.pad_token_id
    m.generation_config.eos_token_id = tok.eos_token_id

    def gen(prompt):
        """v4+ 约定：输入必须手动加 [eos] 前缀，否则与训练分布不一致"""
        ids = [tok.eos_token_id] + tok.encode(prompt)
        out = m.generate(torch.tensor([ids], dtype=torch.long).to(device),
                         max_new_tokens=MAX_NEW, do_sample=False)
        return tok.decode(out[0], skip_special_tokens=True)

    # ---------- ⚠️ 两个 BPE / 通道陷阱（本脚本首版实测踩到）----------
    # 陷阱一【BPE 合并】相邻对 `-107 -106。` 切分成 ['-','1','0','7','Ġ-','1','0','6','ãĢĤ']：
    #   空格和后继的负号被合并成单个 token `Ġ-`。于是 prompt `-107 `（尾空格）切出
    #   [...,'Ġ']，末尾是裸 `Ġ`——这个状态在训练里【从未出现】（负数的后继仍是负数，
    #   空格永远被合并掉），模型被推进分布外，低分是脚本 bug 而不是模型能力。
    #   => 规则：后继 a+1 < 0 时不加尾空格；a+1 >= 0 时加尾空格（此时不合并）。
    #      a=-1 是边界：后继 0 非负，`-1 0。` 切出裸 `Ġ`，加尾空格正确。
    #
    # 陷阱二【通道污染】即使切分正确，裸数字 prompt 的 next-token 仍被别的任务压死。
    #   实测 v7.2（check_ambig_tmp.py）：
    #       '-9' -> '小于':0.563 | '大于':0.341 | ' -':0.081   <- 比较句压死相邻对
    #       '7'  -> '+'   :0.451 | '-'   :0.374 | '大于':0.089 <- 算式压死相邻对
    #       '-9 '-> '8'   :0.873 | '7 ' -> '8':0.962           <- 只有尾空格能救正数
    #   根因是频次：v7.2 比较句 10100 条 vs 相邻对 408 条 = 24.8 : 1。
    #   而负数【无法】用尾空格消歧（见陷阱一），所以 v7.2 的负数后继必然 0%。
    #   => 解法：01k 支柱四引入任务标记 `后继 a b。`。实测 `后继 a` 对正负数
    #      都是 `后继 a a+1。` 的严格前缀（8/8 通过），且与比较句/算式完全不共享
    #      前缀，歧义从根上消除。本脚本以【标记通道为主测】，裸通道降为对照项。
    SUCC_TAG = "后继"          # 必须与 01k_make_math_v8.py 的 SUCC_TAG 一致

    def succ_prompt(a):
        """裸通道：后继为负时不加尾空格（陷阱一）"""
        return f"{a} " if a + 1 >= 0 else f"{a}"

    def tag_prompt(a):
        """标记通道（主测）：`后继 a` —— 无歧义，正负数都成立"""
        return f"{SUCC_TAG} {a}"

    L, ho = args.line_n, args.holdout
    print(f"模型 {args.model} | 数轴范围 -{L}~{L} | 挖洞末位 {ho} | "
          f"外推 {args.extrap_lo}~{args.extrap_hi} | 每子项 {args.n} | seed {args.seed}")

    # ================= [S0] prompt 前缀一致性自检（不通过则拒绝出结果）=================
    # 训练样本 `a b。` 的 token 序列，必须以评估 prompt `a ` 的 token 序列开头。
    # 若不成立，说明 BPE 把尾空格并进了别的 token，测出来的分数是假的。
    print("\n" + "=" * 66)
    print("[S0] prompt 前缀一致性自检（防评估 prompt 与训练分布不一致）")
    # 覆盖：正数各位数 / 负数各位数 / 进位边界 / 过零边界 / 外推四五位数
    probe_nums = [7, 51, 101, 397, 999, 1234,
                  -1, -9, -10, -107, -999, -5678,
                  9, 99, -100, -1000, 12345, -12345]
    bad = []
    # (a) 标记通道：`后继 a` 必须是 `后继 a a+1。` 的严格前缀（主测，必须全 OK）
    for a in probe_nums:
        p, full = tag_prompt(a), f"{SUCC_TAG} {a} {a + 1}。"
        p_ids, f_ids = tok.encode(p), tok.encode(full)
        ok = f_ids[:len(p_ids)] == p_ids
        if not ok:
            bad.append(p)
        print(f"   {'OK  ' if ok else 'FAIL'} 标记 {a:>7}  prompt={tok.tokenize(p)}"
              f"  样本前缀={tok.tokenize(full)[:len(p_ids)]}")
    # (b) 裸通道：对照项，负数侧已知会被比较句污染，这里只验证切分前缀关系
    for a in probe_nums:
        p = succ_prompt(a)
        p_ids = tok.encode(p)
        f_ids = tok.encode(f"{a} {a + 1}。")
        ok = f_ids[:len(p_ids)] == p_ids
        if not ok:
            bad.append(a)
        note = "尾空格" if p.endswith(" ") else "无尾空格(Ġ-合并)"
        print(f"   {'OK  ' if ok else 'FAIL'} 裸   {a:>7} [{note:<16}] prompt={tok.tokenize(p)}"
              f"  样本前缀={tok.tokenize(f'{a} {a + 1}。')[:len(p_ids)]}")
    # 比较句（无空格，负数不受 `Ġ-` 合并影响，但仍要验证前缀关系成立）
    for a, b in [(9, 10), (-107, -106), (-1, 0), (99, 100), (-1000, -999), (1234, 1240)]:
        full = f"{a}小于{b}。"
        head = f"{a}小于"
        p_ids, f_ids = tok.encode(head), tok.encode(full)
        ok = f_ids[:len(p_ids)] == p_ids
        if not ok:
            bad.append(head)
        print(f"   {'OK  ' if ok else 'FAIL'} 比较 {a}小于{b}  prompt={tok.tokenize(head)}"
              f"  完整={tok.tokenize(full)}")
    # (c) 计数链：`数到 a` 必须是 `数到 a a+1 ... a+5。` 的严格前缀
    COUNT_TAG_S0 = "数到"
    for a in [1, 97, -31, -997, 1234]:
        p = f"{COUNT_TAG_S0} {a}"
        full = f"{COUNT_TAG_S0} " + " ".join(str(a + i) for i in range(6)) + "。"
        p_ids, f_ids = tok.encode(p), tok.encode(full)
        ok = f_ids[:len(p_ids)] == p_ids
        if not ok:
            bad.append(p)
        print(f"   {'OK  ' if ok else 'FAIL'} 计数链 {a:>6}  prompt={tok.tokenize(p)}"
              f"  样本前缀={tok.tokenize(full)[:len(p_ids)]}")
    # 算式侧同样检查：`a+b=` 必须是 `a+b=c` 的前缀
    for expr in ["107+1=", "23+12=", "3-5=", "(-3)+2=", "999+1=", "(-995)+1="]:
        full = {"107+1=": "107+1=108", "23+12=": "23+12=35", "3-5=": "3-5= -2",
                "(-3)+2=": "(-3)+2= -1", "999+1=": "999+1=1000",
                "(-995)+1=": "(-995)+1= -994"}[expr]
        p_ids, f_ids = tok.encode(expr), tok.encode(full)
        ok = f_ids[:len(p_ids)] == p_ids
        if not ok:
            bad.append(expr)
        print(f"   {'OK  ' if ok else 'FAIL'} {expr:<12} prompt={tok.tokenize(expr)}"
              f"  完整={tok.tokenize(full)}")
    if bad:
        print(f"\n❌ [S0] 未通过：{bad}\n   评估 prompt 不是训练样本的 token 前缀，"
              f"后续所有分数都不可信。请先修 01k 的格式约定，不要看下面的结果。")
        return
    print("   ✅ [S0] 通过：所有评估 prompt 都是对应训练样本的严格 token 前缀")
    print("   ⚠️ 注意：S0 只保证【tokenizer 层】前缀一致。若拿 v7.2 这类没见过")
    print(f"      `{SUCC_TAG}`/`{COUNT_TAG_S0}` 标记的旧模型来跑，标记通道会接近 0%——")
    print("      那是诚实的基线（旧模型确实没有这个通道），不是脚本 bug。")

    results = []          # (层次, 项目, 正确数, 总数, 失败样例)

    def pick(pool, n):
        """从池子里抽 n 个；池子空则返回空列表（调用方 run 会标记 SKIP）。
        例：拿 v7.2（--line-n 50）跑本脚本时，held-out 洞定义在三位数区间，
        池子必然为空——此时应跳过而不是崩在 rng.sample 上。"""
        return rng.sample(pool, n) if len(pool) >= n else list(pool)

    def run(level, name, cases, prompt_fn, check_fn, show=5, classifier=None):
        """跑一个子项：cases 是待测对象列表，prompt_fn 造 prompt，check_fn 判对错。
        classifier 可选：返回 'OK'/'AMBIG'/'FAIL'，把"前缀歧义"从"真错"里分出来。
        准确率分母【剔除 AMBIG】——歧义样本既不算对也不算错，它测不出能力。"""
        if not cases:
            print(f"   [{level}] {name:<34} SKIP（样本池为空，检查 --line-n/--holdout）")
            return None
        ok, fails, ambig = 0, [], []
        for c in cases:
            p = prompt_fn(c)
            out = gen(p)
            verdict = classifier(c, out) if classifier else ("OK" if check_fn(c, out) else "FAIL")
            if verdict == "OK":
                ok += 1
            elif verdict == "AMBIG":
                ambig.append((p, out))
            else:
                fails.append((p, out))
        n_eff = len(cases) - len(ambig)
        pct = ok / n_eff * 100 if n_eff else 0.0
        results.append((level, name, ok, n_eff, pct))
        tag = f"（另 {len(ambig)} 条前缀歧义已剔除）" if ambig else ""
        print(f"   [{level}] {name:<34} {ok:>3}/{n_eff:<3} = {pct:5.1f}% {tag}")
        for p, o in fails[:show]:
            print(f"        FAIL  {p!r} -> {o!r}")
        for p, o in ambig[:2]:
            print(f"        AMBIG {p!r} -> {o!r}  <- prompt 是更长样本的前缀，非能力问题")
        return pct

    # ================= 后继（数轴的核心操作）=================
    # 判据：输出必须以 `后继 a a+1` 开头。多位数是逐位切分的，所以这要求模型
    # 连续生成 a+1 的【每一位】都正确，比只看首 token 严格得多。
    #
    # ⚠️ 注意 04h_eval_math_v7.py 的 [A] 项是【假测试】：它写的是
    #       out = gen(f"{a} {b}");  good = out.startswith(f"{a} {b}")
    #    把答案 b 直接喂进了 prompt，实际只测"能否补出句号"。这就是为什么
    #    04h 报 [A] 100% 而 04i 探针一测外推就全崩——那个 100% 从未测过数轴。
    #    本脚本的 prompt 只给 a，答案必须由模型自己生成。
    # ---------- ⚠️ 第三类陷阱：prompt 前缀【歧义】（S0 抓不到）----------
    # S0 只验证 "prompt 是某一条训练样本的前缀"，但没验证它是【唯一】前缀。
    # 实测 v8：`后继 9` 同时是 `后继 9 10。`、`后继 99 100。`、`后继 999 1000。`
    # 的前缀（因为 `后继 9` + `99 1000。` 也合法）。贪心解码会选概率最高的延续，
    # 于是出现 `'后继 9' -> '后继 999 1000。'`、`'后继 98' -> '后继 989 990。'`。
    # 这【不是算错】——模型输出的 999->1000 本身是正确的后继关系，
    # 只是它把 prompt 里的 `9` 当成了 `999` 的开头。
    # 必须把这类单独归为 AMBIG，否则会把"评估 prompt 有歧义"误判成"模型能力不足"。
    # 判据：输出形如 `后继 X Y。`，若 X != a 但 str(X).startswith(str(a))，即歧义。
    _SUCC_RE = re.compile(rf"^{re.escape(SUCC_TAG)}\s+(-?\d+)\s+(-?\d+)")

    def classify_succ(a, out):
        """返回 'OK' / 'AMBIG'（前缀歧义，非能力问题）/ 'FAIL'（真错）"""
        if out.startswith(f"{SUCC_TAG} {a} {a + 1}"):
            return "OK"
        m = _SUCC_RE.match(out)
        if m:
            x = int(m.group(1))
            if x != a and str(x).startswith(str(a)):
                return "AMBIG"          # 模型把 a 读成了更长的数 x 的开头
        return "FAIL"

    def succ_check(a, out):
        return classify_succ(a, out) == "OK"

    def succ_check_bare(a, out):
        """裸通道对照（已知会被比较句/算式污染，仅用于量化污染程度）"""
        return out.startswith(f"{a} {a + 1}")

    # L1 训练内：排除洞（末位 ho 的三位数）
    in_pool = [n for n in range(-L, L + 1)
               if not (100 <= n <= L and n % 10 == ho)]
    # L2 held-out 洞：01k 从训练集删掉的那些对
    hole_pool = [n for n in range(100, L + 1) if n % 10 == ho]
    # L3 真外推：完全没训过的四位数
    ex_pool = list(range(args.extrap_lo, args.extrap_hi + 1))

    print("\n" + "=" * 66)
    print(f"[T1] 后继 `{SUCC_TAG} a a+1。` —— 数轴核心操作，三层对照（标记通道）")
    p_in = run("L1", "训练内相邻对（含进位/借位/过零）",
               pick(in_pool, args.n), tag_prompt, succ_check, classifier=classify_succ)
    p_hole = run("L2", f"held-out 洞（末位 {ho} 的三位数，训练集已删）",
                 pick(hole_pool, args.n), tag_prompt, succ_check, classifier=classify_succ)
    p_ex = run("L3", f"真外推 {args.extrap_lo}~{args.extrap_hi}（四位数，完全没训过）",
               pick(ex_pool, args.n), tag_prompt, succ_check, classifier=classify_succ)

    # 负数侧：绝对值递减、符号保持，借位是难点
    print("\n[T2] 负数后继（绝对值递减 / 借位 / 符号保持）")
    neg_in = [n for n in range(-L, 0) if not (100 <= -n <= L and (-n) % 10 == ho)]
    run("L1", "训练内负数相邻对", pick(neg_in, args.n), tag_prompt, succ_check,
        classifier=classify_succ)
    neg_hole = [n for n in range(-L, -99) if (-n) % 10 == ho]
    run("L2", f"held-out 洞（负数，末位 {ho}）",
        pick(neg_hole, args.n), tag_prompt, succ_check, classifier=classify_succ)
    run("L3", f"真外推负数（-{args.extrap_hi}~-{args.extrap_lo}）",
        pick(list(range(-args.extrap_hi, -args.extrap_lo + 1)), args.n),
        tag_prompt, succ_check, classifier=classify_succ)

    # 进位/借位边界：位值系统最难的地方，单独列出来
    # ⚠️ 这里最容易踩前缀歧义：`后继 9` 是 `后继 999 1000。` 的前缀，
    #    `后继 99` 是 `后继 999 1000。` 的前缀。必须用 classifier 把 AMBIG 分出来。
    print("\n[T3] 进位/借位边界（位值系统最难的转移）")
    edges = [9, 19, 99, 199, 499, 899, 999,          # 正向进位（含 999->1000 四位）
             -10, -20, -100, -110, -990, -1000,      # 负向借位（位数减少）
             -1, 0, 98, 1098]                        # 符号消失 / 过零
    edges = [e for e in edges if not (100 <= e <= L and e % 10 == ho)]
    run("L1", "训练内进位/借位边界（固定清单）", edges, tag_prompt, succ_check,
        show=8, classifier=classify_succ)
    run("L3", "外推进位边界（1999/2999/5999/8999）",
        [1999, 2999, 5999, 8999, 1099, 4999], tag_prompt, succ_check,
        show=6, classifier=classify_succ)

    # ---------- 裸通道对照：量化"通道污染"有多严重 ----------
    # 这一项不是为了拿分，而是为了证明支柱四（任务标记）的必要性：
    # 同样的数学能力，裸 prompt 会因为比较句/算式抢占 next-token 而崩掉。
    print(f"\n[T3b] 裸通道对照 `{a} {a + 1}。`（无任务标记，量化通道污染）")
    b_in = run("L1", "训练内相邻对（裸 prompt）",
               pick(in_pool, args.n), succ_prompt, succ_check_bare, show=4)
    b_neg = run("L1", "训练内负数相邻对（裸 prompt，已知被比较句压死）",
                pick(neg_in, args.n), succ_prompt, succ_check_bare, show=4)

    # ================= 前缀无关性：同一个末位转移，不同位数前缀 =================
    # 这是"规则 vs 记忆"最直接的量化：若模型学到的是局部规则，
    # 末位 d->d+1 的成功率应当与前面挂几位数【无关】。
    print("\n[T4] 前缀无关性（同一末位转移 d->d+1，前缀从 0 位到 4 位）")
    d = 3 if ho != 3 else 4          # 选一个没被挖洞的末位
    # ⚠️ 候选池必须【显式构造】，不能用 while + 拒绝采样：
    #    width=0 时 mk() 恒返回 d，去重后永远凑不满 -> 死循环（首版实测卡死 24 分钟）；
    #    width=1 时只有 9 个候选（13,23,...,93），同样凑不满 12 个。
    # 分层：前缀 0/1/2 位 -> 1~3 位数，在训练范围内（L1）；
    #       前缀 3/4 位 -> 4~5 位数，超出 --line-n，是真外推（L3）。
    pref_spec = [
        (0, [d],                                                  "L1"),
        (1, [p * 10 + d for p in range(1, 10)],                   "L1"),   # 13..93
        (2, [p * 10 + d for p in range(10, 100)],                 "L1"),   # 103..993
        (3, [p * 10 + d for p in range(100, 1000)],               "L3"),   # 1003..9993
        (4, [p * 10 + d for p in range(1000, 10000)],             "L3"),   # 10003..99993
    ]
    pref_rows = []
    for width, pool, lv in pref_spec:
        # 排除落在挖洞里的（末位 ho 的三位数）与超出外推上界的
        pool = [c for c in pool
                if not (100 <= c <= L and c % 10 == ho) and c <= args.extrap_hi]
        if not pool:
            print(f"   [{lv}] 前缀 {width} 位  SKIP（候选池为空）")
            continue
        cases = pick(pool, min(12, len(pool)))
        pct = run(lv, f"前缀 {width} 位（{len(str(cases[0]))} 位数，例 "
                      f"{cases[0]}->{cases[0] + 1}）",
                  cases, tag_prompt, succ_check, show=3, classifier=classify_succ)
        if pct is not None:
            pref_rows.append((width, pct))
    if len(pref_rows) >= 2:
        spread = max(p for _, p in pref_rows) - min(p for _, p in pref_rows)
        print(f"   -> 前缀 0~4 位的成功率极差 = {spread:.1f} 个百分点"
              f"（越小越接近真规则：局部转移与前缀长度解耦）")
    else:
        spread = float("nan")
        print("   -> 有效前缀档位不足 2 个，无法计算极差")

    # ================= +1 位移：把数轴和加法焊在一起的检查 =================
    print("\n[T5] `a+1=` 位移（数轴后继的算术写法）")

    def disp_prompt(a):
        return f"{op_num(a)}+1="

    def disp_check(a, out):
        mch = re.search(r"=\s*(-?\d+)", out)
        return bool(mch) and int(mch.group(1)) == a + 1

    run("L1", "训练内 a+1", pick(in_pool, args.n), disp_prompt, disp_check)
    run("L2", f"held-out 洞 a+1（末位 {ho}）",
        pick(hole_pool, args.n), disp_prompt, disp_check)
    run("L3", "真外推 a+1（四位数）", pick(ex_pool, args.n), disp_prompt, disp_check)
    run("L3", "真外推 (-a)+1（负四位数）",
        pick(list(range(-args.extrap_hi, -args.extrap_lo + 1)), args.n),
        disp_prompt, disp_check)

    # ================= 比较句：序关系是否也跟着泛化 =================
    print("\n[T6] 比较句 `a小于b。`（序关系泛化）")

    def cmp_prompt(pair):
        a, b = pair
        return f"{a}小于{b}。"

    def cmp_check(pair, out):
        a, b = pair
        return out.startswith(f"{a}小于{b}。")

    # 洞里的相邻比较也被 01k 剔除了（`107小于108。` 会泄漏后继），故 L2 是干净判据
    hole_set = set(hole_pool)
    cmp_in = [(a, a + 1) for a in in_pool if min(a, a + 1) not in hole_set]
    run("L1", "训练内相邻比较", pick(cmp_in, args.n), cmp_prompt, cmp_check)
    run("L2", f"held-out 洞比较（X{ho} 小于 X{ho}+1）",
        [(a, a + 1) for a in pick(hole_pool, args.n)], cmp_prompt, cmp_check)
    run("L3", "真外推比较（四位数相邻）",
        [(a, a + 1) for a in pick(ex_pool, args.n)], cmp_prompt, cmp_check)
    run("L3", "真外推比较（四位数跨十位 a<a+6）",
        [(a, a + 6) for a in pick(ex_pool, args.n)], cmp_prompt, cmp_check)

    # ================= [T7] v7.2 回归：稠密小范围算术不许退化 =================
    # 04h 的 [B][C] 两项搬过来，确保 v8 为了数轴没有牺牲已学会的算术。
    print("\n[T7] v7.2 回归项（稠密算术 |a|,|b|,|c|<=%d，不许退化）" % args.dense_n)
    D = args.dense_n

    def arith_check(expect):
        def _f(pair, out):
            mch = re.search(r"=\s*(-?\d+)", out)
            return bool(mch) and int(mch.group(1)) == expect(pair)
        return _f

    add_pool = [(a, b) for a in range(-D, D + 1) for b in range(-D, D + 1) if abs(a + b) <= D]
    run("R", "加法（对齐 04h [B]）", pick(add_pool, args.n),
        lambda p: f"{op_num(p[0])}+{op_num(p[1])}=", arith_check(lambda p: p[0] + p[1]))
    sub_pool = [(a, b) for a in range(-D, D + 1) for b in range(-D, D + 1) if abs(a - b) <= D]
    run("R", "减法（对齐 04h [C]）", pick(sub_pool, args.n),
        lambda p: f"{op_num(p[0])}-{op_num(p[1])}=", arith_check(lambda p: p[0] - p[1]))

    # ================= [T9] CoT 通道（v8.2 新增：算法执行能力）=================
    # 预检（check_cot_probe_tmp.py）证实：模型不会自发走 CoT，必须用 `：` 结尾
    # 的 prompt 强制。判据：输出必须包含正确的逐步推理 + 正确的最终答案。
    # L2c = CoT 洞（末位 ho 的四位数，其 CoT 样本被 01m 删除）：
    #   若 L2c 高 => 模型对没见过的 CoT 样本也能执行算法（学的是算法不是背诵）
    # L3 = 六位数 CoT（完全没训过的宽度）：
    #   若 L3 高 => 算法长度无关，真正的位值理解 + 外推 ✅
    print("\n[T9] CoT 通道 `后继 a：...`（算法执行，v8.2 核心）")

    def cot_prompt(a):
        return f"{SUCC_TAG} {a}："

    def cot_check(a, out):
        """最终答案正确 + 必须含逐步推理（`位`字）。
        答案提取：取输出末尾的整数（`，4484。` / `，得5000。` / 截断无句号均可）。
        ⚠️ 不能用 startswith 判 CoT——中间步骤任何一位错都应算错，
           但答案对+有步骤 = 算法执行成功（步骤细节错误会在 L2c/L3 分数体现）。"""
        if "位" not in out:
            return False
        tail = out.strip().rstrip("。")
        m = re.search(r"(-?\d+)\s*$", tail)
        return bool(m) and int(m.group(1)) == a + 1

    # CoT 洞：末位 ho 的四位数（01m 从 CoT 数据里删掉的）
    cot_hole = [n for n in range(1000, 10000) if n % 10 == ho]
    run("L1", "训练内 CoT（1~3 位）", pick(in_pool, args.n), cot_prompt, cot_check, show=4)
    run("L1", "训练内 CoT（4/5 位）",
        pick([n for n in range(1000, L + 1) if n % 10 != ho], args.n),
        cot_prompt, cot_check, show=4)
    run("L2", f"CoT 洞（末位 {ho} 的四位数，CoT 样本已删）",
        pick(cot_hole, args.n), cot_prompt, cot_check, show=4)
    run("L3", "真外推 CoT（六位数，完全没训过）",
        pick(list(range(100001, 999998)), args.n), cot_prompt, cot_check, show=4)
    run("L3", "真外推 CoT（负六位数）",
        pick(list(range(-999998, -100001)), args.n), cot_prompt, cot_check, show=4)

    # ================= [T8] 计数链：长程一致性 =================
    # v7.2 显式放弃了这个能力（HTML 13.5.6：`0 1 2 ... 8 9` 后续不出 10，
    # 因为数据只含两两相邻对）。v8 支柱一用短计数链补上：模型必须在同一条
    # 样本里【连续多次】应用同一个递推规则，还要跨过进位边界。
    # 这是比单步后继强得多的证据：单步可能是查表，连续 6 步且跨进位不行。
    print("\n[T8] 计数链 `数到 a a+1 ... a+5。`（长程一致性 + 跨进位）")
    CL = 6                     # 与 01k 的 --chain-len 一致
    COUNT_TAG = "数到"

    def chain_prompt(start):
        return f"{COUNT_TAG} {start}"

    def chain_check(start, out):
        want = f"{COUNT_TAG} " + " ".join(str(start + i) for i in range(CL))
        return out.startswith(want)

    # 训练内：正向链（含跨进位起点，如 997 -> 997..1002 跨过 999/1000）
    chain_in = [s for s in range(0, L + 2 - CL) if not (100 <= s <= L and s % 10 == ho)]
    run("L1", "训练内正向链", pick(chain_in, args.n), chain_prompt, chain_check, show=4)
    # 训练内：跨进位链（起点在 X97/X98/X99，链内必含 9->10 或 99->100 转移）
    chain_carry = [s for s in range(0, L + 2 - CL)
                   if s % 10 in (7, 8, 9) and not (100 <= s <= L and s % 10 == ho)]
    run("L1", "训练内跨进位链（起点末位 7/8/9）",
        pick(chain_carry, args.n), chain_prompt, chain_check, show=4)
    # 训练内：负数链（绝对值递减，跨借位）
    chain_neg = [s for s in range(-L, -CL + 1) if not (100 <= -s <= L and (-s) % 10 == ho)]
    run("L1", "训练内负数链", pick(chain_neg, args.n), chain_prompt, chain_check, show=4)
    # 真外推：四位数链（完全没训过的区间）
    chain_ex = list(range(args.extrap_lo, args.extrap_hi + 2 - CL))
    run("L3", "真外推四位数链", pick(chain_ex, args.n), chain_prompt, chain_check, show=4)

    # ================= 汇总 =================
    print("\n" + "=" * 66)
    print("汇总（按层次看能力衰减）")
    for lv, label in (("L1", "训练内（记忆即可）"),
                      ("L2", "held-out 洞（只有规则能对）"),
                      ("L3", "真外推（规则 + 位值泛化）"),
                      ("R", "v7.2 回归（不许退化）")):
        rows = [r for r in results if r[0] == lv]
        if not rows:
            continue
        tot_ok = sum(r[2] for r in rows)
        tot_n = sum(r[3] for r in rows)
        print(f"  {lv} {label:<28} {tot_ok:>4}/{tot_n:<4} = {tot_ok / tot_n * 100:5.1f}%")

    def fmt(v):
        return f"{v:.0f}%" if v is not None else "SKIP"

    print("\n判定：")
    print("  · L1 高 + L2/L3 崩  => 查表记忆（v7.2 形态：范围内 100%，51->'4' p=0.985）")
    print("  · L1/L2 高 + L3 崩  => 局部规则形成，但绑定在训练过的位数上")
    print("  · L1/L2/L3 全高     => 数轴（位值递推电路）真正形成 ✅")
    print(f"\n  · 后继三层（标记通道）：L1={fmt(p_in)}  L2(held-out)={fmt(p_hole)}"
          f"  L3(外推)={fmt(p_ex)}")
    print(f"  · 裸通道对照：正数 L1={fmt(b_in)}  负数 L1={fmt(b_neg)}"
          f"（远低于标记通道 => 证实通道污染，支柱四必要）")
    if spread == spread:      # 非 NaN
        print(f"  · 前缀无关性极差 {spread:.1f}pp（<15pp 可认为前缀已基本解耦）")


if __name__ == "__main__":
    main()
