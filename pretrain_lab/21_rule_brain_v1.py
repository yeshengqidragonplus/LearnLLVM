# -*- coding: utf-8 -*-
"""规则脑 v1：不含任何模型，只验证 Package 协议（DESIGN_protocol_v1.md）可执行。

目的：
- 这是"模板训练四步"的第 1 步（设计稿 §1.6）。规则脑跑不通 = 协议本身有错，
  此时训模型是白训；跑通后，它导出的轨迹就是**已验证正确**的训练数据源。
- 只实现 v1 的基座动作：ASK / ROUTE / OUTPUT / DONE / FAIL（D15 的 6 个减去 VERIFY）。
  VERIFY 按 D8 由程序自动跑，不占模型动作位，所以不出现在轨迹里。

核心不变量（设计稿 §1）：
  I1 模型输出不含明文载荷，参数只能是包引用
  I2 序号由程序独占生成，模型不能创建
  I3 模型只读投影，不读缓存
  I4 每轮一个动作，动作空间封闭
  I5 参数取值全来自封闭词表

与既有代码的关系：
- 包引用语法沿用 <|PKG_*|>#n（D2/D6），不新造 H1 句柄（第一稿的错误）。
- 载荷缓存结构参考 code_package_frontend_v1.SessionCache。
- 三个科室起步全用执行器（§1.5），不需要先训任何医生。

规划方式（§1.6 的关键机制）：
  轨迹不是手写的，是从「目标包类型」在**科室能力表上反向搜索**出来的。
  加一个科室 = 能力表加一行，所有能用它拼出的任务都自动获得轨迹。

用法：
    .venv\\Scripts\\python 21_rule_brain_v1.py --self-test
    .venv\\Scripts\\python 21_rule_brain_v1.py --demo
    .venv\\Scripts\\python 21_rule_brain_v1.py --export data/protocol_v1_traj.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

# Windows 控制台输出 UTF-8，避免中文乱码
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


# 包类型标记（与 tokens.json 一致；此处只用于生成引用文本，不涉及 ID）
PKG_ARITH = "<|PKG_ARITH|>"
PKG_CODE = "<|PKG_CODE|>"
PKG_NUM = "<|PKG_NUM|>"
PKG_NAME = "<|PKG_NAME|>"
PKG_TEXT = "<|PKG_TEXT|>"
PKG_TASK = "<|PKG_TASK|>"

MARKER_OF = {
    "code": PKG_CODE, "num": PKG_NUM, "name": PKG_NAME,
    "text": PKG_TEXT, "task": PKG_TASK, "arith": PKG_ARITH,
}

# 基座动作
ACT_ASK = "<|ASK|>"
ACT_ROUTE = "<|ROUTE|>"
ACT_VERIFY = "<|VERIFY|>"      # D8：程序自动跑，v1 不产生此动作
ACT_OUTPUT = "<|OUTPUT|>"
ACT_DONE = "<|DONE|>"
ACT_FAIL = "<|FAIL|>"
V1_ACTIONS = (ACT_ASK, ACT_ROUTE, ACT_VERIFY, ACT_OUTPUT, ACT_DONE, ACT_FAIL)

# 失败原因码（封闭集，I5）
FAIL_CODES = ("缺能力", "超范围", "数据不足", "矛盾输入")


# ================= 第 2 层：包表与引用 =================


@dataclass
class Package:
    """一个包：类型 + 序号 + 真实载荷 + 来源。

    payload 永不进投影（I1/I3）；投影只看到 marker、序号、kind 与来源。
    """

    seq: int
    kind: str
    payload: str                      # 真实值，只进缓存与日志
    prov: list[int] = field(default_factory=list)   # 来源序号，进投影

    @property
    def marker(self) -> str:
        return MARKER_OF[self.kind]

    @property
    def ref(self) -> str:
        """引用文本，如 <|PKG_NUM|>#3。"""
        return f"{self.marker}#{self.seq}"

    def render(self) -> str:
        """投影里的条目：类型 + 序号 + kind + 来源。不含载荷（I1）。"""
        if self.prov:
            src = ",".join(f"#{p}" for p in self.prov)
            return f"{self.marker}#{self.seq} {self.kind}(来自{src})"
        return f"{self.marker}#{self.seq} {self.kind}(输入)"


class PackageTable:
    """包表：分配序号、保存载荷、生成投影。I2 的落地。"""

    def __init__(self) -> None:
        self._packages: dict[int, Package] = {}
        self._next_seq = 1

    def add(self, kind: str, payload: str,
            prov: list[int] | None = None) -> int:
        """创建一个包并返回序号。**仅执行器调用**，模型无法调用。"""
        seq = self._next_seq
        self._next_seq += 1
        self._packages[seq] = Package(seq=seq, kind=kind, payload=payload,
                                      prov=list(prov or []))
        return seq

    def get(self, seq: int) -> Package:
        if seq not in self._packages:
            raise KeyError(f"未知包引用：#{seq}")
        return self._packages[seq]

    def payload(self, seq: int) -> str:
        return self.get(seq).payload

    def kind(self, seq: int) -> str:
        return self.get(seq).kind

    def all_seqs(self) -> list[int]:
        return sorted(self._packages)

    def __len__(self) -> int:
        return len(self._packages)

    def projection(self) -> str:
        """[包] 行：只有类型与来源，绝不出现载荷（I1）。"""
        if not self._packages:
            return "空"
        return "  ".join(self._packages[s].render() for s in self.all_seqs())

    def dump(self) -> dict:
        return {s: {"seq": p.seq, "kind": p.kind, "marker": p.marker,
                    "payload": p.payload, "prov": p.prov}
                for s, p in sorted(self._packages.items())}


# ================= 第 3 层：投影 =================


class Projection:
    """生成模型唯一可见的投影文本（设计稿 §4）。

    格式逐字固定，所有医生共用——投影的读者已经是不同权重的模型（§1.4 C3），
    各科室微调措辞会让交接在训练阶段就悄悄错位。
    """

    LOG_WINDOW = 3   # D8 里的日志窗口：只留最近 3 条，更早压成一句

    def __init__(self, task: str, packages: PackageTable, max_rounds: int = 32):
        self.task = task
        self.packages = packages
        self.max_rounds = max_rounds
        self.log: list[str] = []
        self.error: str | None = None
        self.round = 0

    def append_log(self, line: str) -> None:
        self.log.append(line)

    def _log_block(self) -> str:
        if not self.log:
            return "空"
        shown = self.log[-self.LOG_WINDOW:]
        offset = len(self.log) - len(shown)
        parts = [f"前{offset}步已完成"] if offset > 0 else []
        parts += [f"{i}.{line}" for i, line in enumerate(shown, start=offset + 1)]
        return "; ".join(parts)

    def render(self) -> str:
        lines = [
            f"[任务] {self.task}",
            f"[轮] {self.round}/{self.max_rounds}",
            f"[包] {self.packages.projection()}",
            f"[日志] {self._log_block()}",
        ]
        if self.error:
            lines.append(f"[错误] {self.error}")
        lines.append("[动作]")
        return "\n".join(lines)


# ================= 科室能力表（§1.6 的规划基础） =================


@dataclass
class DeptResult:
    """科室产出：一个包的 kind + 真实载荷。"""

    kind: str
    payload: str


class Department:
    """一个科室（医生或执行器）。协议不关心它是模型还是程序（§1.5）。"""

    name: str
    accepts: tuple[str, ...]      # 接受的包 kind（可多个）
    produces: tuple[str, ...]     # 产出的包 kind（可多个）
    # 聚合科室：accepts 与 produces 同 kind，作用是把多个同 kind 包并成一个。
    # 这类科室在 BFS 规划里会被自环剪枝掉，所以必须显式标注（见 RoutePlanner）。
    aggregates: bool = False

    def run(self, inputs: list[Package]) -> list[DeptResult]:
        raise NotImplementedError


class DeclDept(Department):
    """声明科执行器：从代码包里抽出常量数值，**每个数值一个包**。

    模拟 20 号的解包结果（真实实现应调用 output_decl_packed_v1 医生）。
    """

    name = "声明科"
    accepts = ("code",)
    produces = ("num",)

    def run(self, inputs: list[Package]) -> list[DeptResult]:
        code = inputs[0].payload
        nums = re.findall(r"\b\d+\b", code)
        if not nums:
            raise ValueError("代码里没有可提取的数值")
        return [DeptResult(kind="num", payload=n) for n in nums]


class ArithDept(Department):
    """算数科执行器：把收到的多个数值相加（模拟 09 号的执行器）。

    aggregates=True：它把 N 个 num 包并成 1 个 num 包，kind 不变。
    若在 BFS 里当普通边，num→num 会被 visited 剪掉，导致"两个数根本没相加"
    （本文件第一版就是这么错的：规划出 声明科→进制转换，跳过算数科）。
    """

    name = "算数科"
    accepts = ("num",)
    produces = ("num",)
    aggregates = True

    def run(self, inputs: list[Package]) -> list[DeptResult]:
        total = sum(int(p.payload) for p in inputs)
        return [DeptResult(kind="num", payload=str(total))]


class HexDept(Department):
    """进制转换执行器：十进制 → 十六进制（一行 Python）。"""

    name = "进制转换"
    accepts = ("num",)
    produces = ("text",)

    def run(self, inputs: list[Package]) -> list[DeptResult]:
        value = int(inputs[0].payload)
        return [DeptResult(kind="text", payload=format(value, "X") + "H")]


class RoutePlanner:
    """能力表上的反向搜索：从初始 kind 找到达目标 kind 的科室链（§1.6）。

    加一个科室 = 这张表加一行，所有能用它拼出的任务自动获得轨迹。
    用 BFS 保证最短路径。

    聚合科室（aggregates=True，kind 自环）的正确处理方式：
    **状态里必须带上"哪些聚合科室已用过"**，只按 kind 去重会把它剪掉。

    本项目踩过的坑：只按 kind 做 visited，`num→num` 自环被剪，
    于是规划成 声明科→进制转换，两个数根本没相加；结果虽能跑完 4 轮，
    但语义是错的。用 (kind, 已用聚合集) 做状态后恢复为
    声明科→算数科→进制转换，才是 L1 要的那条链。
    """

    def __init__(self, depts: list[Department]):
        self.depts = depts

    def plan(self, start_kind: str, goal_kind: str,
             start_count: int = 1) -> list[Department] | None:
        """返回科室序列；找不到路径返回 None（对应 FAIL 缺能力）。

        Args:
            start_count: 初始同 kind 包个数。>1 时若存在聚合科室，路径前先插一次。
        """
        queue: deque[tuple[str, tuple[str, ...], list[Department]]] = deque(
            [(start_kind, (), [])])
        visited: set[tuple[str, tuple[str, ...]]] = {(start_kind, ())}
        while queue:
            kind, used_agg, path = queue.popleft()
            if kind == goal_kind and path:
                return path
            for dept in self.depts:
                if kind not in dept.accepts:
                    continue
                if dept.aggregates:
                    if dept.name in used_agg:
                        continue        # 聚合每个 kind 只做一次
                    if start_count == 1 and not path:
                        continue        # 单包时聚合无意义
                    new_kind = kind
                    new_agg = used_agg + (dept.name,)
                else:
                    new_kind = dept.produces[0]
                    new_agg = used_agg
                state = (new_kind, new_agg)
                if state in visited:
                    continue
                visited.add(state)
                new_path = path + [dept]
                if new_kind == goal_kind:
                    return new_path
                queue.append((new_kind, new_agg, new_path))
        # 起点本身就是目标
        if start_kind == goal_kind:
            return []
        return None


# ================= 动作校验（I2 / I5） =================

REF_RE = re.compile(r"^#(\d+)$")

# 每个动作的参数名序列，用于校验个数（固定位置，D2 已采纳）
ACTION_PARAMS: dict[str, tuple[str, ...]] = {
    ACT_ASK: ("slot",),
    ACT_ROUTE: ("refs", "dept"),
    ACT_VERIFY: ("ref", "checker"),
    ACT_OUTPUT: ("ref",),
    ACT_DONE: (),
    ACT_FAIL: ("code",),
}


class ActionError(Exception):
    """动作被拒绝：格式非法、引用不存在、参数不合法。"""


def parse_action(text: str) -> tuple[str, list[str]]:
    """解析一行动作 → (标记, 参数列表)。

    "格式非法"单独成一类错误，因为它指向投影格式问题而非语义问题（§9.1）。
    """
    text = text.strip()
    for mark in V1_ACTIONS:
        if text.startswith(mark):
            rest = text[len(mark):].strip()
            return mark, rest.split() if rest else []
    raise ActionError(f"格式非法：未以合法动作标记开头 -> {text[:40]!r}")


def parse_ref(token: str) -> int:
    m = REF_RE.match(token)
    if not m:
        raise ActionError(f"引用格式错：{token!r} 应为 #<序号>")
    return int(m.group(1))


def validate(mark: str, params: list[str], packages: PackageTable) -> None:
    """校验参数，全部通过才允许执行（I2/I5）。"""
    need = ACTION_PARAMS[mark]
    if len(params) != len(need):
        raise ActionError(f"参数个数错：{mark} 需要 {len(need)} 个"
                          f"（{','.join(need)}），实际 {len(params)} 个")
    if mark == ACT_ROUTE:
        refs, dept = params
        for token in refs.split(","):
            if token:
                packages.get(parse_ref(token))
        packages.get(parse_ref(dept))
    elif mark in (ACT_VERIFY, ACT_OUTPUT):
        packages.get(parse_ref(params[0]))
    elif mark == ACT_FAIL:
        if params[0] not in FAIL_CODES:
            raise ActionError(f"失败原因码不在封闭集：{params[0]!r}")
    elif mark == ACT_ASK:
        if not params[0].strip():
            raise ActionError("ASK 的槽位名不能为空")


# ================= 循环控制器（规则脑） =================


class RuleBrain:
    """规则脑：用确定性规则替代模型出动作。

    它不追求"能用"，只证明**协议可执行**。跑通后导出的轨迹就是
    训练数据的 teacher（§1.6 第 2 步）。
    """

    MAX_ROUNDS = 32
    GOAL_KIND = "text"      # v1 的交付物是文本包

    def __init__(self) -> None:
        self.packages = PackageTable()
        self.depts: list[Department] = [DeclDept(), ArithDept(), HexDept()]
        self.dept_seq: dict[str, int] = {}      # 科室名 → 包引用序号（D6）
        self.trace: list[dict] = []

    # ---- 科室注册：科室名也走包引用（D6）----
    def _register_depts(self) -> None:
        for dept in self.depts:
            seq = self.packages.add("name", dept.name, prov=[])
            self.dept_seq[dept.name] = seq

    def _dept_by_seq(self, seq: int) -> Department:
        name = self.packages.payload(seq)
        for dept in self.depts:
            if dept.name == name:
                return dept
        raise ActionError(f"#{seq} 不是已注册的科室")

    # ---- 规划：能力表反向搜索（每轮重新规划，只看下一步） ----
    def next_dept(self, kind: str) -> Department | None:
        """从当前 kind 到目标 kind 的最短路径，取第一步的科室。

        为什么不能一次性规划完整路径：某个科室会产出**几个包**只有执行后才知道
        （声明科产出 2 个 num），所以聚合步骤必须由执行时发现，不能预先定死。
        """
        planner = RoutePlanner(self.depts)
        path = planner.plan(kind, self.GOAL_KIND, 1)
        return path[0] if path else None

    def aggregator(self, kind: str) -> Department | None:
        for dept in self.depts:
            if dept.aggregates and kind in dept.accepts:
                return dept
        return None

    # ---- 主循环 ----
    def run(self, task: str, initial: list[tuple[str, str]],
            verbose: bool = True) -> dict:
        """跑完一条轨迹。

        Args:
            task: 任务描述（前端给出，不含载荷）
            initial: 初始包 [(kind, payload)]
        """
        self.packages = PackageTable()
        self.dept_seq = {}
        self.trace = []
        self._register_depts()

        start_seqs = [self.packages.add(kind, payload, prov=[])
                      for kind, payload in initial]
        if not start_seqs:
            raise ValueError("至少需要一个初始包")

        proj = Projection(task, self.packages, self.MAX_ROUNDS)

        if verbose:
            print("=" * 70)
            print(proj.render())
            print("=" * 70)

        # 逐轮动态规划：每轮只看"当前 kind → 目标"的第一步。
        # 不能提前规划完整路径：声明科产出几个 num 只有执行后才知道。
        batch = start_seqs
        idx = 0
        while True:
            kinds = {self.packages.kind(s) for s in batch}
            if len(kinds) > 1:
                fail = f"{ACT_FAIL} 矛盾输入"      # 混 kind 批次，v1 不支持
                self._record(proj, idx + 1, fail, "批次里混了不同 kind")
                return self._summary(task, fail)
            kind = kinds.pop()

            if len(batch) > 1:
                # 多个同 kind 包：必须先聚合，否则下游只拿到一个
                dept = self.aggregator(kind)
                if dept is None:
                    fail = f"{ACT_FAIL} 缺能力"
                    self._record(proj, idx + 1, fail, "多个同 kind 包无法合并")
                    return self._summary(task, fail)
            elif kind == self.GOAL_KIND:
                break                              # 到达目标 kind
            else:
                dept = self.next_dept(kind)
                if dept is None:
                    fail = f"{ACT_FAIL} 缺能力"
                    self._record(proj, idx + 1, fail, "能力表无路径")
                    return self._summary(task, fail)

            idx += 1
            proj.round = idx
            before = proj.render()
            refs = ",".join(f"#{s}" for s in batch)
            action = f"{ACT_ROUTE} {refs} #{self.dept_seq[dept.name]}"
            try:
                validate(ACT_ROUTE, [refs, f"#{self.dept_seq[dept.name]}"],
                         self.packages)
                _, produced, result = self._exec_route(batch, dept, refs)
            except (ActionError, KeyError, ValueError) as exc:
                proj.error = str(exc)
                self._record(proj, idx, action, f"[错误] {exc}", before)
                return self._summary(task, action)
            proj.error = None
            proj.append_log(result)
            self._record(proj, idx, action, result, before)
            if verbose:
                print(f"--- 第 {idx} 轮 ---\n{before}\n>>> {action}\n    {result}\n")
            batch = produced
            if idx >= self.MAX_ROUNDS:
                fail = f"{ACT_FAIL} 超范围"
                self._record(proj, idx, fail, "轮次超限")
                return self._summary(task, fail)

        # 自动 VERIFY（D8：科室返回后程序自动跑，不占模型动作位）
        verify_note = self._auto_verify(batch)
        proj.round += 1
        before = proj.render()
        action = f"{ACT_OUTPUT} #{batch[0]}"
        proj.append_log(f"[自动校验] {verify_note}")
        proj.append_log(action)
        self._record(proj, proj.round, action, verify_note, before)
        if verbose:
            print(f"--- 第 {proj.round} 轮 ---\n{before}\n>>> {action}\n"
                  f"    {verify_note}\n")

        proj.round += 1
        before = proj.render()
        self._record(proj, proj.round, ACT_DONE, "结束", before)
        if verbose:
            print(f"--- 第 {proj.round} 轮 ---\n{before}\n>>> {ACT_DONE}\n")
            print("=" * 70)
            print(f"轨迹结束：{len(self.trace)} 轮，共 {len(self.packages)} 个包")
        return self._summary(task, ACT_DONE)

    # ---- 执行一个 ROUTE ----
    def _exec_route(self, batch: list[int], dept: Department,
                    refs: str) -> tuple[bool, list[int], str]:
        inputs = [self.packages.get(s) for s in batch]
        wrong = [p for p in inputs if p.kind not in dept.accepts]
        if wrong:
            raise ActionError(f"科室 {dept.name} 不接受 kind={wrong[0].kind}")
        results = dept.run(inputs)
        produced = [self.packages.add(r.kind, r.payload, prov=batch)
                    for r in results]
        refs_out = ",".join(f"#{s}" for s in produced)
        return True, produced, f"{ACT_ROUTE} {refs} #{self.dept_seq[dept.name]} → {refs_out}"

    def _auto_verify(self, batch: list[int]) -> str:
        """程序自动校验产出包（D8）。不产出动作，只记一句结论。"""
        checks = []
        for seq in batch:
            pkg = self.packages.get(seq)
            if pkg.kind == "num":
                ok = pkg.payload.lstrip("-").isdigit()
                checks.append(f"#{seq} 数字{'合法' if ok else '非法'}")
            elif pkg.kind == "text":
                ok = bool(pkg.payload.strip())
                checks.append(f"#{seq} 非空{'通过' if ok else '失败'}")
        return "；".join(checks) if checks else "无产出"

    def _record(self, proj: Projection, rnd: int, action: str,
                result: str, before: str | None = None) -> None:
        self.trace.append({
            "round": rnd,
            "projection": before if before is not None else "",
            "action": action,
            "result": result,
        })

    def _summary(self, task: str, last_action: str) -> dict:
        return {
            "task": task,
            "rounds": len(self.trace),
            "finished": last_action.strip() == ACT_DONE,
            "packages": len(self.packages),
            "packages_table": self.packages.dump(),
            "trace": self.trace,
        }


# ================= 前端（规则打包，v1 最小版） =================


def frontend_package(user_text: str) -> tuple[str, list[tuple[str, str]]]:
    """最小前端：从原话里抽出代码片段，打包成 code 包。

    真实前端需要处理指代/省略/歧义（那是交流层的活，§1.7）；
    v1 只认"带声明的代码片段"，够跑 L1 即可。
    """
    m = re.search(
        r"((?:(?:static|const|unsigned|signed)\s+)*"
        r"(?:char|short|int|long|float|double|bool)\s+\w+\s*=[^;]*;\s*)+",
        user_text)
    if not m:
        return user_text, []
    return "把这两个常量加起来，用十六进制告诉我", [("code", m.group(0).strip())]


# ================= 自测 / 演示 / 导出 =================


L1_TASK = "把这两个常量加起来，用十六进制告诉我"
L1_INIT = [("code", "const int A = 1234567; const int B = 7654321;")]


def _self_test() -> None:
    brain = RuleBrain()
    result = brain.run(L1_TASK, L1_INIT, verbose=False)

    assert result["finished"], f"轨迹未正常结束：{result['trace'][-1]}"
    assert result["rounds"] == 5, f"预期 5 轮，实际 {result['rounds']}"

    actions = [t["action"].split()[0] for t in result["trace"]]
    assert actions == [ACT_ROUTE, ACT_ROUTE, ACT_ROUTE, ACT_OUTPUT, ACT_DONE], \
        f"动作序列不符：{actions}"

    # 三个科室各走一次
    depts = [brain.packages.payload(int(t["action"].split()[2].lstrip("#")))
             for t in result["trace"][:3]]
    assert depts == ["声明科", "算数科", "进制转换"], f"科室序列不符：{depts}"

    table = result["packages_table"]
    payloads = [v["payload"] for v in table.values()]
    assert "1234567" in payloads and "7654321" in payloads, "数值包缺失"
    assert "8888888" in payloads, f"加法结果错：{payloads}"
    assert "87A238H" in payloads, f"十六进制结果错：{payloads}"

    # I1 核心断言：投影里绝不出现载荷明文
    for rec in result["trace"]:
        for bad in ("1234567", "7654321", "8888888", "87A238H"):
            assert bad not in rec["projection"], \
                f"投影泄漏了载荷 {bad}：\n{rec['projection']}"

    # 投影里必须有 <|PKG_*>#n 形式的引用（D2/D6）
    first = result["trace"][0]["projection"]
    assert f"{PKG_CODE}#" in first and f"{PKG_NAME}#" in first, first

    # 格式非法单独识别
    try:
        parse_action("随便说点什么")
        raise AssertionError("格式非法应被拒绝")
    except ActionError as exc:
        assert "格式非法" in str(exc)

    # 引用不存在的包被拒绝
    try:
        validate(ACT_OUTPUT, ["#99"], brain.packages)
        raise AssertionError("不存在的引用应被拒绝")
    except KeyError:
        pass

    # 参数个数错被拒绝
    try:
        validate(ACT_DONE, ["多余参数"], brain.packages)
        raise AssertionError("多余参数应被拒绝")
    except ActionError as exc:
        assert "参数个数错" in str(exc)

    # 能力表缺路径 → FAIL 缺能力（不是硬凑一个错误动作）
    no_path = RuleBrain()
    no_path.depts = [DeclDept()]      # 没有进制转换，到不了 text
    r2 = no_path.run(L1_TASK, L1_INIT, verbose=False)
    assert not r2["finished"], "缺能力时必须终止"
    assert r2["trace"][-1]["action"] == f"{ACT_FAIL} 缺能力", r2["trace"][-1]

    print(json.dumps({
        "status": "passed",
        "rounds": result["rounds"],
        "actions": actions,
        "depts": depts,
        "packages": result["packages"],
        "hex_result": "87A238H",
        "projection_has_no_payload": True,
        "format_error_distinguished": True,
        "missing_ref_rejected": True,
        "param_count_rejected": True,
        "fail_on_missing_dept": True,
    }, ensure_ascii=False, indent=2))


def _demo() -> None:
    """跑一遍 L1 并打印完整轨迹（人工检查投影可读性）。"""
    user_text = ("这段代码 const int A = 1234567; const int B = 7654321; "
                 "把这两个常量加起来，用十六进制告诉我")
    task, initial = frontend_package(user_text)
    print(f"用户原话：{user_text}\n")
    brain = RuleBrain()
    brain.run(task, initial, verbose=True)


def _export(out: Path) -> dict:
    """导出 L1 轨迹为训练样本（每步一条，D9）。

    {"prompt": 投影, "target": 动作行}，可直接配 03 的 --target-mask --no-pack。
    target 只有静态标记 + 短序号，零字符复制。
    """
    brain = RuleBrain()
    result = brain.run(L1_TASK, L1_INIT, verbose=False)

    rows = []
    for rec in result["trace"]:
        if not rec["projection"]:
            continue
        action = rec["action"].split(" → ")[0].strip()
        rows.append({
            "kind": "protocol_step",
            "prompt": rec["projection"],
            "target": action,
            "text": rec["projection"] + "\n" + action,   # 兼容 03 的 text 字段
            "round": rec["round"],
        })

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    # 审计：target 只含标记 + 序号，不得出现载荷明文
    for row in rows:
        for bad in ("1234567", "7654321", "8888888", "87A238H"):
            assert bad not in row["target"], f"target 泄漏载荷：{row['target']}"
        assert row["target"].split()[0] in V1_ACTIONS, row["target"]

    return {"samples": len(rows), "out": str(out),
            "targets": [r["target"] for r in rows],
            "package_table": result["packages_table"]}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="规则脑 v1：不含模型，验证 Package 协议可执行。")
    parser.add_argument("--self-test", action="store_true",
                        help="自测：L1 五轮轨迹 + 投影不泄漏载荷 + 错误分类")
    parser.add_argument("--demo", action="store_true",
                        help="打印 L1 完整轨迹，人工检查投影可读性")
    parser.add_argument("--export", type=Path,
                        help="导出轨迹训练样本（每步一条）到指定 jsonl")
    args = parser.parse_args()

    if args.self_test:
        _self_test()
    elif args.demo:
        _demo()
    elif args.export:
        print(json.dumps(_export(args.export), ensure_ascii=False, indent=2))
    else:
        parser.error("请提供 --self-test / --demo / --export 之一")


if __name__ == "__main__":
    main()
