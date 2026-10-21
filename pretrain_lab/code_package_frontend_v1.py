# -*- coding: utf-8 -*-
"""在 tokenizer 前识别 C 代码区域，生成父/子 Package，并把载荷存入会话缓存。"""

from __future__ import annotations

import argparse
import json
import re
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path


PACKAGE = "<|PKG_CODE|>"
# 语句内部的嵌套原子包：数值字面量与标识符（方案 A，与 PKG_ARITH/PKG_CODE 同族）
PKG_NUM = "<|PKG_NUM|>"
PKG_NAME = "<|PKG_NAME|>"

FENCE_RE = re.compile(r"```(?:c|h|cpp|C)?\s*\n?(.*?)```", re.DOTALL)
INLINE_RE = re.compile(r"`([^`]+)`")
# 声明识别：允许 const / static const / constexpr 等限定符前缀，
# 否则 "const int limit = 20;" 会从 int 处截断、丢掉 const（用户要求支持常量声明）
DECL_RE = re.compile(
    r"\b(?:(?:static|const|constexpr|volatile|extern)\s+)*"
    r"(?:unsigned\s+|signed\s+)?"
    r"(?:char|short|int|long|float|double|void|bool)\s+"
    r"[A-Za-z_]\w*\s*(?:=[^;]*)?;"
)

# C/C++ 关键字：识别标识符时排除，避免把 int/const 当成变量名
C_KEYWORDS = frozenset({
    "auto", "break", "case", "char", "const", "continue", "default", "do",
    "double", "else", "enum", "extern", "float", "for", "goto", "if", "inline",
    "int", "long", "register", "restrict", "return", "short", "signed",
    "sizeof", "static", "struct", "switch", "typedef", "union", "unsigned",
    "void", "volatile", "while", "bool", "true", "false", "class", "namespace",
    "template", "typename", "public", "private", "protected", "new", "delete",
    "constexpr", "nullptr", "this", "virtual", "friend", "operator",
})
# 描述文本里可能出现、但不应当成标识符的词
ATOM_STOPWORDS = frozenset({"C"})

# 原子识别正则：只用 ASCII 边界，避免中文被 \w 吞进标识符
CHAR_LIT_RE = re.compile(r"'(?:\\.|[^'\\])'")
# 数值字面量允许可选正负号（负初始值 short n = -456; 的载荷应含符号）；
# 前置 lookbehind 保证 "x-1" 里紧跟标识符的减号不被当成数值符号
NUM_LIT_RE = re.compile(
    r"(?<![A-Za-z0-9_.])[+-]?"
    r"(?:\d+(?:\.\d+)?(?:[eE][+-]?\d+)?|\.\d+(?:[eE][+-]?\d+)?)"
    r"[uUlLfF]*"
    r"(?![A-Za-z0-9_])"
)
IDENT_RE = re.compile(r"(?<![A-Za-z0-9_])[A-Za-z_][A-Za-z0-9_]*(?![A-Za-z0-9_])")


@dataclass
class ChildPackage:
    """一个语句子包；payload 仅存在缓存，不拼入路由输入。"""

    handle: str
    package_type: str
    payload: str
    start: int
    end: int
    # 嵌套原子包句柄（按出现顺序）；未开启 nested 时为空，保持旧行为不变
    atom_handles: list[str] = field(default_factory=list)


@dataclass
class AtomPackage:
    """语句内部的原子子包：一个数值字面量或一个标识符。"""

    handle: str
    package_type: str      # "num_literal" | "identifier"
    payload: str
    marker: str            # PKG_NUM | PKG_NAME
    parent_handle: str     # 所属语句子包（或 nl_description 子包）
    start: int
    end: int


@dataclass
class ParentPackage:
    """原始代码区域父包，索引该区域下所有语句子包。"""

    handle: str
    package_type: str
    payload: str
    child_handles: list[str]


@dataclass
class SessionCache:
    """单次会话的 Package 载荷缓存。"""

    session_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    parents: dict[str, ParentPackage] = field(default_factory=dict)
    children: dict[str, ChildPackage] = field(default_factory=dict)
    atoms: dict[str, AtomPackage] = field(default_factory=dict)


def split_c_statements(source: str) -> list[tuple[str, int, int]]:
    """按顶层分号或换行切分 C 语句，忽略字符串、注释、括号和 for 头部内的分号。"""
    statements: list[tuple[str, int, int]] = []
    start = 0
    paren = brace = bracket = 0
    quote: str | None = None
    escaped = False
    line_comment = block_comment = False
    i = 0
    while i < len(source):
        ch = source[i]
        nxt = source[i + 1] if i + 1 < len(source) else ""
        if line_comment:
            if ch == "\n":
                line_comment = False
                if paren == brace == bracket == 0:
                    _append_statement(statements, source, start, i)
                    start = i + 1
            i += 1
            continue
        if block_comment:
            if ch == "*" and nxt == "/":
                block_comment = False
                i += 2
            else:
                i += 1
            continue
        if quote:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == quote:
                quote = None
            i += 1
            continue
        if ch == "/" and nxt == "/":
            line_comment = True
            i += 2
            continue
        if ch == "/" and nxt == "*":
            block_comment = True
            i += 2
            continue
        if ch in "\"'":
            quote = ch
        elif ch == "(":
            paren += 1
        elif ch == ")":
            paren = max(0, paren - 1)
        elif ch == "{":
            brace += 1
        elif ch == "}":
            brace = max(0, brace - 1)
            if brace == 0 and paren == bracket == 0:
                _append_statement(statements, source, start, i + 1)
                start = i + 1
        elif ch == "[":
            bracket += 1
        elif ch == "]":
            bracket = max(0, bracket - 1)
        elif ch == ";" and paren == brace == bracket == 0:
            _append_statement(statements, source, start, i + 1)
            start = i + 1
        elif ch == "\n" and paren == brace == bracket == 0:
            _append_statement(statements, source, start, i)
            start = i + 1
        i += 1
    _append_statement(statements, source, start, len(source))
    return statements


def _append_statement(out: list[tuple[str, int, int]], source: str, start: int, end: int) -> None:
    raw = source[start:end]
    left = len(raw) - len(raw.lstrip())
    value = raw.strip()
    if value and not value.startswith(("//", "/*")):
        out.append((value, start + left, end - (len(raw) - len(raw.rstrip()))))


def extract_code_regions(request: str) -> tuple[str, list[tuple[str, int, int]]]:
    """取出 fenced/inline code；无代码围栏时识别保守的 C 声明语句。"""
    matches = list(FENCE_RE.finditer(request))
    if not matches:
        matches = list(INLINE_RE.finditer(request))
    if matches:
        regions = [(m.group(1), m.start(1), m.end(1)) for m in matches]
        cleaned = request
        for m in reversed(matches):
            cleaned = cleaned[:m.start()] + " " + cleaned[m.end():]
        return cleaned.strip(), regions
    match = DECL_RE.search(request)
    if match:
        # 一旦发现明确 C 声明，将其后连续代码区域一并交给词法切分器；
        # 避免只打包第一条语句。自然语言请求通常位于声明之前。
        code = request[match.start():].strip()
        return request[:match.start()].strip(), [(code, match.start(), len(request))]
    return request.strip(), []


def extract_atoms(text: str) -> list[tuple[str, str, str, int, int]]:
    """在一段文本（语句或描述）里按出现顺序切出原子：字符字面量、数值字面量、标识符。

    返回 [(marker, package_type, payload, start, end), ...]。
    - 字符字面量 'A' 归为 num_literal（值的一种）；
    - 标识符排除 C 关键字与停用词，只保留真正的变量名；
    - 只用 ASCII 边界，中文描述里的中文不会被误当成标识符。
    """
    spans: list[tuple[int, int, str, str, str]] = []
    covered = [False] * len(text)

    def overlaps(s: int, e: int) -> bool:
        return any(covered[i] for i in range(s, min(e, len(text))))

    def mark(s: int, e: int) -> None:
        for i in range(s, min(e, len(text))):
            covered[i] = True

    # 1) 字符字面量优先（避免里面的字母被当成标识符）
    for m in CHAR_LIT_RE.finditer(text):
        s, e = m.span()
        spans.append((s, e, PKG_NUM, "num_literal", m.group(0)))
        mark(s, e)
    # 2) 数值字面量
    for m in NUM_LIT_RE.finditer(text):
        s, e = m.span()
        if overlaps(s, e):
            continue
        spans.append((s, e, PKG_NUM, "num_literal", m.group(0)))
        mark(s, e)
    # 3) 标识符（排除关键字/停用词/已覆盖区）
    for m in IDENT_RE.finditer(text):
        s, e = m.span()
        word = m.group(0)
        if overlaps(s, e) or word in C_KEYWORDS or word in ATOM_STOPWORDS:
            continue
        spans.append((s, e, PKG_NAME, "identifier", word))
        mark(s, e)

    spans.sort(key=lambda x: x[0])
    return [(marker, ptype, payload, s, e) for s, e, marker, ptype, payload in spans]


def _attach_atoms(cache: SessionCache, child_handle: str, text: str,
                  base_offset: int) -> list[str]:
    """给一个语句子包挂上嵌套原子包，返回原子句柄列表（按出现顺序）。"""
    atom_handles: list[str] = []
    for marker, ptype, payload, s, e in extract_atoms(text):
        ah = uuid.uuid4().hex
        cache.atoms[ah] = AtomPackage(ah, ptype, payload, marker,
                                      child_handle, base_offset + s, base_offset + e)
        atom_handles.append(ah)
    cache.children[child_handle].atom_handles = atom_handles
    return atom_handles


def package_request(request: str, cache: SessionCache | None = None,
                    nested: bool = False) -> dict:
    """打包代码并缓存载荷；返回待 tokenizer 的 router_view 与仅供审计的缓存句柄。

    nested=True 时，在每个语句子包内部再切出 <|PKG_NUM|>/<|PKG_NAME|> 原子子包
    （方案 A）；默认 False 保持 v2/v3 既有行为与数据 sha256 不变。
    """
    cache = cache or SessionCache()
    intent, regions = extract_code_regions(request)
    child_handles: list[str] = []
    parent_handles: list[str] = []
    atom_handles: list[str] = []
    for region, region_start, _ in regions:
        region_child_handles: list[str] = []
        for statement, local_start, local_end in split_c_statements(region):
            handle = uuid.uuid4().hex
            child = ChildPackage(handle, "code_statement", statement,
                                 region_start + local_start, region_start + local_end)
            cache.children[handle] = child
            child_handles.append(handle)
            region_child_handles.append(handle)
            if nested:
                atom_handles.extend(
                    _attach_atoms(cache, handle, statement, region_start + local_start))
        if region_child_handles:
            parent_handle = uuid.uuid4().hex
            cache.parents[parent_handle] = ParentPackage(
                parent_handle, "code_region", region, region_child_handles
            )
            parent_handles.append(parent_handle)
    has_code = bool(child_handles)
    # 子包数量/载荷均不暴露给路由器，避免其从 payload 或长度推断代码细节。
    router_view = f"{intent} {PACKAGE}".strip() if has_code else intent
    return {
        "session_id": cache.session_id,
        "parent_handle": parent_handles[0] if len(parent_handles) == 1 else None,
        "parent_handles": parent_handles,
        "router_view": router_view,
        "package_type": "code" if has_code else "none",
        "child_handles": child_handles,
        "atom_handles": atom_handles,
        "cache": cache,
    }


def expert_input(cache: SessionCache, child_handle: str, question: str) -> str:
    """路由命中代码专家后才从缓存解包载荷，供子包解析器/专家使用。"""
    child = cache.children[child_handle]
    return f"{PACKAGE}\n语言=C\n子包类型={child.package_type}\n载荷：{child.payload}\n问题：{question}\n回答："


def atom_lines(cache: SessionCache, atom_handles: list[str]) -> str:
    """把原子包渲染成"标记 → 载荷"清单，供专家解包时读取真实数值/名字。"""
    return "；".join(f"{cache.atoms[h].marker} → {cache.atoms[h].payload}"
                    for h in atom_handles)


def expert_input_nested(cache: SessionCache, child_handle: str, question: str,
                        language: str = "C") -> str:
    """嵌套版专家输入：语句子包 + 其内部数值/标识符原子包的解包清单。"""
    child = cache.children[child_handle]
    atoms = atom_lines(cache, child.atom_handles)
    return (f"{PACKAGE}\n语言={language}\n子包类型={child.package_type}\n"
            f"载荷：{child.payload}\n原子包：{atoms}\n问题：{question}\n回答：")


def package_description(description: str, cache: SessionCache | None = None,
                        language: str = "C") -> dict:
    """反向（自然语言描述 → 代码）用的打包器。

    描述里出现的数值与变量名同样打成 <|PKG_NUM|>/<|PKG_NAME|> 原子包并存入缓存：
    路由器只看到"描述里有一个名字包和一个数值包"这种结构信息，看不到 i 和 20；
    专家被选中后才从缓存解包出真实载荷，用于写出代码。
    """
    cache = cache or SessionCache()
    atoms = extract_atoms(description)
    marked, last, atom_handles = [], 0, []
    child_handle = uuid.uuid4().hex
    for marker, ptype, payload, start, end in atoms:
        marked.append(description[last:start])
        marked.append(marker)
        ah = uuid.uuid4().hex
        cache.atoms[ah] = AtomPackage(ah, ptype, payload, marker,
                                      child_handle, start, end)
        atom_handles.append(ah)
        last = end
    marked.append(description[last:])
    marked_view = "".join(marked)
    cache.children[child_handle] = ChildPackage(
        child_handle, "nl_description", marked_view, 0, len(description),
        atom_handles)
    markers = []
    for ah in atom_handles:
        m = cache.atoms[ah].marker
        if m not in markers:
            markers.append(m)
    return {
        "session_id": cache.session_id,
        "child_handle": child_handle,
        "atom_handles": atom_handles,
        "marked_view": marked_view,
        "raw_description": description,
        "marker_kinds": markers,
        "router_view": f"{marked_view} {' '.join(markers)}".strip(),
        "cache": cache,
    }


def expert_input_description(cache: SessionCache, child_handle: str,
                             question: str, language: str = "C") -> str:
    """反向任务的专家输入：带标记的描述 + 原子包解包清单。"""
    child = cache.children[child_handle]
    atoms = atom_lines(cache, child.atom_handles)
    return (f"语言={language}\n子包类型={child.package_type}\n"
            f"描述：{child.payload}\n原子包：{atoms}\n问题：{question}\n回答：")


def _self_test() -> None:
    request = "解释这段代码：int i = 0; char *s = \"a;b\"; for (int j=0; j<3; j++) { i += j; }"
    packaged = package_request(request)
    cache = packaged["cache"]
    payloads = [cache.children[h].payload for h in packaged["child_handles"]]
    assert PACKAGE in packaged["router_view"]
    assert "int i = 0" not in packaged["router_view"]
    assert "a;b" not in packaged["router_view"]
    assert len(payloads) == 3, payloads
    assert payloads[0] == "int i = 0;"
    assert payloads[1] == 'char *s = "a;b";'
    assert payloads[2].startswith("for (") and payloads[2].endswith("}")
    child_view = expert_input(cache, packaged["child_handles"][0], "解释声明")
    assert "int i = 0;" in child_view and PACKAGE in child_view
    # 默认不开嵌套：不产生原子包，v2/v3 既有数据与 sha256 不受影响
    assert packaged["atom_handles"] == [] and not cache.atoms

    # ---- 嵌套原子包（方案 A）----
    nested = package_request("解释代码：const int limit = 20;", nested=True)
    ncache = nested["cache"]
    nh = nested["child_handles"][0]
    natoms = [ncache.atoms[a] for a in nested["atom_handles"]]
    assert [a.payload for a in natoms] == ["limit", "20"], [a.payload for a in natoms]
    assert [a.marker for a in natoms] == [PKG_NAME, PKG_NUM]
    assert all(a.parent_handle == nh for a in natoms)
    # 关键字 const/int 不得被当成标识符
    assert "const" not in [a.payload for a in natoms]
    nview = expert_input_nested(ncache, nh, "这行声明是什么意思？")
    assert "载荷：const int limit = 20;" in nview
    assert f"{PKG_NAME} → limit" in nview and f"{PKG_NUM} → 20" in nview

    # 字符字面量与浮点后缀
    lit = package_request("解释代码：float ratio = 0.5f; char grade = 'B';", nested=True)
    lpayloads = [lit["cache"].atoms[a].payload for a in lit["atom_handles"]]
    assert lpayloads == ["ratio", "0.5f", "grade", "'B'"], lpayloads
    lmarkers = [lit["cache"].atoms[a].marker for a in lit["atom_handles"]]
    assert lmarkers == [PKG_NAME, PKG_NUM, PKG_NAME, PKG_NUM]

    # ---- 反向：自然语言描述 → 代码 ----
    desc = package_description("声明一个整型变量 i，初始值是 20")
    dcache = desc["cache"]
    assert desc["marked_view"] == f"声明一个整型变量 {PKG_NAME}，初始值是 {PKG_NUM}", desc["marked_view"]
    assert [dcache.atoms[a].payload for a in desc["atom_handles"]] == ["i", "20"]
    # 路由视图不得泄漏真实载荷
    assert "20" not in desc["router_view"] and " i" not in desc["router_view"]
    dview = expert_input_description(dcache, desc["child_handle"], "写出对应的 C 声明语句。")
    assert f"{PKG_NAME} → i" in dview and f"{PKG_NUM} → 20" in dview

    print(json.dumps({
        "status": "passed",
        "router_view": packaged["router_view"],
        "child_payloads": payloads,
        "expert_input_example": child_view,
        "payload_hidden_from_router": True,
        "parent_child_index": {h: cache.parents[h].child_handles for h in packaged["parent_handles"]},
        "session_cached": len(cache.children) == 3 and len(cache.parents) == 1,
        "nested_example": nview,
        "nested_atoms": [{"marker": a.marker, "payload": a.payload,
                          "type": a.package_type} for a in natoms],
        "reverse_marked_view": desc["marked_view"],
        "reverse_expert_input": dview,
        "reverse_payload_hidden_from_router": True,
        "backward_compatible_default_no_atoms": True,
    }, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="在 tokenizer 前把 C 代码打包并缓存，再生成路由输入。")
    parser.add_argument("--request", help="原始自然语言请求，可使用反引号或代码围栏")
    parser.add_argument("--self-test", action="store_true", help="验证代码提取、子包切分、缓存及防泄漏")
    parser.add_argument("--out", type=Path, help="可选：将演示缓存快照写为 JSON")
    args = parser.parse_args()
    if args.self_test:
        _self_test()
        return
    if not args.request:
        parser.error("请提供 --request 或 --self-test")
    result = package_request(args.request)
    cache: SessionCache = result.pop("cache")
    result["parents"] = [asdict(cache.parents[h]) for h in result["parent_handles"]]
    result["children"] = [asdict(cache.children[h]) for h in result["child_handles"]]
    result["demo_cache_snapshot"] = "此文件仅演示运行时缓存结构；实际请求由 SessionCache 持有载荷。"
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
