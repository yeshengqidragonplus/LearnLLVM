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
FENCE_RE = re.compile(r"```(?:c|h|cpp|C)?\s*\n?(.*?)```", re.DOTALL)
INLINE_RE = re.compile(r"`([^`]+)`")
DECL_RE = re.compile(
    r"\b(?:unsigned\s+)?(?:char|short|int|long|float|double|void|bool)\s+"
    r"[A-Za-z_]\w*\s*(?:=[^;]*)?;"
)


@dataclass
class ChildPackage:
    """一个语句子包；payload 仅存在缓存，不拼入路由输入。"""

    handle: str
    package_type: str
    payload: str
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


def package_request(request: str, cache: SessionCache | None = None) -> dict:
    """打包代码并缓存载荷；返回待 tokenizer 的 router_view 与仅供审计的缓存句柄。"""
    cache = cache or SessionCache()
    intent, regions = extract_code_regions(request)
    child_handles: list[str] = []
    parent_handles: list[str] = []
    for region, region_start, _ in regions:
        region_child_handles: list[str] = []
        for statement, local_start, local_end in split_c_statements(region):
            handle = uuid.uuid4().hex
            child = ChildPackage(handle, "code_statement", statement,
                                 region_start + local_start, region_start + local_end)
            cache.children[handle] = child
            child_handles.append(handle)
            region_child_handles.append(handle)
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
        "cache": cache,
    }


def expert_input(cache: SessionCache, child_handle: str, question: str) -> str:
    """路由命中代码专家后才从缓存解包载荷，供子包解析器/专家使用。"""
    child = cache.children[child_handle]
    return f"{PACKAGE}\n语言=C\n子包类型={child.package_type}\n载荷：{child.payload}\n问题：{question}\n回答："


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
    print(json.dumps({
        "status": "passed",
        "router_view": packaged["router_view"],
        "child_payloads": payloads,
        "expert_input_example": child_view,
        "payload_hidden_from_router": True,
        "parent_child_index": {h: cache.parents[h].child_handles for h in packaged["parent_handles"]},
        "session_cached": len(cache.children) == 3 and len(cache.parents) == 1,
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
