#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""用不透明 package 替换加法表达式，验证前置打包与原文恢复。"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass


ARITHMETIC_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_.+\-*/^()])\d+\s*\+\s*\d+(?![A-Za-z0-9_.+\-*/^()])"
)
PACKAGE_MARKER = "<|PKG_ARITH|>"


@dataclass(frozen=True)
class Package:
    """运行时保存的 package；handle 不放进路由器输入文本。"""

    handle: str
    kind: str
    raw_payload: str
    start: int
    end: int


@dataclass(frozen=True)
class RouteResult:
    """路由结果；不支持或不确定的请求统一使用 UNKNOWN。"""

    route: str
    reason: str
    result: str | None = None


class PackageSession:
    """将首个整数加法表达式换成通用标记，并在会话表中保留原文。"""

    def __init__(self, text: str) -> None:
        self.original_text = text
        matches = list(ARITHMETIC_PATTERN.finditer(text))
        if len(matches) > 1:
            raise ValueError(
                f"v1 暂时只支持一个整数加法表达式；检测到 {len(matches)} 个。"
            )
        self.packages: dict[str, Package] = {}
        if matches:
            match = matches[0]
            self.packages["p0001"] = Package(
                handle="p0001",
                kind="arithmetic",
                raw_payload=match.group(0),
                start=match.start(),
                end=match.end(),
            )
            self.controller_view = (
                text[: match.start()] + PACKAGE_MARKER + text[match.end() :]
            )
        else:
            self.controller_view = text
        # 统一 package 路由视图中的水平空白；原始 payload 保持逐字不变。
        self.controller_view = re.sub(r"[^\S\r\n]+", " ", self.controller_view).strip()

    def unpack(self, handle: str) -> str:
        """按运行时句柄恢复 package 的原始载荷。"""
        try:
            return self.packages[handle].raw_payload
        except KeyError as exc:
            raise KeyError(f"未知 package 句柄：{handle}") from exc


DIRECT_INTENT = re.compile(
    r"计算|算一下|算出|求和|求结果|结果|答案|等于多少|是多少|"
    r"\b(?:calculate|compute|sum)\b",
    re.IGNORECASE,
)
EXPLANATION_INTENT = re.compile(
    r"竖式|步骤|过程|推导|解释|怎么计算|如何计算|"
    r"\b(?:step[- ]by[- ]step|show (?:your )?work|explain)\b",
    re.IGNORECASE,
)
NEGATIVE_INTENT = re.compile(
    r"不要|不需要|别.*(?:算|计算)|\b(?:do not|don't|dont|never)\b",
    re.IGNORECASE,
)


def route_request(controller_view: str, package_kinds: list[str]) -> RouteResult:
    """只依据路由视图和 package 类型选工具，不读取 package payload。"""
    if package_kinds != ["arithmetic"] or controller_view.count(PACKAGE_MARKER) != 1:
        return RouteResult("UNKNOWN", "缺少唯一且受支持的算式 package")
    if EXPLANATION_INTENT.search(controller_view):
        return RouteResult("UNKNOWN", "当前尚未接入竖式/解释专家")
    if NEGATIVE_INTENT.search(controller_view):
        return RouteResult("UNKNOWN", "请求明确表示不执行计算")
    # 仅输入一个已识别的算式时，按计算请求处理；payload 仍不进入路由器。
    if controller_view.strip() == PACKAGE_MARKER:
        return RouteResult("CALCULATOR", "裸算式默认请求计算")
    if not DIRECT_INTENT.search(controller_view):
        return RouteResult("UNKNOWN", "未识别到明确的直接计算意图")
    return RouteResult("CALCULATOR", "直接计算意图 + 算式 package")


def calculator_expert(raw_payload: str) -> str:
    """执行正整数二元加法；解析失败时交由上层返回 UNKNOWN。"""
    match = re.fullmatch(r"\s*(\d+)\s*\+\s*(\d+)\s*", raw_payload)
    if not match:
        raise ValueError("算式不符合 v1 整数加法专家的输入格式")
    return str(int(match.group(1)) + int(match.group(2)))


def process_request(text: str) -> tuple[PackageSession | None, RouteResult]:
    """打包、路由并按需调用计算专家；所有未命中路径返回 UNKNOWN。"""
    try:
        session = PackageSession(text)
    except ValueError as exc:
        return None, RouteResult("UNKNOWN", str(exc))
    kinds = [package.kind for package in session.packages.values()]
    route = route_request(session.controller_view, kinds)
    if route.route != "CALCULATOR":
        return session, route
    try:
        value = calculator_expert(session.unpack("p0001"))
    except (KeyError, ValueError) as exc:
        return session, RouteResult("UNKNOWN", f"计算专家拒绝输入：{exc}")
    return session, RouteResult("CALCULATOR", route.reason, value)


def self_test() -> None:
    """检查打包、路由、计算、未知回退及日期不误打包。"""
    session = PackageSession("请对 1234567 + 2222 做竖式计算。")
    assert session.controller_view == "请对 <|PKG_ARITH|> 做竖式计算。"
    assert "1234567" not in session.controller_view
    assert session.unpack("p0001") == "1234567 + 2222"
    one_space = PackageSession("计算 1+2")
    many_spaces = PackageSession("计算     1+2")
    assert one_space.controller_view == many_spaces.controller_view == "计算 <|PKG_ARITH|>"
    assert many_spaces.unpack("p0001") == "1+2"
    assert PackageSession("8888+6666").controller_view == "<|PKG_ARITH|>"
    date_session = PackageSession("今天是 2026-09-30。")
    assert not date_session.packages
    assert date_session.controller_view == "今天是 2026-09-30。"
    cases = [
        ("请计算 1234567+2222", "CALCULATOR", "1236789"),
        ("计算  15468+9879877989", "CALCULATOR", "9879893457"),
        ("计算 15468+9879877989", "CALCULATOR", "9879893457"),
        ("8888+6666", "CALCULATOR", "15554"),
        ("请对 1234567+2222 做竖式计算", "UNKNOWN", None),
        ("今天是 2026-09-30", "UNKNOWN", None),
        ("请计算 1+2，并计算 3+4", "UNKNOWN", None),
        ("请计算 1*2", "UNKNOWN", None),
        ("请计算 1*2+3", "UNKNOWN", None),
        ("不要计算 1+2", "UNKNOWN", None),
    ]
    for text, expected_route, expected_result in cases:
        _, actual = process_request(text)
        assert actual.route == expected_route, (text, actual)
        assert actual.result == expected_result, (text, actual)
    print("package/router/tool 自检通过：支持路径调用计算器，其余路径均返回 UNKNOWN。")


def check_tokenizer(tokenizer_path: str) -> None:
    """临时注册 package 特殊 token，验证其可成为单个 token ID。"""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, local_files_only=True
    )
    before_size = len(tokenizer)
    before_ids = tokenizer.encode(PACKAGE_MARKER, add_special_tokens=False)
    added = tokenizer.add_special_tokens(
        {"additional_special_tokens": [PACKAGE_MARKER]}
    )
    marker_ids = tokenizer.encode(PACKAGE_MARKER, add_special_tokens=False)
    in_context_ids = tokenizer.encode(
        f"请对{PACKAGE_MARKER}做竖式计算。", add_special_tokens=False
    )
    decoded = tokenizer.decode(marker_ids, skip_special_tokens=False)
    if len(marker_ids) != 1 or decoded != PACKAGE_MARKER:
        raise RuntimeError("package 特殊 token 未能单 token 编码或无法精确解码。")
    print(f"tokenizer: {tokenizer_path}")
    print(f"原始词表长度: {before_size}")
    print(f"未注册标记 token 数: {len(before_ids)}")
    print(f"临时新增 token 数: {added}")
    print(f"扩展后词表长度: {len(tokenizer)}")
    print(f"package token ID: {marker_ids[0]}")
    print(f"带中文上下文 token 数: {len(in_context_ids)}")
    print(f"标记解码往返: {decoded == PACKAGE_MARKER}")
    print("注意：此次扩展只在内存中进行，没有改写 tokenizer 文件。")


def main() -> None:
    """提供 package、路由和计算器专家的命令行入口。"""
    parser = argparse.ArgumentParser(
        description="打包正整数加法式，路由至计算器专家；未命中统一返回 UNKNOWN。"
    )
    parser.add_argument("--text", help="包含一个整数加法表达式的原始文本。")
    parser.add_argument("--self-test", action="store_true", help="运行内置自检。")
    parser.add_argument(
        "--check-tokenizer",
        metavar="PATH",
        help="临时验证 package 标记能否作为单个特殊 token 编码。",
    )
    args = parser.parse_args()
    if args.check_tokenizer:
        check_tokenizer(args.check_tokenizer)
        return
    if args.self_test:
        self_test()
        return
    if not args.text:
        parser.error("请提供 --text，或使用 --self-test / --check-tokenizer。")

    session, result = process_request(args.text)
    print(f"路由：{result.route}")
    print(f"说明：{result.reason}")
    if session is not None:
        print(f"路由器可见输入：{session.controller_view}")
        print(f"路由器可见的 package 类型：{[p.kind for p in session.packages.values()]}")
        print("package 句柄和 payload 保留在运行时表，不发送给路由器。")
    if result.route == "CALCULATOR":
        print(f"专家：整数加法计算器\n结果：{result.result}")


if __name__ == "__main__":
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except AttributeError:
            pass
    main()
