# 编程专家分数演变：12/60 → 58/60（2026-10-21）

同一份声明数据、同一基座（`output_v5c1`，65.32M，hidden 384/4 层/8 头）、
同样 `--target-mask --no-pack` lr 2e-4 40 epoch 1840 步，唯一差别是 **target 是否打包**。

| 方向 | 打包前 `output_decl_expert_v1` | 打包后 `output_decl_packed_v1` |
|---|---|---|
| 代码→中文注释 | 0/24 | **24/24 严格** |
| 中文注释→代码 | 0/24 | **22/24 严格，24/24 部分** |
| 分号语义 | 12/12 | 12/12 |
| 合计 | **12/60 = 20%** | **58/60 严格 = 96.7%，60/60 部分 = 100%** |

## 结论：target 打包（token filter）就是复制墙的解法
模型输出的是

    声明一个无符号整型静态常量 <|PKG_NAME|>，初始值是 <|PKG_NUM|>（无符号字面量）。static const 表示程序运行期间保持不变的静态常量。
    static const unsigned int <|PKG_NAME|> = <|PKG_NUM|>;

零字符复制；程序解包替换回载荷即得明文。**65M 完全够用，之前不是容量问题。**

## 同时修掉的评估假失败
`04f` 原来用 `tokenizer.decode(..., skip_special_tokens=True)`，会把
`<|PKG_NAME|>`/`<|PKG_NUM|>` 当 special token 直接删掉，模型明明输出正确标记，
评估器却只看到「声明一个整型变量 ，初始值是 」→ 判 0。
现在改成 `skip_special_tokens=False`，只清 eos/pad，标记保留到解包步骤再替换。

**写任何涉及包标记的评估脚本都必须注意这一点**，否则会制造"学会了但判 0"的假失败。

## 剩余 2 条严格失败
`const int AVOGADRO = 598459;` → 输出 `const double <|PKG_NAME|> = <|PKG_NUM|>;`：
标记位置全对，只有类型词 `int` 判成 `double`。属分类错误，扩类型×形态覆盖即可，与复制无关。
