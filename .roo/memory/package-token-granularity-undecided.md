# 待决：包 token 的粒度（per-key 还是 per-type）—— 2026-10-23 用户提出后未定

**状态：未决。用户说「让我想想」。在此之前的 v8 结论中，有两条建立在下述
per-key 假设上，因此暂不可引用。**

## 用户指出的机制（这条是确定的）

> 对于 embedding 相同类型的 Package 他们的 token 是一样的，等展开以后再重新展开内容
> 算 embedding，所以我说是嵌套循环式的，直到有了最终结果。展开是一个替换过程不走大模型。

拆成三条：

1. **同类型 = 同一 token = 同一行 embedding**。投影里出现两个 `<|PKG_NUM|>`，
   模型看到的是同一个 token 两次，它不区分「第一个 / 第二个」。
2. **展开 = 替换**，由程序做，**不走大模型**；替换出的内容若仍含包标记，就再展开、
   再重算 embedding → 嵌套循环，直到没有标记、得到最终结果。
3. 所以模型要做的事只是「知道这里有一个数字包」，不需要数位置、不需要认序号。

## 与现有实现的冲突（这是我之前的错误）

`tokens.json` 的 `_package_keys` 把包做成 **156 个独立 token**
（`<|PKG_NUM|,#A>` … `<|PKG_TASK|,#Z>`，ID 151660..151815），每个 key 一行独立 embedding。
这与上面第 1 条冲突。

而更早的 [`code_package_frontend_v1.py`](../../pretrain_lab/code_package_frontend_v1.py)
原本是对的：句柄用 `uuid.uuid4().hex`，**模型只看 `marker`，句柄只在程序侧**。
后来加 156 个 token 才是偏离。

## 因此下面两条结论暂不可引用（前提是 per-key token）

- **「`type_id` 类型辨识 0.97/0.97/0.94 成立」** —— 题目是「`<|PKG_NUM|,#Z>` 是什么类型」。
  若同类型共用一个 token，模型看到的 token 本身就**是**类型，这道题不成立。
- **「未训练 key 零样本为零、是原子整体的固有代价」** —— 0/60 是「156 个独立 embedding
  且该行未训练」的必然结果。若只有 6 个类型 token，就**不存在**未训练的 key，
  这个「代价」在用户的设计里不存在。之前把它定性为「原子整体的固有代价」是错的。

仍然成立、不依赖 per-key 假设的：`semantic_pick` 0.96、`distance` 800 token 全对
（包表里只有一个 NUM 时 `<|PKG_NUM|>` 本身无歧义，靠的正是「token 即类型」）。

## 用户尚未决定的分叉

1. 词表只留 6 个类型标记，`#A/#B` 只在缓存/日志里做地址、**不进模型输入**；
2. 投影里仍写 `#A` 让模型看见顺序，但 `#A` 用已有 token 拼、不是独立注册 token。

## 一旦定了，要改的落点

- `tokens.json`：`_package_keys` / `reserved` / `next_free_id_after_keys`
- `03_train_pretrain.py`：`--package-keys` 注册块（156 个 key 的单 token 断言）
- `22/23`（v7）、`24/25`（v8）：`KEY_RE`、`make_key`、判分、审计断言
- `README` 3.10、`data/pronoun_probe_v8_report.json`、`atomic-key-zero-shot-is-zero.md`
