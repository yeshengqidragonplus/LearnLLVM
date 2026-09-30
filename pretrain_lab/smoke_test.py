# -*- coding: utf-8 -*-
"""smoke_test.py：跨电脑检查训练、保存、重载和短解码，自动清理临时产物。"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from runtime import add_runtime_args

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description="小规模链路自检，结果不代表数学能力")
    add_runtime_args(parser)
    args = parser.parse_args()
    runtime = ["--device", args.device, "--dtype", args.dtype]

    def run(script, *arguments):
        subprocess.run([sys.executable, str(ROOT / script), *arguments], cwd=ROOT, check=True)

    run("check_env.py", *runtime)
    run("nave.py")
    with tempfile.TemporaryDirectory(prefix="pretrain-smoke-") as temp:
        work = Path(temp)
        data = work / "samples.jsonl"
        data.write_text("".join(json.dumps({"text": f"问: {a}+1 = ?\n答: {a+1}"},
                                          ensure_ascii=False) + "\n" for a in range(12)), encoding="utf-8")
        output = work / "run"
        run("03_train_pretrain.py", "--data", str(data), "--out", str(output),
            "--hidden", "64", "--layers", "1", "--heads", "4", "--no-pack", "--batch", "2",
            "--bucket", "--epochs", "1", "--loss-chunk", "8", "--verify-loss",
            "--no-checkpoints", "--nave", *runtime)
        record = json.loads((output / "run.json").read_text(encoding="utf-8"))
        assert record["status"] == "completed" and len(record["epochs"]) == 1
        run("04e_eval_v5.py", "--model", str(output / "final_model"), "--nave",
            "--n", "1", "--max-new-tokens", "4", *runtime)
    print("链路自检通过，临时产物已清理。短解码分数不能作为能力评估。")


if __name__ == "__main__":
    main()
