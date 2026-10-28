# -*- coding: utf-8 -*-
"""
参数敏感性实验脚本（单因子法 / one-factor-at-a-time）。

在 run.py 的基础上扩展：对每个超参数，每次只改变它的取值、其余参数保持默认，
遍历指定的 10 个模型 × 3 个数据集（CWRU / MFPT / XJTU）的全部 source→target 任务。

被考察的参数及其取值（其余参数取 DEFAULT 中的默认值）：
    max_epoch   : 30 / 40 / 50 / 60 / 70
    batch_size  : 16 / 32 / 64 / 128 / 256
    random_state: 10 / 50 / 100 / 500 / 1000
    opt         : sgd / adam
    lr          : 0.001 / 0.005 / 0.01 / 0.02 / 0.05
    backbone    : CNN / ResNet

输出结构：
    results/sensitivity/<param>/<model>_<dataset>_detail.csv   # 长表：每个任务每个取值一行
    results/sensitivity/<param>/_summary.csv                   # 该参数的均值汇总
    results/sensitivity/_master_summary.csv                    # 全部参数的均值汇总
    logs/sensitivity/<param>/<model>_<src>_to_<tgt>__<value>.log

用法（请用安装了 torch 的 pytorch 环境运行）：
    python run_sensitivity.py
"""

import subprocess
import sys
import os
import re
import csv
import itertools
from datetime import datetime

# ==================== 可配置区域 ====================

# 运行 train.py 的 Python 解释器。
# 默认用「当前运行本脚本的解释器」(sys.executable)——只要用 pytorch 环境启动本脚本，
# 它就会自动用同一个 pytorch 环境去调 train.py，跨机器/跨服务器无需再改路径。
PYTHON = os.environ.get("PYTHON", sys.executable)

# 是否把每个任务的完整训练输出实时打印到终端（True 会非常冗长，建议 False）。
VERBOSE = False

# 进度打印：即使 VERBOSE=False，也实时打印「包含这些关键字」的行，
# 让你随时能看到当前训练到第几个 epoch、val 指标如何，而不会被海量日志刷屏。
SHOW_PROGRESS = True
PROGRESS_KEYWORDS = ["Epoch", "Val-Acc", "Val-F1", "Best", "Detect"]

CUDA_DEVICE = 0
TRAIN_MODE = "single_source"

# 默认参数（单因子法中，除被考察的那个参数外，其余参数固定为这里的值）
DEFAULT = {
    "max_epoch": 30,
    "batch_size": 64,
    "random_state": 1,
    "opt": "sgd",
    "lr": 0.01,
    "backbone": "CNN",
}

# 敏感性网格：参数名 -> 取值列表（顺序即运行顺序）
SENSITIVITY_GRID = {
    "max_epoch": [ 40,50, 60, 70],
    "batch_size": [16, 32, 128, 256],
    "random_state": [10, 50, 100, 500, 1000],
    "opt": ["adam"],
    "lr": [0.001, 0.005, 0.02, 0.05],
    "backbone": [ "ResNet"],
}

# 需要考察的模型
MODELS = [
    "ACDANN", "ADACL", "CDAN", "DAN", "DANN",
    "MCD", "CCDG", "DGNIS", "FANDA", "OT_FaultDiagnosis",
]

# 数据集及其域（source / target 同名即可，任务为所有 source != target 的组合）
DATASETS = {
    "CWRU": ["CWRU_0", "CWRU_1", "CWRU_2", "CWRU_3"],
    "MFPT": ["MFPT_0", "MFPT_1", "MFPT_2"],
    "XJTU": ["XJTU_0", "XJTU_1"],
}

# ==================== 运行筛选（None = 跑全部；给列表 = 只跑列出的内容） ====================
# 想只跑某个参数时，把 SELECTED_PARAMS 改成例如 ["max_epoch"]
SELECTED_PARAMS = ["lr"]       # 例：["max_epoch"] 或 ["max_epoch", "lr"]
SELECTED_MODELS = None    # 例：["DAN", "DANN"]
SELECTED_DATASETS = None      # 例：["XJTU"]

# ==================== 工具函数 ====================


def build_tasks(domains):
    """生成 source != target 的所有任务组合。"""
    return [(s, t) for s, t in itertools.product(domains, repeat=2) if s != t]


def value_label(v):
    """把参数取值转成可用于文件名/标签的字符串。"""
    return str(v)


def extract_best_from_log(log_text):
    """
    从日志文本中提取最优指标。

    兼容两类汇总格式：
    1) 标准格式（ACDANN/ADACL/CDAN/DAN/DANN/MCD/CCDG）：
           Best Accuracy: 0.9 (Epoch 3, F1=0.85)
           Best F1-score: 0.85 (Epoch 5, Acc=0.88)
    2) 特殊格式（DGNIS/FANDA/OT_FaultDiagnosis）：
           DGNIS/FANDA/OT-FD Best: Acc=0.9 (Ep3) | F1=0.85 (Ep5)

    返回 dict；对特殊格式无法得到的交叉指标（F1_at_Best_Acc / Acc_at_Best_F1）填 ''。
    """
    m_acc = re.search(r"Best Accuracy: ([\d\.]+) \(Epoch (\d+), F1=([\d\.]+)\)", log_text)
    m_f1 = re.search(r"Best F1-score: ([\d\.]+) \(Epoch (\d+), Acc=([\d\.]+)\)", log_text)

    if m_acc and m_f1:
        best_acc, best_acc_epoch, best_acc_f1 = float(m_acc.group(1)), int(m_acc.group(2)), float(m_acc.group(3))
        best_f1, best_f1_epoch, best_f1_acc = float(m_f1.group(1)), int(m_f1.group(2)), float(m_f1.group(3))
        return dict(Best_Acc=best_acc, Best_Acc_Epoch=best_acc_epoch, F1_at_Best_Acc=best_acc_f1,
                    Best_F1=best_f1, Best_F1_Epoch=best_f1_epoch, Acc_at_Best_F1=best_f1_acc)

    # 特殊格式
    m_special = re.search(r"Best: Acc=([\d\.]+) \(Ep(\d+)\) \| F1=([\d\.]+) \(Ep(\d+)\)", log_text)
    if m_special:
        return dict(Best_Acc=float(m_special.group(1)), Best_Acc_Epoch=int(m_special.group(2)),
                    F1_at_Best_Acc='',
                    Best_F1=float(m_special.group(3)), Best_F1_Epoch=int(m_special.group(4)),
                    Acc_at_Best_F1='')

    return None  # 未匹配到任何汇总信息


def run_single(cmd, log_file):
    """运行单个 train.py 子进程，返回 (returncode, 日志全文)。"""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    with open(log_file, "w", encoding="utf-8") as f:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if VERBOSE:
            for line in proc.stdout:
                print(line, end="")
                f.write(line)
        else:
            for line in proc.stdout:
                f.write(line)
                if SHOW_PROGRESS and any(kw in line for kw in PROGRESS_KEYWORDS):
                    print("    " + line, end="", flush=True)
        proc.wait()
    with open(log_file, "r", encoding="utf-8") as f:
        log_text = f.read()
    return proc.returncode, log_text


def load_done_keys(detail_csv):
    """从已存在的 detail CSV 读取已完成 (value, src, tgt)，用于断点续跑。"""
    done = set()
    if os.path.exists(detail_csv):
        with open(detail_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                done.add((row["Value"], row["Source"], row["Target"]))
    return done


# ==================== 主流程 ====================


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # 应用配置区的筛选（None = 跑全部）
    if SELECTED_PARAMS:
        unknown = [p for p in SELECTED_PARAMS if p not in SENSITIVITY_GRID]
        if unknown:
            print(f"❌ 未知参数: {unknown}，可选: {list(SENSITIVITY_GRID.keys())}")
            return
        grid = {p: SENSITIVITY_GRID[p] for p in SELECTED_PARAMS}
    else:
        grid = SENSITIVITY_GRID

    if SELECTED_MODELS:
        unknown = [m for m in SELECTED_MODELS if m not in MODELS]
        if unknown:
            print(f"❌ 未知模型: {unknown}，可选: {MODELS}")
            return
        models = SELECTED_MODELS
    else:
        models = MODELS

    if SELECTED_DATASETS:
        unknown = [d for d in SELECTED_DATASETS if d not in DATASETS]
        if unknown:
            print(f"❌ 未知数据集: {unknown}，可选: {list(DATASETS.keys())}")
            return
        datasets = {d: DATASETS[d] for d in SELECTED_DATASETS}
    else:
        datasets = DATASETS

    result_root = "results/sensitivity"
    log_root = "logs/sensitivity"
    os.makedirs(result_root, exist_ok=True)
    os.makedirs(log_root, exist_ok=True)

    # 预计算所有任务
    tasks_by_dataset = {d: build_tasks(datasets[d]) for d in datasets}
    total_runs = 0
    for param in grid:
        n = len(grid[param]) * len(models) * sum(len(tasks_by_dataset[d]) for d in datasets)
        total_runs += n

    overall_start = datetime.now()
    print("🎯 参数敏感性实验（单因子法）")
    print(f"📅 开始时间: {overall_start.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🐍 Python: {PYTHON}")
    print(f"🔧 模型: {models}")
    print(f"📚 数据集: {list(datasets.keys())}")
    print(f"⚙️  待考察参数: {list(grid.keys())}")
    print(f"🔢 预计总训练次数: {total_runs}")
    print("=" * 80)

    master_csv = f"{result_root}/_master_summary.csv"
    master_exists = os.path.exists(master_csv)
    master_writer = open(master_csv, "a", newline="", encoding="utf-8")
    master_csv_w = csv.writer(master_writer)
    if not master_exists:
        master_csv_w.writerow(["Param", "Model", "Dataset", "Value", "Avg_Acc", "Avg_F1", "Num_Tasks"])

    total_completed = 0
    total_failed = 0

    for param, values in grid.items():
        print("\n" + "#" * 80)
        print(f"# 参数敏感性: {param}  ->  {values}")
        print("#" * 80)

        # 该参数的汇总缓存
        summary_rows = []  # (model, dataset, value, avg_acc, avg_f1, n)

        for model in models:
            for dataset, domains in datasets.items():
                tasks = tasks_by_dataset[dataset]
                detail_csv = f"{result_root}/{param}/{model}_{dataset}_detail.csv"
                done_keys = load_done_keys(detail_csv)

                # 首次写入表头
                if not os.path.exists(detail_csv):
                    os.makedirs(os.path.dirname(detail_csv), exist_ok=True)
                    with open(detail_csv, "w", newline="", encoding="utf-8") as f:
                        csv.writer(f).writerow([
                            "Model", "Dataset", "Param", "Value", "Source", "Target",
                            "Best_Acc", "Best_Acc_Epoch", "F1_at_Best_Acc",
                            "Best_F1", "Best_F1_Epoch", "Acc_at_Best_F1",
                        ])

                # 汇总每个 value 的指标
                for value in values:
                    vlabel = value_label(value)
                    accs, f1s = [], []

                    for src, tgt in tasks:
                        if (vlabel, src, tgt) in done_keys:
                            continue

                        params = dict(DEFAULT)
                        params[param] = value

                        # backbone 通过 model_name 后缀选择：ResNet -> "<model>_ResNet"
                        effective_model = model
                        if params.get("backbone", "CNN") == "ResNet":
                            effective_model = model + "_ResNet"

                        log_file = (f"{log_root}/{param}/"
                                    f"{model}_{src}_to_{tgt}__{vlabel}.log")
                        cmd = [
                            PYTHON, "train.py",
                            "--model_name", effective_model,
                            "--source", src,
                            "--target", tgt,
                            "--train_mode", TRAIN_MODE,
                            "--cuda_device", str(CUDA_DEVICE),
                            "--max_epoch", str(params["max_epoch"]),
                            "--batch_size", str(params["batch_size"]),
                            "--random_state", str(params["random_state"]),
                            "--opt", str(params["opt"]),
                            "--lr", str(params["lr"]),
                        ]

                        print(f"[{param}={vlabel}] {model} {dataset} {src}->{tgt} ...", flush=True)

                        returncode, log_text = run_single(cmd, log_file)

                        if returncode != 0:
                            print(f"    ❌ 失败 (returncode={returncode})")
                            total_failed += 1
                            with open(detail_csv, "a", newline="", encoding="utf-8") as f:
                                csv.writer(f).writerow([model, dataset, param, vlabel, src, tgt, "", "", "", "", "", ""])
                            continue

                        metrics = extract_best_from_log(log_text)
                        if metrics is None:
                            print(f"    ⚠️  未匹配到汇总指标（见日志 {log_file}）")
                            total_failed += 1
                            with open(detail_csv, "a", newline="", encoding="utf-8") as f:
                                csv.writer(f).writerow([model, dataset, param, vlabel, src, tgt, "", "", "", "", "", ""])
                            continue

                        with open(detail_csv, "a", newline="", encoding="utf-8") as f:
                            csv.writer(f).writerow([
                                model, dataset, param, vlabel, src, tgt,
                                metrics["Best_Acc"], metrics["Best_Acc_Epoch"], metrics["F1_at_Best_Acc"],
                                metrics["Best_F1"], metrics["Best_F1_Epoch"], metrics["Acc_at_Best_F1"],
                            ])

                        accs.append(float(metrics["Best_Acc"]))
                        f1s.append(float(metrics["Best_F1"]))
                        total_completed += 1
                        print(f"    ✅ Acc={metrics['Best_Acc']:.4f} (Ep{metrics['Best_Acc_Epoch']}), "
                              f"F1={metrics['Best_F1']:.4f} (Ep{metrics['Best_F1_Epoch']})")

                    # 汇总当前 (model, dataset, value)
                    if accs:
                        avg_acc = sum(accs) / len(accs)
                        avg_f1 = sum(f1s) / len(f1s)
                    else:
                        avg_acc = avg_f1 = float("nan")
                    summary_rows.append((model, dataset, vlabel, avg_acc, avg_f1, len(accs)))

        # 写该参数的汇总
        summary_csv = f"{result_root}/{param}/_summary.csv"
        with open(summary_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["Model", "Dataset", "Value", "Avg_Acc", "Avg_F1", "Num_Tasks"])
            for model, dataset, vlabel, avg_acc, avg_f1, n in summary_rows:
                w.writerow([model, dataset, vlabel, f"{avg_acc:.4f}", f"{avg_f1:.4f}", n])
                master_csv_w.writerow([param, model, dataset, vlabel, f"{avg_acc:.4f}", f"{avg_f1:.4f}", n])
        master_writer.flush()

        print(f"📄 参数 {param} 完成，汇总: {summary_csv}")

    master_writer.close()

    overall_end = datetime.now()
    print("\n" + "=" * 80)
    print("🏁 全部参数敏感性实验完成!")
    print("=" * 80)
    print(f"⏱️  总耗时: {overall_end - overall_start}")
    print(f"✅ 成功任务: {total_completed}")
    print(f"❌ 失败/未匹配任务: {total_failed}")
    print(f"📁 结果目录: {result_root}/")
    print(f"📁 日志目录: {log_root}/")
    print("=" * 80)


if __name__ == "__main__":
    main()
