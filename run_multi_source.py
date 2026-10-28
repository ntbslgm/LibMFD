import subprocess
import sys
import itertools
import os
import re
import csv
from datetime import datetime


def extract_best_from_log(log_text):
    """从日志文本中提取最优指标"""
    best_acc = best_f1 = 0.0
    best_acc_epoch = best_f1_epoch = 0
    best_acc_f1 = best_f1_acc = 0.0

    m_acc = re.search(r"Best Accuracy: ([\d\.]+) \(Epoch (\d+), F1=([\d\.]+)\)", log_text)
    m_f1 = re.search(r"Best F1-score: ([\d\.]+) \(Epoch (\d+), Acc=([\d\.]+)\)", log_text)

    if m_acc:
        best_acc, best_acc_epoch, best_acc_f1 = map(float, m_acc.groups())
        best_acc_epoch = int(best_acc_epoch)

    if m_f1:
        best_f1, best_f1_epoch, best_f1_acc = map(float, m_f1.groups())
        best_f1_epoch = int(best_f1_epoch)

    return dict(
        Best_Acc=best_acc,
        Best_Acc_Epoch=best_acc_epoch,
        F1_at_Best_Acc=best_acc_f1,
        Best_F1=best_f1,
        Best_F1_Epoch=best_f1_epoch,
        Acc_at_Best_F1=best_f1_acc
    )


def run_dataset(dataset_name, domains, model_name, train_mode, cuda_device):
    """运行单个数据集的所有任务（支持 single_source / multi_source）"""

    print("\n" + "=" * 80)
    print(f"🚀 开始运行数据集: {dataset_name}")
    print("=" * 80)

    # 为该数据集创建独立的结果文件夹
    dataset_result_dir = f"results/multi_source/{dataset_name}"
    dataset_log_dir = f"logs/multi_source/{dataset_name}"

    os.makedirs(dataset_result_dir, exist_ok=True)
    os.makedirs(dataset_log_dir, exist_ok=True)

    # 生成该数据集的所有任务组合
    tasks = []
    if train_mode == 'multi_source':
        # n-1 源 + 1 目标（leave-one-out）：每个域轮流做 target，其余所有域 join 成源
        for tgt in domains:
            srcs = [d for d in domains if d != tgt]
            tasks.append((','.join(srcs), tgt))   # 如 ("CWRU_1,CWRU_2,CWRU_3", "CWRU_0")
    else:
        # single_source / source_combine: src != tgt 的笛卡尔积
        tasks = [(s, t) for s, t in itertools.product(domains, repeat=2) if s != t]

    # 创建该数据集的结果CSV文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_csv = f"{dataset_result_dir}/{model_name}_{train_mode}_{dataset_name}_results_{timestamp}.csv"

    with open(result_csv, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            "Source", "Target",
            "Best_Acc", "Best_Acc_Epoch", "F1_at_Best_Acc",
            "Best_F1", "Best_F1_Epoch", "Acc_at_Best_F1"
        ])

    total_tasks = len(tasks)
    print(f"📊 数据集 {dataset_name}: 共 {total_tasks} 个任务")

    completed_tasks = 0
    failed_tasks = []

    for idx, (src, tgt) in enumerate(tasks, 1):
        print("-" * 60)
        print(f"[{idx}/{total_tasks}] {src} → {tgt}")
        print("-" * 60)

        # 逗号在文件名里换成下划线，避免歧义
        log_file = f"{dataset_log_dir}/{model_name}_{train_mode}_{src.replace(',', '_')}_to_{tgt}.log"

        cmd = [
            sys.executable, "train.py",
            "--model_name", model_name,
            "--source", src,
            "--target", tgt,
            "--train_mode", train_mode,
            "--cuda_device", str(cuda_device)
        ]

        # ✅ 关键修复：添加 UTF-8 编码和错误处理
        with open(log_file, "w", encoding="utf-8") as f:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8',  # 强制使用 UTF-8 编码
                errors='replace'   # 遇到无法解码的字符用替代字符替换
            )

            # 实时显示输出
            for line in proc.stdout:
                print(line, end="")
                f.write(line)

            proc.wait()

        if proc.returncode != 0:
            print(f"❌ 任务失败: {src} → {tgt}")
            failed_tasks.append(f"{src}→{tgt}")
            continue

        # 提取结果并保存到CSV
        with open(log_file, "r", encoding="utf-8") as f:
            metrics = extract_best_from_log(f.read())

        with open(result_csv, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                src, tgt,
                metrics["Best_Acc"], metrics["Best_Acc_Epoch"], metrics["F1_at_Best_Acc"],
                metrics["Best_F1"], metrics["Best_F1_Epoch"], metrics["Acc_at_Best_F1"]
            ])

        completed_tasks += 1
        print(f"✅ 完成: {src} → {tgt}")
        print(f"   Acc: {metrics['Best_Acc']:.4f} (Epoch {int(metrics['Best_Acc_Epoch'])}), "
              f"F1: {metrics['Best_F1']:.4f} (Epoch {int(metrics['Best_F1_Epoch'])})\n")

    # 打印数据集总结
    print("\n" + "=" * 80)
    print(f"📋 数据集 {dataset_name} 运行完成")
    print(f"✅ 成功: {completed_tasks}/{total_tasks}")
    if failed_tasks:
        print(f"❌ 失败: {failed_tasks}")
    print(f"📄 结果文件: {result_csv}")
    print("=" * 80)

    return completed_tasks, total_tasks, failed_tasks


def main():
    """主函数：运行多个数据集"""

    # ==================== 配置区域 ====================
    model_name = "Mixstyle"
    train_mode = "multi_source"
    cuda_device = 0

    # 定义多个数据集及其域（multi_source 下源和目标来自同一集合）
    datasets = {
        "CWRU": ["CWRU_0", "CWRU_1", "CWRU_2", "CWRU_3"],
        "JNU": ["JNU_0", "JNU_1", "JNU_2"],
        "MFPT": ["MFPT_0", "MFPT_1", "MFPT_2"],
        "OU": ["OU_0", "OU_1", "OU_2", "OU_3"],
        "PU": ["PU_0", "PU_1", "PU_2", "PU_3"],
        "SEBEAR": ["SEBEAR_0", "SEBEAR_1"],
        "SEGEAR": ["SEGEAR_0", "SEGEAR_1"],
        "SQ": ["SQ_0", "SQ_1", "SQ_2"],
        "XJTU": ["XJTU_0", "XJTU_1"],
        "GBN": ["GBN_0", "GBN_1", "GBN_2", "GBN_3", "GBN_4"],
        "GBR": ["GBR_0", "GBR_1", "GBR_2", "GBR_3", "GBR_4"],
    }

    # 选择要运行的数据集（可以只选部分）
    selected_datasets = ["CWRU","JNU","MFPT", "OU","PU", "SQ", "GBN","GBR"]# "CWRU","JNU","MFPT", "OU","PU", "SQ", "GBN","GBR"

    # ==================== 运行区域 ====================

    # 创建总结果文件夹
    os.makedirs("results", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    overall_start_time = datetime.now()

    print("🎯 开始运行多个数据集")
    print(f"📅 开始时间: {overall_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🔧 模型: {model_name}")
    print(f"⚙️  训练模式: {train_mode}")
    print(f"🎮 CUDA设备: {cuda_device}")
    print(f"📚 数据集: {selected_datasets}")

    total_completed = 0
    total_failed = 0
    all_failures = []

    for dataset_name in selected_datasets:
        if dataset_name not in datasets:
            print(f"⚠️  警告: 数据集 {dataset_name} 未定义，跳过")
            continue

        domains = datasets[dataset_name]

        completed, total, failures = run_dataset(
            dataset_name=dataset_name,
            domains=domains,
            model_name=model_name,
            train_mode=train_mode,
            cuda_device=cuda_device
        )

        total_completed += completed
        total_failed += len(failures)
        all_failures.extend([f"{dataset_name}:{f}" for f in failures])

    overall_end_time = datetime.now()
    duration = overall_end_time - overall_start_time

    print("\n" + "=" * 80)
    print("🏁 所有数据集运行完成!")
    print("=" * 80)
    print(f"📅 开始时间: {overall_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📅 结束时间: {overall_end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"⏱️  总耗时: {duration}")
    print(f"✅ 总成功任务: {total_completed}")
    print(f"❌ 总失败任务: {total_failed}")

    if all_failures:
        print("\n❌ 失败任务列表:")
        for failure in all_failures:
            print(f"   - {failure}")

    print("\n📁 结果目录结构:")
    print("   results/")
    for dataset_name in selected_datasets:
        print(f"   ├── {dataset_name}/")
        print(f"   │   └── {dataset_name}_results_*.csv")
    print("   logs/")
    for dataset_name in selected_datasets:
        print(f"   ├── {dataset_name}/")
        print(f"   │   └── *.log")

    print("=" * 80)


if __name__ == "__main__":
    main()
