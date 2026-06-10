from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
import datetime

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from faers_sglt2_dka.preprocess import build_analysis_datasets


def main():
    parser = argparse.ArgumentParser(description="Build deduplicated all-case dataset and target-drug model dataset.")
    parser.add_argument("--config", default="config/terms.yml")
    parser.add_argument("--interim-dir", default="data/interim")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--target-terms-key", default="target_event_terms",
                        choices=["target_event_terms", "target_event_terms_extended"],
                        help="Which PT term list to use (default: core terms)")
    args = parser.parse_args()

    print("=" * 60)
    print("FAERS SGLT2-DKA 数据集构建")
    print("=" * 60)
    print(f"开始时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"配置文件: {args.config}")
    print(f"中间数据目录: {args.interim_dir}")
    print(f"处理数据目录: {args.processed_dir}")
    print(f"目标术语: {args.target_terms_key}")
    print("-" * 60)

    start_time = time.time()

    try:
        all_cases, model_dataset, screening = build_analysis_datasets(
            args.interim_dir, args.processed_dir, args.config, target_terms_key=args.target_terms_key
        )

        print("\n" + "=" * 60)
        print("数据集构建完成!")
        print("=" * 60)
        print(f"all_cases: {all_cases.shape}")
        print(f"model_dataset: {model_dataset.shape}")
        print("\n=== DKA 标签分布 ===")
        print(model_dataset["label_target_event"].value_counts(dropna=False))
        print("\n=== 筛选统计 (用于流程图) ===")
        for k, v in screening.items():
            print(f"  {k}: {v:,}")

        # 显示文件保存位置
        processed_dir = Path(args.processed_dir)
        print("\n=== 文件保存位置 ===")
        print(f"  all_cases.parquet: {processed_dir / 'all_cases.parquet'}")
        print(f"  model_dataset.parquet: {processed_dir / 'model_dataset.parquet'}")
        print(f"  drug_features.parquet: {processed_dir / 'drug_features.parquet'}")
        print(f"  indication_features.parquet: {processed_dir / 'indication_features.parquet'}")
        print(f"  outcome_features.parquet: {processed_dir / 'outcome_features.parquet'}")
        print(f"  reporter_features.parquet: {processed_dir / 'reporter_features.parquet'}")
        print(f"  screening_counts.json: {processed_dir / 'screening_counts.json'}")

        elapsed_time = time.time() - start_time
        print(f"\n总耗时: {elapsed_time:.2f} 秒 ({elapsed_time/60:.2f} 分钟)")
        print(f"完成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)

    except Exception as e:
        print(f"\n错误: {str(e)}")
        print("数据集构建失败，请检查错误信息并重新运行。")
        sys.exit(1)


if __name__ == "__main__":
    main()