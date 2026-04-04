from config import args
import os
import shutil


def delete_experiment_by_id(base_dirs, target_id):
    """
    删除所有 base_dirs 中编号为 target_id 的文件夹。
    """
    target_id = str(target_id)
    print(f"Start deleting experiment ID: {target_id}...")

    for base_dir in base_dirs:
        target_path = os.path.join(base_dir, target_id)
        if os.path.exists(target_path):
            try:
                shutil.rmtree(target_path)
                print(f" - Deleted: {target_path}")
            except Exception as e:
                print(f" ! Error deleting {target_path}: {e}")
        else:
            print(f" - Not found (skipped): {target_path}")
    print("Deletion complete.\n")


base_dirs_to_clean = [
    f"/data1/liujiaqi/FCGL_results/acc/html/{args.dataset}",
    f"/data1/liujiaqi/FCGL_results/acc/png/{args.dataset}",
    f"/data1/liujiaqi/FCGL_results/aa/html/{args.dataset}",
    f"/data1/liujiaqi/FCGL_results/aa/png/{args.dataset}",
    f"/data1/liujiaqi/FCGL_results/af/html/{args.dataset}",
    f"/data1/liujiaqi/FCGL_results/af/png/{args.dataset}"
]

if __name__ == "__main__":
    target = input(f"请输入要删除的数据集 {args.dataset} 的实验编号 (例如 0): ").strip()

    # 二次确认，防止手滑
    confirm = input(f"警告：这将删除所有目录下编号为 '{target}' 的文件夹及其内容。\n确认删除吗？(y/n): ")

    if confirm.lower() == 'y':
        delete_experiment_by_id(base_dirs_to_clean, target)
    else:
        print("操作已取消。")
