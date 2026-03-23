import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.signal import savgol_filter  # 用于平滑曲线


def plot_original_accuracy_plotly(self, accuracies, save_path):
    """
    使用 Plotly 绘制原始准确率曲线（只保留点，去掉连线），并保存为 HTML 文件。

    Args:
        accuracies (list): 每轮的测试集准确率列表。
        save_path (str): HTML 文件的保存路径。
    """
    fig = go.Figure()

    # 添加原始准确率曲线（只保留点，去掉连线）
    fig.add_trace(go.Scatter(
        x=list(range(len(accuracies))),
        y=accuracies,
        mode='markers',  # 只显示点
        name='Accuracy',
        marker=dict(color='blue', size=8)  # 设置点的样式
    ))

    # 设置布局
    fig.update_layout(
        title="Federated Learning Test Accuracy Curve",
        xaxis_title="Communication Round",
        yaxis_title="Test Accuracy",
        template="plotly_white"
    )

    # 保存为 HTML 文件
    fig.write_html(save_path)
    print(f"Original accuracy curve (HTML) saved to {save_path}")


def plot_original_accuracy_matplotlib(self, accuracies, save_path):
    """
    使用 Matplotlib 绘制原始准确率曲线（只保留点，去掉连线），并保存为图片文件。

    Args:
        accuracies (list): 每轮的测试集准确率列表。
        save_path (str): 图片文件的保存路径。
    """
    plt.figure(figsize=(20, 10))  # 调整图像大小

    # 绘制原始准确率曲线（只保留点，去掉连线）
    plt.scatter(range(len(accuracies)), accuracies,
                color='blue', label='Accuracy', s=50)  # s 控制点的大小

    # 设置横轴坐标和坐标格
    x_ticks = list(range(0, len(accuracies), len(accuracies) // 5))  # 6 个坐标点
    plt.xticks(x_ticks)  # 设置横轴坐标
    plt.grid(True, which='both', axis='x',
             linestyle='--', linewidth=0.5)  # 设置坐标格

    # 设置标题和标签
    plt.xlabel("Communication Round")
    plt.ylabel("Test Accuracy")
    plt.title("Federated Learning Test Accuracy Curve")
    plt.legend()

    # 保存为图片文件
    plt.savefig(save_path)
    print(f"Original accuracy curve (image) saved to {save_path}")

    plt.close()  # 关闭图像，释放内存


def plot_smoothed_accuracy_plotly(accuracies, save_path, window_len=201):
    """
    使用 Plotly 绘制拟合图像（保留原始点并添加平滑曲线），并保存为 HTML 文件。

    Args:
        accuracies (list): 每轮的测试集准确率列表。
        save_path (str): HTML 文件的保存路径。
    """
    if len(accuracies) > 1:  # 确保有足够的数据点进行平滑
        # 使用 Savitzky-Golay 滤波器平滑曲线
        window_length = min(len(accuracies), window_len)  # 窗口长度（奇数）
        if window_length > 2:  # 确保窗口长度足够
            polyorder = 3  # 多项式阶数
            smoothed_accuracies = savgol_filter(
                accuracies, window_length, polyorder)

            # 使用 Plotly 绘制拟合图像
            fig = go.Figure()

            # 添加原始点（不连线）
            fig.add_trace(go.Scatter(
                x=list(range(len(accuracies))),
                y=accuracies,
                mode='markers',  # 只显示点
                name='Accuracy Points',
                marker=dict(color='blue', size=8)  # 设置点的样式
            ))

            # 添加平滑曲线
            fig.add_trace(go.Scatter(
                x=list(range(len(smoothed_accuracies))),
                y=smoothed_accuracies,
                mode='lines',  # 只显示线
                name='Smoothed Accuracy',
                line=dict(color='red', width=2)  # 设置线的样式
            ))

            # 设置布局
            fig.update_layout(
                title="Federated Learning Test Accuracy Curve (Smoothed)",
                xaxis_title="Communication Round",
                yaxis_title="Test Accuracy",
                template="plotly_white"
            )

            # 保存为 HTML 文件
            fig.write_html(save_path)
            print(f"Smoothed accuracy curve (HTML) saved to {save_path}")
        else:
            print("Not enough data points for smoothing.")
    else:
        print("Not enough data points for smoothing.")


def plot_smoothed_accuracy_matplotlib(accuracies, save_path, window_len=201):
    """
    使用 Matplotlib 绘制拟合图像（保留原始点并添加平滑曲线），并保存为图片文件。

    Args:
        accuracies (list): 每轮的测试集准确率列表。
        save_path (str): 图片文件的保存路径。
    """
    if len(accuracies) > 1:  # 确保有足够的数据点进行平滑
        # 使用 Savitzky-Golay 滤波器平滑曲线
        window_length = min(len(accuracies), window_len)  # 窗口长度（奇数）
        if window_length > 2:  # 确保窗口长度足够
            polyorder = 3  # 多项式阶数
            smoothed_accuracies = savgol_filter(
                accuracies, window_length, polyorder)

            # 使用 Matplotlib 绘制拟合图像
            plt.figure(figsize=(20, 10))  # 调整图像大小

            # 绘制原始点（不连线）
            plt.scatter(range(len(accuracies)), accuracies,
                        color='blue', label='Accuracy Points', s=50)

            # 绘制平滑曲线
            plt.plot(range(len(smoothed_accuracies)), smoothed_accuracies,
                     linestyle='-', color='red', label='Smoothed Accuracy', linewidth=2)

            # 设置横轴坐标和坐标格
            x_ticks = list(
                range(0, len(accuracies), len(accuracies) // 5))  # 6 个坐标点
            plt.xticks(x_ticks)  # 设置横轴坐标
            plt.grid(True, which='both', axis='x',
                     linestyle='--', linewidth=0.5)  # 设置坐标格

            # 设置标题和标签
            plt.xlabel("Communication Round")
            plt.ylabel("Test Accuracy")
            plt.title("Federated Learning Test Accuracy Curve (Smoothed)")
            plt.legend()

            # 保存为图片文件
            plt.savefig(save_path)
            print(f"Smoothed accuracy curve (image) saved to {save_path}")

            plt.close()  # 关闭图像，释放内存
        else:
            print("Not enough data points for smoothing.")
    else:
        print("Not enough data points for smoothing.")
