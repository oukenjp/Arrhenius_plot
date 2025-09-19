import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
#created by j.wang https://github.com/oukenjp/Arrhenius_plot
# ================== 参数配置 ==================
files = [
    {"filename": "1", "col_x": 0, "col_y": 1, "label": "undoped", "color": None, "marker": None, "linestyle": None,},
    {"filename": "2", "col_x": 0, "col_y": 1, "label": r"5% doped", "color": None, "marker": None, "linestyle": None,},
    {"filename": "3", "col_x": 0, "col_y": 1, "label": r"10% doped", "color": None, "marker": None, "linestyle": None,},
]

# 绘图参数设置
font_title = {'family': 'serif', 'weight': 'bold', 'size': 22}
font_label = {'family': 'sans-serif', 'size': 16}
tick_config = {
    "major":{"which":"major","length":6,"width":1,"direction":"out","labelsize":14},
    "minor":{"which":"minor","length":4,"width":1,"direction":"out"}
    }
line_width = 2
point_size = 50
legend_fontsize = 12
figsize = (8,8)
default_linestyle = "-"
main_title = ''
outfile = "arrhenius.svg"
use_fit = True  # 是否进行拟合
show_Ea = True  # 是否在图上显示Ea值

# 颜色池（自动分配）
cmap = plt.get_cmap('tab10', 10)
auto_colors = [cmap(i) for i in range(cmap.N)]
#auto_colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'cyan', 'magenta']
auto_markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'x', '+']

# === 坐标轴的刻度配置 ===
# 定义你希望显示的特定温度值
specific_temperatures = [30]
# 定义你希望显示的固定步长温度范围
range_temperatures = np.arange(-100, 500, 50)
# 将两个数组合并，得到最终的刻度列表
desired_temperatures_celsius = np.concatenate((specific_temperatures, range_temperatures))
# 希望显示的grid线位置（对应下轴）
grid_ticks = 1000 / (np.array(desired_temperatures_celsius) + 273.15)
# =============================================

def read_data(filename, comment_char="#", n_lines=10):
    """
    自动读取数据文件，支持：
    - 注释行自动跳过
    - 表头自动识别
    - 自动识别常见分隔符（空格、多空格、制表符、逗号等）
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"文件不存在: {filename}")

    # 预读前几行，兼容 \r\n 和 \n
    sample_lines = []
    with open(filename, 'r', newline='') as f:
        for _ in range(n_lines):
            line = f.readline()
            if not line:
                break
            sample_lines.append(line.rstrip("\r\n"))

    sample = "\n".join(sample_lines)

    # 统计注释行
    comment_lines = [line for line in sample_lines if line.strip().startswith(comment_char)]
    n_comment_lines = len(comment_lines)

    # 尝试使用 Sniffer 猜测分隔符
    use_whitespace = False
    try:
        dialect = csv.Sniffer().sniff(sample)
        delimiter = dialect.delimiter
        # 如果 Sniffer 猜出单空格，但行中有多个连续空格，改用正则匹配空白
        if delimiter == ' ' and any('  ' in line for line in sample_lines):
            delimiter = r'\s+'
            use_whitespace = True
    except csv.Error:
        # Sniffer 失败 → 回退到任意空白
        delimiter = r'\s+'
        use_whitespace = True

    # 判断首个非注释行是否为表头
    header_line_index = n_comment_lines
    if header_line_index >= len(sample_lines):
        header_option = None
    else:
        line_to_check = sample_lines[header_line_index]
        if use_whitespace:
            header_tokens = line_to_check.split()
        else:
            header_tokens = line_to_check.split(delimiter)

        def is_number(x):
            try:
                float(x)
                return True
            except ValueError:
                return False

        if all(not is_number(tok) for tok in header_tokens):
            header_option = 0
        else:
            header_option = None

    # 读取数据
    if use_whitespace:
        df = pd.read_csv(
            filename,
            sep=r'\s+',
            engine='python',
            comment=comment_char,
            header=header_option
        )
    else:
        df = pd.read_csv(
            filename,
            sep=delimiter,
            engine='python',
            comment=comment_char,
            header=header_option
        )

    return df

def auto_scan_files(extensions=['.xy', '.csv', '.txt', '.dat','']):
    """
    自动扫描当前文件夹中的数据文件，并生成默认配置。
    """
    print("未检测到文件输入，正在自动扫描当前文件夹...")
    
    file_list = []
    # 获取当前脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    for filename in os.listdir(script_dir):
        # 排除隐藏文件和文件夹
        if filename.startswith('.'):
            continue
        
        # 检查文件扩展名
        _, ext = os.path.splitext(filename)
        if ext.lower() in extensions:
            file_list.append({
                "filename": filename,
                "col_x": 0,
                "col_y": 1,
                "label": os.path.splitext(filename)[0],
                "color": None,
                "marker": None,
                "linestyle": None,
                "use_fit": True
            })
    
    if not file_list:
        print("未找到任何符合条件的数据文件。")
    else:
        print(f"找到以下文件: {[f['filename'] for f in file_list]}")

    return file_list

# ==================== 功能实现 ====================
# 如果files列表为空，则进行自动扫描
if not files:
    files = auto_scan_files()
    if not files:
        exit() # 如果没有找到文件，退出脚本


fig = plt.figure(figsize=figsize)
#ax = plt.gca()
ax = fig.add_axes([0.11, 0.11, 0.78, 0.78]) 
handles = []
labels = []

# ==================== 创建上 X 轴 ====================
def bottom_to_top(x):
    return 1000/x - 273.15

def top_to_bottom(T_celsius):
    return 1000/(T_celsius + 273.15)

secax = ax.secondary_xaxis('top', functions=(bottom_to_top, top_to_bottom))
secax.set_xlabel("Temperature (°C)", fontdict=font_label)

# 设置上轴想显示的刻度
secax.set_xticks(desired_temperatures_celsius)
secax.set_xticklabels([f"{t}" for t in desired_temperatures_celsius])

# 保留你原来的刻度参数
secax.tick_params(**tick_config['major'])
secax.tick_params(**tick_config['minor'])

# 下轴刻度依然用原来的自定义
ax.tick_params(**tick_config['major'])
ax.tick_params(**tick_config['minor'])
# =============================================================

for idx, f in enumerate(files):
    try:
        data = read_data(f['filename'])
    except FileNotFoundError as e:
        print(e)
        continue

    if f['col_x'] >= data.shape[1] or f['col_y'] >= data.shape[1]:
        print(f"警告: 文件 {f['filename']} 的列索引超出范围，跳过绘图")
        continue

    a = data.iloc[:, f['col_x']].values  # 摄氏度
    b = data.iloc[:, f['col_y']].values  # 电导率

    # 下 X 轴: 1000/(T[K])
    T_K = a + 273.15
    x_bottom = 1000 / T_K

    color = f["color"] if f["color"] else auto_colors[idx % len(auto_colors)]
    linestyle = f["linestyle"] if f["linestyle"] else default_linestyle
    data_label = f["label"] if f["label"] else os.path.basename(f["filename"])
    marker = f["marker"] if f["marker"] else auto_markers[idx % len(auto_markers)]

    if use_fit==True:
        # 拟合模式
        y_fit = np.log(b * T_K)
        coeffs = np.polyfit(x_bottom, y_fit, 1)
        y_line = coeffs[0]*x_bottom + coeffs[1]
        
        # 绘制拟合线和数据点
        line, = ax.plot(x_bottom, y_line, linestyle='--', color=color, label=data_label + ' Fit')
        scatter = ax.scatter(x_bottom, y_fit, color=color, 
                   marker=marker, s=point_size, label=data_label)
        
        # 收集handles和labels
        handles.append(scatter)
        labels.append(data_label)
        # handles.append(line)
        # labels.append(data_label + ' Fit')
        
        ax.set_ylabel(r'ln($\sigma$T[S K])', fontdict=font_label)
        
        slope, intercept = coeffs
        print(f"文件: {f['filename']} Ea = {-0.08617343*slope:.4f}, ln(σ0) = {intercept:.4f}")

        if show_Ea==True:
            # 末点坐标（对应绘图坐标系）
            x_end = x_bottom[0]
            y_end = y_fit[0]

            dx = (max(x_bottom) - min(x_bottom)) * 0.28
            dy = (max(y_fit)-min(y_fit)) * 0.1

            x0, x1 = min(x_bottom), max(x_bottom)
            y0, y1 = np.polyval(coeffs, [x0, x1])
            x0_disp, y0_disp = ax.transData.transform((x0, y0))
            x1_disp, y1_disp = ax.transData.transform((x1, y1))

            plt.annotate(rf"$E_\mathrm{{a}}$ = {-0.08617*slope:.3f} eV",
                        xy=(x_end, y_end),
                        xytext=(x_end-dx, y_end+dy),
                        textcoords='data',
                        fontsize=16,
                        color=color,
                        ha='left', va='center',
                        rotation=np.degrees(np.arctan2(y1_disp - y0_disp, x1_disp - x0_disp)))

    else:
        # 正常绘图模式
        line, = ax.plot(x_bottom, b, linestyle=linestyle, color=color, 
                marker=marker, markersize=8, label=data_label)
        # 收集handles和labels
        handles.append(line)
        labels.append(data_label)
        
        ax.set_yscale('log')
        ax.set_ylabel(r'$\sigma$ (S/cm)', fontdict=font_label)

    # 下 X 轴label
    ax.set_xlabel(r'$1000/(T[K])$', fontdict=font_label)

# 设置labels
plt.title(main_title, fontdict=font_title)
# ax.legend(loc="best", fontsize=legend_fontsize)
plt.legend(handles=handles, labels=labels, loc="best", fontsize=legend_fontsize)

# 绘制竖直网格线
x_min, x_max = ax.get_xlim()
x_in_range = [x for x in grid_ticks if x_min <= x <= x_max]
for x in x_in_range:
    ax.axvline(x, linestyle='--', color='gray', alpha=0.5)

# plt.show()
plt.savefig(outfile)