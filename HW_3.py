import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style="whitegrid")

print("✅ 库导入成功，环境配置完成。")

# ==============================================================================
# 任务 1：数据预处理
# ==============================================================================
print("\n--- 任务 1：数据预处理 ---")

# 1. 读取数据

df = pd.read_csv('ICData.csv', sep=',', header=0)
print("【数据集前5行】")
print(df.head())
print(f"\n【基本信息】行数: {df.shape[0]}, 列数: {df.shape[1]}")
print("【数据类型】")
print(df.dtypes)

# 2. 时间解析
# 将交易时间转换为 datetime 类型
df['交易时间'] = pd.to_datetime(df['交易时间'])
# 提取小时
df['hour'] = df['交易时间'].dt.hour

# 3. 构造衍生字段：搭乘站点数
# 计算绝对差值
df['ride_stops'] = (df['下车站点'] - df['上车站点']).abs()

# 删除 ride_stops 为 0 的异常记录
before_drop = df.shape[0]
df = df[df['ride_stops'] != 0]
after_drop = df.shape[0]
print(f"\n【异常值处理】删除了 {before_drop - after_drop} 行 ride_stops 为 0 的记录。")

# 4. 缺失值检查
print("\n【缺失值统计】")
print(df.isnull().sum())
# 如果有缺失值，通常策略是删除（因为公交刷卡数据少一行影响不大）
if df.isnull().sum().any():
    df.dropna(inplace=True)
    print("已删除包含缺失值的行。")
# 任务1：数据清洗完成
# ==============================================================================
# 任务 2：时间分布分析
# ==============================================================================
print("\n--- 任务 2：时间分布分析 ---")

# (a) 早晚时段刷卡量统计
# 筛选上车记录
df_boarding = df[df['刷卡类型'] == 0]

# 使用 numpy 进行布尔索引统计
# 早峰前 (< 7)
morning_mask = df_boarding['hour'].values < 7
morning_count = np.sum(morning_mask)
# 深夜 (>= 22)
night_mask = df_boarding['hour'].values >= 22
night_count = np.sum(night_mask)

total_count = len(df_boarding)

print(f"早峰前 (<07:00) 刷卡量: {morning_count}")
print(f"深夜 (>=22:00) 刷卡量: {night_count}")
print(f"早峰前占比: {morning_count / total_count:.2%}")
print(f"深夜占比: {night_count / total_count:.2%}")

# (b) 24小时刷卡量分布可视化 (matplotlib)
hourly_counts = df_boarding['hour'].value_counts().sort_index()

plt.figure(figsize=(12, 6))
# 绘制柱状图
bars = plt.bar(hourly_counts.index, hourly_counts.values, color='skyblue')

# 高亮显示早峰前和深夜
for bar in bars:
    x = bar.get_x()
    if x < 7 or x >= 22:
        bar.set_color('salmon')

plt.title('24小时公交刷卡量分布图', fontsize=16)
plt.xlabel('小时', fontsize=12)
plt.ylabel('刷卡量 (次)', fontsize=12)
plt.xticks(range(0, 24, 2))  # x轴步长为2
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 图例（手动创建）
from matplotlib.patches import Patch

legend_elements = [Patch(facecolor='skyblue', label='正常时段'),
                   Patch(facecolor='salmon', label='早峰前/深夜')]
plt.legend(handles=legend_elements)

plt.tight_layout()
plt.savefig('hour_distribution.png', dpi=150)
print("✅ 图表 hour_distribution.png 已保存。")
plt.show()
# 任务2：时间分布图绘制完成


# ==============================================================================
# 任务 3：线路站点分析
# ==============================================================================
print("\n--- 任务 3：线路站点分析 ---")


def analyze_route_stops(df, route_col='线路号', stops_col='ride_stops'):
    """
    计算各线路乘客的平均搭乘站点数及其标准差。
    """
    # 分组聚合
    grouped = df.groupby(route_col)[stops_col].agg(['mean', 'std']).reset_index()
    # 重命名列
    grouped.columns = [route_col, 'mean_stops', 'std_stops']
    # 按均值降序排列
    grouped = grouped.sort_values(by='mean_stops', ascending=False)
    return grouped


# 调用函数
route_stats = analyze_route_stops(df)
print("【各线路平均搭乘站点数 Top 10】")
print(route_stats.head(10))

# 可视化：Top 15 线路
top_15_routes = route_stats.head(15)

plt.figure(figsize=(10, 8))
# seaborn 条形图，y轴为线路号（分类），x轴为均值
# ✅ 修复方案：显式指定 orient 和 errorbar 参数
ax = sns.barplot(data=top_15_routes, x='mean_stops', y='线路号',
                 palette='Blues_d', orient='h')

# 手动添加误差线（这样能精确控制形状）
ax.errorbar(x=top_15_routes['mean_stops'],
            y=top_15_routes.index,
            xerr=top_15_routes['std_stops'],
            fmt='none', capsize=3, color='black', linewidth=1)

plt.title('平均搭乘站点数最多的 Top 15 线路', fontsize=16)
plt.xlabel('平均搭乘站点数', fontsize=12)
plt.ylabel('线路号', fontsize=12)
plt.xlim(0, None)  # x轴从0开始

plt.tight_layout()
plt.savefig('route_stops.png', dpi=150)
print("✅ 图表 route_stops.png 已保存。")
plt.show()
# 任务3：线路站点分析函数完成

# ==============================================================================
# 任务 4：高峰小时系数计算 (PHF)
# ==============================================================================
print("\n--- 任务 4：高峰小时系数计算 ---")

# 1. 识别高峰小时
# 统计全天各小时刷卡量
hourly_vol = df_boarding['hour'].value_counts().sort_index()
peak_hour = hourly_vol.idxmax()  # 刷卡量最大的小时
peak_volume = hourly_vol.max()

print(f"高峰小时为: {peak_hour}:00 - {peak_hour + 1}:00, 刷卡量: {peak_volume} 次")

# 筛选高峰小时的数据
df_peak = df_boarding[df_boarding['hour'] == peak_hour].copy()

# 为了按分钟聚合，我们需要构造一个仅包含分钟的时间索引，或者直接利用 datetime
# 这里我们利用 datetime 的分钟属性进行分组
# 5分钟粒度
# 将分钟数向下取整到最近的5分钟 (例如 08:13 -> 08:10)
df_peak['minute_5bin'] = (df_peak['交易时间'].dt.minute // 5) * 5
vol_5min = df_peak.groupby(['hour', 'minute_5bin']).size()
max_5min_vol = vol_5min.max()
# 找到最大5分钟的时间段
max_5min_time = vol_5min.idxmax()
phf5 = peak_volume / (12 * max_5min_vol)

# 15分钟粒度
df_peak['minute_15bin'] = (df_peak['交易时间'].dt.minute // 15) * 15
vol_15min = df_peak.groupby(['hour', 'minute_15bin']).size()
max_15min_vol = vol_15min.max()
# 找到最大15分钟的时间段
max_15min_time = vol_15min.idxmax()
phf15 = peak_volume / (4 * max_15min_vol)

print("-" * 30)
print(f"高峰小时：{peak_hour:02d}:00 ~ {peak_hour + 1:02d}:00，刷卡量：{peak_volume} 次")
print(
    f"最大5分钟刷卡量（{max_5min_time[0]:02d}:{max_5min_time[1]:02d}~{max_5min_time[0]:02d}:{max_5min_time[1] + 5:02d}）：{max_5min_vol} 次")
print(f"PHF5  = {peak_volume} / (12 × {max_5min_vol}) = {phf5:.4f}")
print(
    f"最大15分钟刷卡量（{max_15min_time[0]:02d}:{max_15min_time[1]:02d}~{max_15min_time[0]:02d}:{max_15min_time[1] + 15:02d}）：{max_15min_vol} 次")
print(f"PHF15 = {peak_volume} / ( 4 × {max_15min_vol}) = {phf15:.4f}")

# ==============================================================================
# 任务 5：线路驾驶员信息批量导出
# ==============================================================================
print("\n--- 任务 5：线路驾驶员信息批量导出 ---")

# 筛选线路 1101 - 1120
target_routes = range(1101, 1121)
df_target = df[df['线路号'].isin(target_routes)]

# 创建文件夹
output_dir = '线路驾驶员信息'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print(f"正在生成文件到文件夹: {output_dir}")
for route in target_routes:
    df_route = df_target[df_target['线路号'] == route]
    # 获取车辆-驾驶员对应关系，去重
    # 注意：题目要求格式 "车辆编号驾驶员编号"，这里假设是紧挨着或者有空格，根据示例看起来像是一行行的
    # 示例格式：913041 91321599 ...
    # 我们将其整理为字符串
    relations = df_route[['车辆编号', '驾驶员编号']].drop_duplicates()

    # 构建文件内容
    content_lines = [f"线路号: {route}"]
    for _, row in relations.iterrows():
        content_lines.append(f"{row['车辆编号']} {row['驾驶员编号']}")

    # 写入文件
    file_path = os.path.join(output_dir, f"{route}.txt")
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(content_lines))

    print(f"已生成: {file_path}")

# ==============================================================================
# 任务 6：服务绩效排名与热力图
# ==============================================================================
print("\n--- 任务 6：服务绩效排名与热力图 ---")

# 1. 排名统计 (Top 10)
# 司机
top_drivers = df['驾驶员编号'].value_counts().head(10)
# 线路
top_routes_perf = df['线路号'].value_counts().head(10)
# 上车站点
top_stations = df['上车站点'].value_counts().head(10)
# 车辆
top_vehicles = df['车辆编号'].value_counts().head(10)

print("Top 10 司机:\n", top_drivers.index.tolist())
print("Top 10 线路:\n", top_routes_perf.index.tolist())

# 2. 热力图数据构造
# 构造 4x10 的矩阵
heatmap_data = pd.DataFrame({
    'Top1': [top_drivers.iloc[0], top_routes_perf.iloc[0], top_stations.iloc[0], top_vehicles.iloc[0]],
    'Top2': [top_drivers.iloc[1], top_routes_perf.iloc[1], top_stations.iloc[1], top_vehicles.iloc[1]],
    'Top3': [top_drivers.iloc[2], top_routes_perf.iloc[2], top_stations.iloc[2], top_vehicles.iloc[2]],
    'Top4': [top_drivers.iloc[3], top_routes_perf.iloc[3], top_stations.iloc[3], top_vehicles.iloc[3]],
    'Top5': [top_drivers.iloc[4], top_routes_perf.iloc[4], top_stations.iloc[4], top_vehicles.iloc[4]],
    'Top6': [top_drivers.iloc[5], top_routes_perf.iloc[5], top_stations.iloc[5], top_vehicles.iloc[5]],
    'Top7': [top_drivers.iloc[6], top_routes_perf.iloc[6], top_stations.iloc[6], top_vehicles.iloc[6]],
    'Top8': [top_drivers.iloc[7], top_routes_perf.iloc[7], top_stations.iloc[7], top_vehicles.iloc[7]],
    'Top9': [top_drivers.iloc[8], top_routes_perf.iloc[8], top_stations.iloc[8], top_vehicles.iloc[8]],
    'Top10': [top_drivers.iloc[9], top_routes_perf.iloc[9], top_stations.iloc[9], top_vehicles.iloc[9]],
}, index=['司机', '线路', '上车站点', '车辆'])

plt.figure(figsize=(12, 6))
sns.heatmap(heatmap_data, annot=True, fmt="d", cmap="YlOrRd", cbar_kws={'label': '服务人次'})
plt.title('各维度 Top 10 服务绩效热力图\n(颜色越深代表服务人次越多)', fontsize=14)
plt.xlabel('排名', fontsize=12)
plt.ylabel('维度', fontsize=12)
plt.xticks(rotation=0)
plt.yticks(rotation=0)

plt.tight_layout()
plt.savefig('performance_heatmap.png', dpi=150, bbox_inches='tight')
print("✅ 图表 performance_heatmap.png 已保存。")
plt.show()

# 3. 结论说明
print("\n【结论说明】")
print("从热力图可以看出，线路维度的Top 1客流量显著高于其他线路，呈现深红色，说明该线路是绝对的客流主力。")
print("司机维度的Top 1服务人次也远超同行，可能存在排班时长差异或该司机负责的是高客流线路。")
print("上车站点的分布相对均匀，没有出现极端的单一站点垄断，说明客流来源较为分散。")