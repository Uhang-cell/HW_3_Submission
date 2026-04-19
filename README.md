# 王鸣宇 - 25317037 - 第三次人工智能编程作业


## 1. 任务拆解与 AI 协作策略

本次作业我采用了**“分步增量式”**的协作策略，将6个任务拆解为独立的单元，逐个击破，以确保代码的可读性和版本控制的清晰度。

1.  **环境与数据预处理（任务1）：** 首先让 AI 根据数据样例推断分隔符（TSV/CSV）并编写读取代码。重点在于让 AI 理解“搭乘站点数”是需要计算的衍生字段，而非原始数据。
2.  **可视化专项（任务2 & 任务3）：** 针对 Matplotlib 和 Seaborn 的不同要求分别提问。特别是任务2，我特意强调了“必须使用 Numpy 布尔索引”，防止 AI 默认使用 Pandas 的 `.loc` 方法。
3.  **算法逻辑攻坚（任务4）：** 这是本次作业最难的部分。我向 AI 描述了 PHF 的业务逻辑（高峰小时/5分钟粒度/15分钟粒度），让 AI 帮助我构建时间分组的逻辑（`dt.floor` 或 `//` 运算）。
4.  **文件与热力图（任务5 & 任务6）：** 主要是利用 AI 生成文件夹遍历和热力图矩阵构造的样板代码。

这种策略避免了一次性生成长代码导致的逻辑混乱，也方便我针对每个任务进行单独的 `git commit`。

## 2. 核心 Prompt 迭代记录

**场景：** 任务 3 - 线路站点分析（函数封装）

*   **初代 Prompt：**
    > "写一个 Python 函数，计算各线路的平均搭乘站点数和标准差，用 Seaborn 画条形图。"
*   **AI 生成的问题：**
    1.  函数名被 AI 自动命名为 `calculate_route_stats`，不符合题目要求的 `analyze_route_stops`。
    2.  参数名直接写死了 `'线路号'`，没有设置默认参数。
    3.  画图时没有处理误差棒（Error Bar）的形状不匹配问题（导致了 `ValueError: 'xerr' shape` 错误）。
*   **优化后的 Prompt：**
    > "严格按照以下函数签名编写代码：`def analyze_route_stops(df, route_col='线路号', stops_col='ride_stops'):`。请计算各线路乘客的平均搭乘站点数及其标准差，返回 DataFrame 并按均值降序排列。随后绘制 Top 15 的水平条形图，**注意：Seaborn 的 barplot 现在版本不支持直接传入 xerr 数组，请使用 ax.errorbar() 手动添加误差线**。"
*   **结果：**
    AI 修正了函数签名，并采用了 `ax.errorbar` 的方式解决了绘图报错问题。

## 3. Debug 记录

**报错现象：**
在任务 2 运行时，控制台输出了大量黄色警告：
`UserWarning: Glyph 23567 (\N{CJK UNIFIED IDEOGRAPH-5C0F}) missing from font(s) Arial.`
图表中的中文标题显示为方框或乱码。

**解决过程：**
1.  **分析：** 这是因为 Matplotlib 默认使用的字体（如 Arial）不包含中文字库。
2.  **排查：** 我检查了代码，虽然设置了 `SimHei`，但可能因为环境原因未生效。
3.  **修复：** 我在代码开头增加了更鲁棒的字体配置，并加入了负号显示修复。
    ```python
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    ```
    同时，为了确保在所有环境下运行，我还在 Prompt 中明确要求 AI "在绘图前设置中文字体支持"。
## 环境配置与问题修复
- **中文乱码修复**：为防止图表中中文显示为方块，已在代码头部添加 `plt.rcParams` 设置，自动适配当前系统的中文字体（SimHei/Microsoft YaHei/Arial Unicode MS），并修复了负号显示异常的问题。
## 4. 人工代码审查（逐行中文注释）

以下是对 **任务 4：高峰小时系数计算 (PHF)** 的核心代码进行的人工审查与注释。这部分逻辑较为复杂，涉及时间粒度的聚合。

```python
# --- 任务 4：高峰小时系数计算 (PHF) 核心逻辑审查 ---

# 1. 识别高峰小时
# 统计全天各小时的刷卡量 (hourly_vol 是一个 Series，索引为小时，值为计数)
hourly_vol = df_boarding['hour'].value_counts().sort_index()
# 找出刷卡量最大的那个小时的索引 (例如 8 点)
peak_hour = hourly_vol.idxmax()
# 获取该高峰小时的具体刷卡量数值
peak_volume = hourly_vol.max()

# 2. 5分钟粒度统计 (计算 PHF5)
# 筛选出高峰小时的数据副本
df_peak = df_boarding[df_boarding['hour'] == peak_hour].copy()
# 构造 5分钟 分箱：利用整除 // 和 乘法，将分钟数映射到 0, 5, 10, 15...
# 例如：08:13 的分钟是 13, 13//5=2, 2*5=10 -> 归类为 08:10 这个5分钟区间
df_peak['minute_5bin'] = (df_peak['交易时间'].dt.minute // 5) * 5
# 按小时和刚才构造的5分钟区间进行分组计数
vol_5min = df_peak.groupby(['hour', 'minute_5bin']).size()
# 找到这个高峰小时内，流量最大的那个5分钟窗口的刷卡量
max_5min_vol = vol_5min.max()

# 3. 计算 PHF5 公式
# PHF 公式：高峰小时总流量 / (12 * 高峰小时内最大5分钟流量)
# (因为 60分钟 / 5分钟 = 12 个区间)
phf5 = peak_volume / (12 * max_5min_vol)

# 4. 15分钟粒度统计 (计算 PHF15)
# 逻辑同上，只是分箱粒度改为 15 分钟
df_peak['minute_15bin'] = (df_peak['交易时间'].dt.minute // 15) * 15
vol_15min = df_peak.groupby(['hour', 'minute_15bin']).size()
max_15min_vol = vol_15min.max()
# PHF 公式：高峰小时总流量 / (4 * 高峰小时内最大15分钟流量)
# (因为 60分钟 / 15分钟 = 4 个区间)
phf15 = peak_volume / (4 * max_15min_vol)
