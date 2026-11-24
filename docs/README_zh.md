# 仓库使用导航

本仓库对应论文 Code Availability 中的流程：预处理 → 检测与切窗 → 事件关联 → 技术验证 → 绘图复现。测试环境为 Python 3.12。

| 我想做什么 | 从哪里开始 | 是否需要下载全量波形 |
|---|---|---|
| 查看处理命令与阶段顺序 | [处理流程](workflows.md) | 样例不需要 |
| 滤波、触发、切窗与人工复核模板 | 01_preprocessing / 02_event_detection / 03_manual_annotation / [三台站样例](../examples/continuous) | 样例不需要；整天数据需主数据集 |
| 重建/核查五类事件关联表 | [event_association](../event_association/README.md) / [data/data logs](../data/data%20logs) | 核查随包表无需全量波形；重建需主数据与校验包 |
| 读取分类波形 / pickle-free 校验 | 04_data_loading / 07_pickle_free_records | NPY 需主数据；NPZ 需独立 Figshare 包 |
| 分类基线与技术验证 | 05_classification / 06_validation / [validation_results](../validation_results) | 重绘已存结果可不训练；重训需分类波形 |
| 复现论文图件 | [figures/README.md](../figures/README.md) / figures/reproduce.py | 多数图件有随包数值输入 |

五类标签文件夹：microseismic、blasting、small_landslide、mechanical_mining、transport_vehicles。

在仓库根目录安装依赖后运行：

~~~sh
python -m pip install -r requirements.txt
python 01_preprocessing/preprocess_raw.py --raw-dir examples/continuous --out-dir outputs/demo/filtered
python 02_event_detection/detect_events.py --raw-dir examples/continuous --out-dir outputs/demo/candidates
python 03_manual_annotation/review_candidates.py --candidates outputs/demo/candidates --out-dir outputs/demo/review
python figures/reproduce.py --figure all
~~~

数据链接：

- 主数据集：https://doi.org/10.6084/m9.figshare.34074750
- pickle-free 校验包：https://doi.org/10.6084/m9.figshare.34128987

上传到 GitHub 时，保留仓库 .git，将本目录内容置于仓库根目录并使 README.md 位于首页。上传前可运行 python verify_release.py。
