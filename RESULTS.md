# 作业二最终成果与提交入口

## 最终版本（v4 linear）：83.7259%

- 方法：54 项无参考答案的候选/题目特征 + 1,024 维固定 embedding，使用多正例 listwise logistic 目标（λ=0.01），仅用训练集拟合。
- 官方验证：**748/798 = 93.7343%**；验证难题子集（≥2 个不同非 None 答案）：**305/355 = 85.9155%**。
- 官方最终测试：**391/467 = 83.7259%**（目标 83.5% 已达成）。
- 原始记录预测加独立评分 **1.323 秒**（本地 i7-10870H，各库线程上限为 4，未监测原始 OS 总线程）。独立后测采用单线程库设置，完整新进程评测 **1.756 秒**，采样 OS 线程峰值为 3；参考机未验证。
- 资产磁盘大小 **25,442 字节**；保守解压计数为 **26,395 字节**，不是将压缩文件大小当作解压大小。
- 推理只依赖 NumPy；scikit-learn/SciPy 仅用于训练。
- 训练可复现：`code/train_linear_v2.py` 独立重建的均值、标准差和权重与冻结 bundle **逐元素完全一致**；训练耗时 10.96 秒。

### 证据文件

| 内容 | 路径 |
|---|---|
| 冻结推理实现 | `code/submission.py` |
| 特征 | `code/features_v2.py` |
| 训练脚本 | `code/train_linear_v2.py`（最终）/ `code/train_v2.py`（对照实验） |
| 冻结资产 | `code/runs_linear/model_bundle/` |
| 验证评分 | `code/runs_linear/validation-score.json` |
| 最终测试评分 | `code/runs_linear/test-score.json` |
| 冻结哈希 | `code/runs_linear/freeze.json` |
| 后测完整性补充 | `code/runs_linear/review/integrity.json` |
| OOF 代理比较说明 | `code/runs_linear/review/selection-provenance.json` |
| 资源记录 | `code/runs_linear/reproduction.json` |
| 训练指标 | `code/runs_linear/train_metrics.json` |
| 报告 | `report/REPORT.md`、`report/REPORT.pdf` |
| 复现命令 | `code/REPRODUCE.md` |

## 与历史版本的关系

| 版本 | 验证总体 | 验证难题子集 | 官方测试 |
|---|---|---|---|
| v1 初始提交 | 92.6065% (739/798) | 83.38% (296/355) | 81.7987% (382/467) |
| v2（历史） | — | — | 82.2270% (384/467) |
| v3（历史） | 93.6090% (747/798) | — | 81.3704% (380/467) |
| v2 集成实验 | 94.3609% (753/798) | 87.32% (310/355) | 83.0835% (388/467) |
| **v4 linear（最终）** | **93.7343% (748/798)** | **85.92% (305/355)** | **83.7259% (391/467)** |

训练集分组 OOF 中，线性为 1712/2114（80.98%），代理集成为 1705/2114（80.65%）。差距只有 7 题；代理使用一个 MLP 种子和 30 个 epoch，实际被测试的集成使用三个种子和验证选择的 epoch，因此不能称为最终集成的完全相同交叉验证，也不能据此确立统计优势。它是偏向简单模型的补充证据。

最终线性与后期集成都在各自测试前冻结。集成测试先于后续 OOF 分析，之后才冻结并测试最终线性；此前各版本测试和同学公开成绩也已知。报告披露完整历史，不称为一次完全盲测。原始 freeze 漏记最终训练脚本，已另加后测哈希补充，未改写测试前记录。最终推理源码和权重不作此次文档整理的修改。

## 仍需你完成

1. 阅读并理解 `submission.py`、`features_v2.py`、`train_linear_v2.py` 与报告，确认 AI 使用披露。
2. 将完整源码、报告、`runs_linear/` 证据和 checkpoint 发布到公开仓库；确保 `.gitignore` 不会排除 `code/runs_linear/model_bundle`。
3. 也可直接上传 `dist/MP2-checkpoint.zip` 作为 checkpoint 链接。
4. 在作业网站填写真实学号和官方分数 **83.7259**，填写代码与 checkpoint 链接。
5. 点击 **Submit on GitHub** 并实际创建 issue，保存提交编号。
6. 截止时间：**2026 年 10 月 31 日 23:59（北京时间，UTC+8）**。
