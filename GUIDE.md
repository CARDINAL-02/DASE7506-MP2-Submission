# DASE7506 作业二：最终提交 GUIDE

**最终版本为 v4 linear，官方测试 391/467 = 83.72591006423983%，已达到 83.5% 目标。** 验证为 748/798（93.7343%），验证难题子集为 305/355（85.9155%）。代码、权重、报告和本地复现已准备；公开仓库及网站提交尚未完成。

原始作业要求见 `guide/GUIDE.md`。以下步骤是成品的检查与提交路线，不能替代教师要求。

## 1. 认准最终文件，避免混用历史版本

| 用途 | 最终文件 |
|---|---|
| 推理接口 | `code/submission.py` |
| 特征 | `code/features_v2.py` |
| 训练 | `code/train_linear_v2.py` |
| 资产 | `code/runs_linear/model_bundle/` |
| 官方分数 | `code/runs_linear/test-score.json` |
| 报告 | `report/REPORT.pdf` |
| 精确命令 | `code/REPRODUCE.md` |
| 发布前清单 | `SUBMISSION.md` |

训练文件名保留了历史 `v2`，但正式版本标识是 `v4-linear`，目录是 `runs_linear`。旧 `runs/model_bundle_v2` 和 `model_bundle_v3` 都不是此次最终资产；集成实验位于 `runs_v2`，也不是最终提交模型。

## 2. 理解要提交的内容

每题有四个候选解答，程序返回一个 Python 整数 0–3，官方只检查所选候选的最终数值答案。最终方法使用 54 项参考答案无关特征和 1,024 维固定向量，训练目标提高所有正确候选的总概率，而不是要求多个正确候选平均分配概率。理解损失、特征、标准化和推理接口后再提交，并保留 AI 帮助披露。

## 3. 环境和数据

使用 Python 3.12。当前 VS Code 环境在工作区根目录 `.venv`；从项目 `code/` 运行时为 `../../.venv/Scripts/python.exe`。新副本可在 `code/` 创建自己的 `.venv`。

- 推理依赖：`requirements.txt`（NumPy）。
- 训练依赖：`requirements-training.txt`（另含 SciPy/scikit-learn）。
- 报告依赖：`requirements-report.txt`。

源码精简包不带课程题库 JSONL/NPZ；从原始课程包还原 `code/data/`，用原始 manifest 检查哈希，保留 LICENSE/NOTICE。完整 Release 包已包含数据。

## 4. 本地免训练复现

按 `code/REPRODUCE.md` 的命令运行官方 `predict.py` 和 `score_predictions.py`。必须使用 `submission.py` 与 `runs_linear/model_bundle`；输出另取新名，不覆盖冻结结果。设置 OMP、OpenBLAS、MKL、NumExpr 的线程上限均为 1，预测和评分顺序执行。

最终复核已再次得到 391/467，且所有预测选择与冻结文件一致。原始记录的双进程耗时为 1.323 秒；单线程保守设置的后测记录详见 `runs_linear/review/integrity.json`。本机耗时不能替代参考机资格认定。

## 5. 检查资产与证据

磁盘压缩文件共 25,442 字节，按解压 NPZ 成员（含 NPY 头）和配置保守计数为 **26,395 字节**，低于 16 MiB。不能把压缩大小写成解压后大小。

原始 `freeze.json` 保留不改；它漏记最终训练脚本的问题已在后测补充清单中披露。补充清单包含最终脚本哈希、精确训练重建结果、优化器状态和官方复现记录，不伪称为测试前记录。

## 6. 核对报告及研究边界

报告最多 10 页，保留初始/改进方法、消融、错误分析、资源和 AI 披露。最终训练只用 train，validation 用于开发与选择，test 只由官方评分器评分。此前测试成绩和公开排行榜已知，报告必须披露迭代历史，不称为一次完全盲测。

训练折外比较在线性与代理集成之间只有 7/2,114 题差距，且代理集成与实际三种子集成不同；不应夸大统计结论。历史信息对照和置信分析不能冒充最终 v4 的独立消融或校准结果。

## 7. 重新训练仅作复现

在独立副本运行 `train_linear_v2.py --output runs_linear/retrained_bundle`，不要覆盖最终资产。原脚本仍会写固定的 `train_metrics.json`，须先在副本备份该记录。最终复核已经仅用训练集重建，均值、标准差、权重逐元素完全一致。

## 8. 检查发布包和忽略规则

优先用 `dist/MP2-submission.zip` 发布最终精简仓库材料，使用 `dist/MP2-checkpoint.zip` 作为匹配权重包。更多历史研究在 `MP2-source.zip`，完整课程数据在 `MP2-complete.zip`，大 ZIP 作为 Release assets，不要提交进 Git 源码历史。

确认 `.gitignore` 和全局规则没有忽略 `code/runs_linear/model_bundle/`、官方评分、原始 freeze 和 review 补充证据。修改任何交付文件后，在工作区根目录运行 `.venv/Scripts/python.exe package_release.py` 重建 manifest 和 ZIP，再核对 `dist/SHA256.json`。

## 9. 填写并实际提交

截止为 **2026 年 10 月 31 日 23:59（北京时间，UTC+8）**，个人作业占课程 10%。最终条目须包含真实学号、公开源码链接以及匹配 checkpoint/configuration 链接。

在 [提交网站](https://xudongwu-0.github.io/courses/dase7506/mp2/) 填入官方 accuracy **83.72591006423983**，然后点击 **Submit on GitHub** 并实际创建 issue；仅生成链接不算提交。保存 issue 编号，后续按教师要求参与复现。

## 10. 提交前自查

- [ ] 已理解代码、特征、训练目标及 AI 披露。
- [ ] 使用最终 `runs_linear/model_bundle`，没有混用旧模型与新分数。
- [ ] 官方评分为 391/467，报告及网页数值一致。
- [ ] 报告≤10页，资源与测试暴露记录如实披露。
- [ ] 公开仓库包含最终训练脚本、小权重、官方 JSON 及完整性补充清单。
- [ ] 所有链接可访问，资产无需重训即可运行，数据许可证保留。
- [ ] 已填真实学号并实际创建提交 issue。
