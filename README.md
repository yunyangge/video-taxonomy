# Video Taxonomy

视频标签语义分类引擎，采用**第三种方案：L1 ＋ L1→L2 完整路径 ＋ L1→末端完整路径**。使用调用方提供的类目树，匹配标签、确定每个视频的内容类别，再统计一级类目的视频占比。

**仓库只提供代码，不包含实际类目体系、业务配额或真实视频元数据。** 配置、模型、输入数据和运行结果均由调用方在仓库外提供。

## 给 Agent 的 Skill

**Agent 入口：[SKILL.md](SKILL.md)**。读取本 README 的 agent，应先完整读取根目录的 `SKILL.md`，再按其中的输入检查、执行分支和验收条件运行。详细字段见 [docs/usage.md](docs/usage.md)。

可将下面这段交给 agent，并把输入项替换为实际位置：

```text
使用本仓库根目录 SKILL.md 中的 video-taxonomy skill，按第三种方案完成全套流程。
外部类目配置：实际配置文件路径
视频元数据：实际 JSONL / Parquet 文件路径（可以有多个）
本地模型：实际 Qwen3-Embedding-0.6B 模型目录
输出目录：仓库外的新目录
平台：实际平台标识；多平台时读取各记录的 platform 字段
如果已有兼容的中心向量、标签向量或查询表，请检查后复用。
完成配置校验、中心编码、标签匹配、逐视频投票、去重和各 L1 视频占比汇总，
记录各阶段耗时，交付逐视频结果、分布文件和核验摘要。
缺少必要输入时先检查当前任务已有文件，只询问仍缺失的部分；不要构造正式类目或伪造运行结果。
```

Skill 覆盖两条执行路径：**直接用模型编码标签**，以及**复用兼容向量／查询表**。现有查询表不足以覆盖全部标签时，明确报告未匹配量，或在有模型的情况下编码补齐。正式数据始终保存在仓库外。

## 计算方法

1. 为每个一级节点、二级节点和三级叶子生成一个候选中心。路径默认用 ` / ` 连接；未展开的二级叶子只保留一次。中心数量等于一级、二级、三级节点数量之和。
2. 使用 Qwen3-Embedding-0.6B 的完整 **1024 维**向量，不截取前若干维。标签与中心使用同样的清洗、最长 96 token、左填充、最后一个 token 池化和 L2 归一化。
3. 每条有效标签与全部候选中心计算余弦相似度，选择最高者。不先筛选 L1，不用 K-means 重新拟合类目。默认没有额外的相似度拒判阈值；无效或零向量保留为未知。
4. 同一视频的重复标签只投一票。先按匹配结果所属的 L1 计票；票数相同时比较相似度之和，再按配置中的类目顺序确定。每个视频只计入一个 L1。
5. 各 L1 占比 = 该 L1 的去重视频数 / 全部去重视频数。无有效标签的视频单列为未知，也可查看排除未知后的占比。

输出下级类别时，只在获胜的 L1 分支内继续投票。直接命中 L1 的票与 L2 分支票共同比较；在 L2 内，直接命中 L2 的票与其叶子票比较。获胜结果可以停在 L1 或 L2，未知下级不会被强行补齐。制作形态单独统计，不参与内容中心竞争。

## 安装

需要 Python 3.10 或以上。

```bash
python -m pip install -e .
# 需要模型编码时，再安装可选依赖
python -m pip install -e '.[model]'
```

## 使用

先准备仓库外的类目配置、视频标签元数据和本地模型权重（或已生成的兼容中心向量及查询表）。本仓库不下载模型、不自带实际类目，也不把合成测试当成真实视频准确率评估。

外部配置使用 `schema_version=2`、`matching_mode=all_paths`。已有叶子匹配配置需要先迁移；此命令保留类目定义，只更新匹配策略和指纹，不覆盖原文件：

```bash
video-taxonomy migrate-config --input /path/outside/repo/old-categories.json --output /path/outside/repo/categories-all-paths.json
```

随后使用**新运行目录**重新生成中心与查询表，不能复用旧方案的分类结果：

```bash
video-taxonomy --config /path/outside/repo/categories-all-paths.json validate-config
video-taxonomy --config /path/outside/repo/categories-all-paths.json embed-centers --run-dir /path/outside/repo/runs/scheme3 --model /path/to/Qwen3-Embedding-0.6B
video-taxonomy --config /path/outside/repo/categories-all-paths.json classify --run-dir /path/outside/repo/runs/scheme3 --model /path/to/Qwen3-Embedding-0.6B --input /path/to/videos.jsonl --platform example --output /path/outside/repo/runs/scheme3/classified.jsonl
video-taxonomy --config /path/outside/repo/categories-all-paths.json aggregate --input /path/outside/repo/runs/scheme3/classified.jsonl --output /path/outside/repo/runs/scheme3/distribution.json
```

也可通过 `VIDEO_TAXONOMY_CONFIG` 指定外部配置，使用 `python -m video_taxonomy` 调用。模型仅从本地读取；`VIDEO_MODEL_DIR` 可指定模型目录。

默认只使用 `tags` 作为内容证据，与标签实验一致。需要同时使用标题和简介时，显式传入 `classify --evidence-source metadata`；其结果属于另一种证据口径，应单独统计。兼容的完整标签向量可通过 `assign-tags` 重新匹配到新中心，详见 [使用说明](docs/usage.md)。

结果属于元信息候选标签，不代表观看视频后的判断。相似度不是概率；采用第三种方案并不意味着已经通过人工真值验证准确率。

## 测试

```bash
python -m unittest discover -s tests -v
```

测试仅使用程序生成的虚构类目、合成向量与临时文件，覆盖全层级中心、二级叶子去重、超过 255 个中心、直接命中 L1/L2、视频投票、负相似度、旧配置迁移、缓存隔离、未知类别及汇总一致性。

不要将私有配置、导出的类目数据、运行缓存或真实样本加入提交。
