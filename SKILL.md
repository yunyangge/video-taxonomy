---
name: video-taxonomy
description: 根据外部类目配置与视频标签，按第三种方案执行中心编码、语义匹配、逐视频投票、去重和一级占比汇总。适用于完整运行或继续已有分类任务，支持模型编码与兼容向量缓存；不负责生成真实类目或采集视频。
---

# 视频标签分类：完整执行流程

目标是运行仓库现有程序，交付逐视频分类、每个平台的 L1 占比、各阶段耗时和核验摘要。读取配置、输入和输出字段时参照 [docs/usage.md](docs/usage.md)。下述命令从仓库根目录执行；`--config` 放在子命令之前，其余参数放在所属子命令之后。

## 输入与执行边界

先检查当前任务已经提供的路径和已有产物，只询问仍缺失的必要输入：

| 输入 | 要求 |
|---|---|
| 外部类目配置 | 仓库 schema 2 的 JSON；使用 `all_paths`，或可迁移的 `leaf_only` 配置。任意表格或其他 JSON 结构不能直接作为该配置使用。 |
| 视频元数据 | JSONL 或 Parquet，可提供多个文件。默认从 `tags` 读取标签；记录可包含 `platform`、`video_id`／`id`、`url`。 |
| 模型或缓存 | 新建中心、新标签编码需要本地 Qwen3-Embedding-0.6B 权重；兼容的完整运行目录及查询表可以不加载模型。仅有标签向量而没有中心向量时，仍需要模型编码中心。 |
| 运行位置 | 仓库外的输出目录；新配置、新中心表达使用新的运行目录。记录实际设备和批大小。 |

实际类目、私有标签、源文件、业务约束、运行日志和结果都保存在仓库外。仓库中的 `tests/fixtures.py` 只用于合成验证，不能替代正式输入。执行分类不包含下载模型、抓取数据、上传产物或改写 Git 历史的授权；依照用户对这些动作已有的授权处理。

## 分类规则

- 候选中心为 **L1 ＋ L1→L2 ＋ L1→末端完整路径**的并集，每个节点只出现一次。未展开的 L2 叶子不重复加入。路径默认以 ` / ` 连接。
- 使用完整 1024 维向量，保持 NFC、空白清洗、最多 2048 字符、最长 96 token、左填充、最后 token 池化和 L2 归一化一致。
- 每条有效标签与全部中心比较余弦相似度，选择最高者。默认 `--min-score -1 --min-margin 0`。标签先匹配中心，再映射到 L1。
- 默认只用 `tags`。需要让标题和简介投票时，由用户选择 `--evidence-source metadata`，单独记录这一统计口径。
- 视频内重复标签一票，按 L1 票数、相似度之和、配置顺序决定唯一 L1；下级在获胜分支内投票。直接命中 L1/L2 可以成为最终节点，不强行分配叶子。
- 多文件尽量在同一次 `classify` 中处理，以便统一去重。程序按平台与身份保留第一条记录；无身份记录独立计数。若数据要求其他重复记录规则，先说明差异。

## 1. 安装与预检查

使用 Python 3.10 或以上，在隔离环境中安装仓库。仅运行已有向量／查询表时安装基本依赖；需要编码时安装模型依赖：

```bash
python -m pip install -e .
# 需要创建中心或编码新标签时执行：
python -m pip install -e '.[model]'
video-taxonomy --help
```

将以下变量设置为本次任务的实际路径与参数。代码块中的路径是占位示例，不可原样当作真实输入：

```bash
VT_CONFIG=/absolute/path/to/categories.json
VT_RUN_DIR=/absolute/path/to/new-run
VT_MODEL_DIR=/absolute/path/to/Qwen3-Embedding-0.6B
VT_INPUT=/absolute/path/to/videos.jsonl
VT_PLATFORM=actual-platform
VT_DEVICE=cpu
```

有可用、已获准使用的 CUDA GPU 时可设置 `VT_DEVICE=cuda:0`；不要假定机器必定有 GPU。多平台输入优先保留记录中的 `platform` 字段；没有该字段时，`--platform` 是本次输入的默认值。

先检查配置，不加载模型：

```bash
video-taxonomy --config "$VT_CONFIG" validate-config
```

若明确是旧版 `leaf_only` 配置，保留原文件并迁移到仓库外的新文件，再将 `VT_CONFIG` 指向迁移结果：

```bash
video-taxonomy migrate-config --input "$VT_CONFIG" --output /absolute/path/to/categories-all-paths.json
VT_CONFIG=/absolute/path/to/categories-all-paths.json
video-taxonomy --config "$VT_CONFIG" validate-config
```

缺字段或指纹错误时先核对原始配置，不通过删字段、伪造类目或关闭校验绕过。确认输入文件可读、输出位置可写、容量足够；沿用同一个 Python 环境执行后续命令。

## 2. 取得本次运行的中心向量

新运行先编码中心：

```bash
video-taxonomy --config "$VT_CONFIG" embed-centers \
  --run-dir "$VT_RUN_DIR" --model "$VT_MODEL_DIR" \
  --device "$VT_DEVICE" --batch-size 128 --path-separator slash
```

成功后应有 `run.json`、`center_vectors.npy`、`center_vectors.json`。程序校验配置／编码指纹、中心形状与文件散列；已有中心不覆盖。

复用已有运行目录时，先检查这三个文件与本次配置、第三种方案、分隔符和编码规则匹配。匹配后跳过 `embed-centers`；不匹配则选新目录。改变类目、路径表达或向量维度后重新生成中心及查询表。

## 3. 选择标签匹配路径

### A. 直接编码视频标签

常规输入使用此路径。`classify` 内部提取和编码标签、比较全部中心、逐视频投票和去重：

```bash
video-taxonomy --config "$VT_CONFIG" classify \
  --run-dir "$VT_RUN_DIR" --model "$VT_MODEL_DIR" \
  --device "$VT_DEVICE" --batch-size 128 \
  --input "$VT_INPUT" --platform "$VT_PLATFORM" \
  --evidence-source tags --min-score -1 --min-margin 0 \
  --output "$VT_RUN_DIR/classified.jsonl"
```

多个文件写在同一 `--input` 后面，例如 `--input "$VT_INPUT" /absolute/path/to/another.parquet`。若输入已经提供逐记录平台，则保留它们；不同平台分别统计，不构造跨平台全网占比。

### B. 复用完整标签向量或查询表

若有兼容标签向量，先建立本次中心对应的查询表：

```bash
VT_VECTOR_CACHE=/absolute/path/to/full-vector-cache
VT_TAG_SOURCE=/absolute/path/to/tags.parquet
video-taxonomy --config "$VT_CONFIG" assign-tags \
  --run-dir "$VT_RUN_DIR" \
  --legacy-path-vectors "$VT_VECTOR_CACHE" --tag-source "$VT_TAG_SOURCE" \
  --batch-size 4096 --min-score -1 --min-margin 0
```

该适配器要求标签文件含 `tag` 列，向量按相同 Parquet row group 排列；缓存有 `center_encoding.json` 及 `vectors/000000.npy` 等文件。每行必须是完整 1024 维，并保持同一模型及预处理。零向量沿用原缓存的过滤结果。若来源不符合该格式，先验证并转换，不仅改文件名。

已有本次运行兼容的 `tag_lookup/metadata.json` 及各数组时，跳过 `assign-tags`。然后按下面命令分类，字典外标签保留为未知：

```bash
video-taxonomy --config "$VT_CONFIG" classify \
  --run-dir "$VT_RUN_DIR" --input "$VT_INPUT" --platform "$VT_PLATFORM" \
  --tag-lookup "$VT_RUN_DIR/tag_lookup" --lookup-only \
  --evidence-source tags --min-score -1 --min-margin 0 \
  --output "$VT_RUN_DIR/classified.jsonl"
```

若希望用模型补齐字典外标签，去掉 `--lookup-only`，并提供 `--model "$VT_MODEL_DIR" --device "$VT_DEVICE"`。阈值与查询表保持一致。明确记录选了“字典外保留未知”还是“编码补齐”，两者覆盖率不同。

## 4. 汇总与验收

```bash
video-taxonomy --config "$VT_CONFIG" aggregate \
  --input "$VT_RUN_DIR/classified.jsonl" \
  --output "$VT_RUN_DIR/distribution.json"
```

默认统计全部去重视频。`--accepted-only` 用于用户明确要求的已复核准入统计，不用于默认数据集分布。多个分类结果合并前确认运行指纹、证据来源和阈值一致，且不存在重复视频；有重复时统一输入重新分类。

至少核对：

- `classified.jsonl.meta.json` 中 `completed=true`；输出行数等于 `stats.unique_rows`，原始行数等于唯一行数加重复行数。
- `distribution.json` 中各平台总量与逐视频结果一致；`primary`、`secondary`、`content_node`、`leaf`、`production_form` 每个维度加未知项后均覆盖全部视频。
- L1 的视频数等于该 L1 下最终节点的计数总和，包含直接停在 L1 的视频。L1 已知、L2/leaf 未知的记录仍是 L1 已分类。
- L1 全部分母占比读取 `dimension=primary` 的 `share_all`；排除未知后的占比读取 `share_classified`，明确注明分母。不要混加不同统计维度，也不要用标签频次替代视频数。
- 记录未归类视频、无身份记录和字典外证据量。`unmatched_evidence` 是批次内未命中字典的不同证据计数之和，不应表述为全局去重标签数。

每个实际执行阶段记录开始、结束、墙钟耗时、处理规模、设备和批大小。`classify` 同时包含在线编码、匹配和视频处理时，报告整体耗时，不编造内部阶段拆分。已有缓存被跳过时标注“复用”，不填成重新计算零耗时。

## 交付与异常处理

交付逐视频结果及其 `.meta.json`、分布 JSON，以及按平台汇总的 L1 数量／占比、未归类量、去重结果和各阶段耗时。报告采用的配置指纹、模型、完整维度、第三种方案与证据来源。大数据任务逐视频结果保留在运行目录，不将全部内容粘贴到回复中。

命令失败时保留日志并检查具体原因。已有输出或 `.partial` 文件、配置／向量不匹配、重复视频等错误不通过覆盖文件或忽略检查消除；定位后使用新输出路径，或在用户已授权的范围内修复输入再运行。明确说明未完成的阶段。

修改代码后运行 `python -m unittest discover -s tests -v`。合成测试只证明软件流程，不证明真实模型的分类准确率。最终汇报不把测试产物当作正式结果；没有实际模型／数据运行证据时，不声称完成真实推理。
