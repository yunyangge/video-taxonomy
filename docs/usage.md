# 外部配置与第三种方案

本仓库不附带真实类目。`--config` 或 `VIDEO_TAXONOMY_CONFIG` 必须指向仓库外的 JSON 配置；缺失时程序停止并提示。

## 配置与迁移

配置使用 `schema_version=2`、`matching_mode=all_paths`；包括 `primary`、`production_forms`、`production_form_priority`、`embedding`、`unknown_id` 与 `fingerprint`。每个一级节点包含 `secondaries`；二级节点包含 `children`（可以为空）、`quota`、`restricted` 与 `restriction`。各级 ID 唯一，同一父节点下名称唯一。`fingerprint` 为移除该字段后，按键排序、紧凑 JSON 编码的 SHA-256。

顶层还必须提供 `taxonomy_version`。`unknown_id` 固定为 `unclassified`。一级和制作形态也需要 `quota`（不设配额可为 `{}`）；`production_forms` 和 `production_form_priority` 可以同时为空列表。`embedding` 必须提供 `model_name`、`dimensions`、`pooling`、`max_length` 和 `normalization`，配置值应与下述编码规则一致。

agent 执行前必须先找到这些输入：外部配置、视频元数据、本地模型权重或兼容的向量及查询表。不要把文档里的路径占位符当成真实文件。先运行不加载模型的预检查：

```bash
video-taxonomy --config /outside/all-paths.json validate-config
```

成功时返回中心总数、各级数量及维度；配置缺字段、指纹不符或仍采用旧策略会报错。合成测试可以验证安装和流程，真实语义匹配还需要调用方的模型及数据。

`tests/fixtures.py` 提供程序生成的虚构结构，仅用于理解接口和运行测试。实际类目和约束由调用方保管。

旧 `matching_mode=leaf_only` 配置通过以下命令迁移；原文件不改动，指纹重新计算：

```bash
video-taxonomy migrate-config --input /outside/old.json --output /outside/all-paths.json
```

新策略不能读取旧中心运行目录或旧标签查询表。迁移后，新建运行目录并重新执行 `embed-centers`、`assign-tags`（如果使用查询表）、`classify`、`aggregate`。

## 候选中心与标签匹配

第三种方案的中心是下列集合的并集：

- 每个 L1 的名称。
- 每个 L1→L2 完整路径。
- 每个 L1→末端完整路径。

每个节点只有一个中心，二级叶子不重复加入。节点顺序为配置树的先序顺序：L1、其 L2、该 L2 的叶子。默认分隔符为 ` / `。`--path-separator space` 只用于显式对比，改变分隔符必须另建运行目录。

使用完整 1024 维 Qwen3-Embedding-0.6B 特征：NFC、合并空白、最多 2048 字符、最长 96 token、左填充、最后 token 池化、L2 归一化；不附加指令提示词。对全部中心比较余弦相似度，取最高者。无效或零向量不分配类别。默认 `--min-score -1 --min-margin 0`，不增加低置信度拒判；显式设置更严格的阈值属于不同实验口径。

可复用兼容的完整 1024 维标签向量：

```bash
video-taxonomy --config /outside/all-paths.json assign-tags --run-dir /outside/runs/scheme3 --legacy-path-vectors /outside/full-vector-cache --tag-source /outside/tags.parquet
video-taxonomy --config /outside/all-paths.json classify --run-dir /outside/runs/scheme3 --input /outside/videos.jsonl --tag-lookup /outside/runs/scheme3/tag_lookup --lookup-only --output /outside/runs/scheme3/videos.jsonl
```

向量目录需要 `center_encoding.json` 和按 Parquet row group 排列的 `vectors/000000.npy` 等文件，预处理、模型、池化和归一化必须一致。沿用缓存中的零向量过滤结果；只复用向量，重新匹配全部中心。查询表记录 `center_ids`、运行指纹与阈值。`--lookup-only` 让字典外证据保持未知，不启动模型编码。

## 视频输入与投票

输入支持 JSONL 和 Parquet。每条可提供 `title`、`description`、`tags`、`platform`、`video_id`、`url`；标签字符串按逗号切分。默认 `--evidence-source tags`，只让标签参与内容投票；`metadata` 额外让标题和简介参与。重复证据清洗后只计一票。新证据会过滤空值、纯数字、乱码标记、URL 等明显无效内容；这不替代外部已复核的噪声标签筛选。

每个标签先匹配到一个节点，然后映射到所属 L1。L1 票数最多者胜出；同票时比较相似度之和，再按配置顺序确定。**每个视频只归一个 L1，不将标签出现次数当成视频数量。**

下级输出在获胜 L1 内计算：直接 L1 票与各 L2 分支票比较。如果直接 L1 胜出，停止下探；否则在胜出的 L2 内比较直接 L2 票和各叶子票。同样使用票数、相似度之和、配置顺序处理平票。这样既保留父子一致性，也允许分类停在内部节点。

视频按平台和身份去重，保留输入中第一次出现的记录；没有身份的行单独计数。BVD 身份包含来源域名。`effective_duration_seconds` 与原始 `duration_seconds` 分开处理。

外部已复核标注可提供 `content_node_id`（任意层级）、`content_secondary_id` 或 `content_leaf_id`（仅真实叶子）。多个字段必须一致。内部 L2 也可作为已复核结果，此时 `leaf_id` 仍为未知。`production_form`、`is_aigc` 保留原有独立处理逻辑。

## 输出和统计

视频输出包含：

| 字段 | 含义 |
|---|---|
| `primary_id` | 视频唯一的 L1；无有效证据时为 `unclassified` |
| `secondary_id` | 已确定的 L2；直接停在 L1 时为 `unclassified` |
| `content_node_id`、`content_node_level` | 视频最终停留的节点及层级，可以是 L1、L2 或 L3 |
| `leaf_id`、`leaf_level` | 仅当最终节点确实是叶子时填写，否则为未知／空 |
| `tertiary_id` | 仅 L3 叶子有值 |
| `accepted_votes` | 有效节点匹配票数，包括直接 L1/L2 匹配 |

汇总包含 `primary`、`secondary`、`content_node`、`leaf`、`production_form` 五个统计维度，每个维度加上未知项后都覆盖全部视频。`content_node` 记录互斥的最终节点分配，L1 占比按 `primary` 行读取：

`L1 视频数 = 直接停在该 L1 的视频数 + 最终停在其所有下级节点的视频数`。

`share_all` 使用全部去重视频作为分母；`share_classified` 使用该统计维度中已分类视频作为分母。L1 已知而 L2 未知，并不代表内容整体无法归类。不能把 `primary`、`secondary` 与 `leaf` 的汇总行相加，那会重复计数。程序校验每个维度的总数，并核对 L1 与最终节点的汇总一致性。

不同运行指纹、证据来源或拒判阈值的输出不能混合汇总。普通输出为待复核元信息候选，未经过人工真值准确率评估。配额仅在已复核、满足准入条件且有效时长齐全时评估。分享输出前检查其中的 ID、URL、标题与配置指纹；不要上传真实类目或源文档。
