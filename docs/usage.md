# 外部配置与输入

本仓库不附带真实类目。`--config` 或 `VIDEO_TAXONOMY_CONFIG` 必须指向仓库之外的 JSON 配置；缺失时程序停止并提示。

配置使用 schema_version=2、matching_mode=leaf_only；包括 primary、production_forms、production_form_priority、embedding、unknown_id 与 fingerprint。每个一级节点包含 secondaries；二级节点包含 children（可以为空）、quota、restricted 与 restriction。各级 id 必须唯一。fingerprint 为移除 fingerprint 字段后，按键排序、紧凑 JSON 编码的 SHA-256。`tests/fixtures.py` 提供程序生成的虚构结构，仅用于理解接口和运行测试。

输入支持 JSONL 和 Parquet。每条可提供 title、description、tags、platform、video_id、url；标签字符串会按逗号切分。`effective_duration_seconds` 与原始 `duration_seconds` 分开处理。按平台和身份去重，没有身份的行会单独计数。

外部已复核标注可通过 content_secondary_id、content_leaf_id、production_form、is_aigc 提供。父子不一致时报错。普通输出是待复核的元信息候选。

`assign-tags --legacy-path-vectors <目录> --tag-source <Parquet>` 可复用兼容的完整 1,024 维向量。要求模型、池化、文本清洗和归一化一致。`classify --tag-lookup <目录> --lookup-only` 不编码字典外证据。所有数据与缓存由调用方提供。

配额参数由外部配置决定；仅在已复核、满足准入条件且有效时长齐全时评估。分享分类结果前，另行检查其中的 ID、URL、标题与配置指纹。不要上传真实类目或源文档。
