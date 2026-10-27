# Video Taxonomy

视频元信息分类引擎：使用调用方提供的类目配置，以完整路径匹配叶子节点，再按一级、二级和叶子逐层统计。

**仓库只提供代码，不包含实际类目体系、业务配额或真实视频元数据。** 私有配置必须保留在仓库之外，通过参数或环境变量传入。

## 安装

需要 Python 3.10 或以上。

```bash
python -m pip install -e .
# 需要模型编码时，再安装可选依赖
python -m pip install -e '.[model]'
```

## 使用

```bash
video-taxonomy --config /path/outside/repo/categories.json embed-centers --run-dir runs/example --model models/Qwen3-Embedding-0.6B --path-separator slash
video-taxonomy --config /path/outside/repo/categories.json classify --run-dir runs/example --model models/Qwen3-Embedding-0.6B --input data/videos.jsonl --platform example --output runs/example/classified.jsonl
video-taxonomy --config /path/outside/repo/categories.json aggregate --input runs/example/classified.jsonl --output runs/example/distribution.json
```

也可通过 `VIDEO_TAXONOMY_CONFIG` 指定外部配置，使用 `python -m video_taxonomy` 调用。模型仅从本地读取；`VIDEO_MODEL_DIR` 可指定模型目录。参数必须放在对应的全局/子命令位置，见 `--help`。

## 规则

- 只匹配叶子。有三级时匹配三级，没有三级时匹配二级。
- 类目中心使用从一级到叶子的完整路径，默认以 ` / ` 连接。
- 逐证据匹配后，按一级 → 二级 → 叶子投票；同票按相似度总和及类目顺序确定。
- 内容与制作形态分别统计。没有有效证据时保留无法归类。
- 配置或编码方式改变后使用新的运行目录，不混用旧缓存。

结果属于元信息候选标签，不代表观看视频后的判断。相似度不是概率。

## 测试

```bash
python -m unittest discover -s tests -v
```

测试仅使用程序生成的虚构类目、合成向量与临时文件，不读取实际类目。包含超过 255 个叶子的编码、分层投票、去重、未知类目和统计检查。

输入、配置和输出字段见 [使用说明](docs/usage.md)。不要将私有配置、导出的类目数据、运行缓存或真实样本加入提交。
