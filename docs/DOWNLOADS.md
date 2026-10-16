# Downloads and local assets / 下载与本地资产

## Code / 代码

```bash
git clone https://github.com/Cyril-Fan/SPCP.git
```

Without Git / 无 Git：[Download main as ZIP / 下载 main 分支 ZIP](https://github.com/Cyril-Fan/SPCP/archive/refs/heads/main.zip). Extract the archive, enter its directory, and follow the root README. 下载后先解压，再按根目录 README 安装和准备资产。

## CLIP

```bash
python scripts/download_clip.py --weights
```

| Asset / 资产 | Source / 来源 | SHA-256 |
|---|---|---|
| Source / 源码 | [CoOp pinned commit](https://github.com/KaiyangZhou/CoOp/tree/ff61507c790454bce7c5052c3ac39e60772f1f89/clip) | Four exact hashes in `spcp/config.py` |
| BPE tokenizer | Same commit / 同一提交 | `924691ac288e54409236115652ad4aa250f48203de50a9e4722a6ecd48d6804a` |
| ViT-B/16 weights / 权重 | [OpenAI official file](https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt) | `5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f` |

Default locations are `external/original_clip/clip/` and `external/ViT-B-16.pt`. The script performs explicit downloads and hashes each file; `--check-only` uses no network. Files with a different existing hash are not silently replaced. Keep third-party copyright/license notices when redistributing upstream material; license copies are in `third_party/`.

默认路径为上表对应的 `external/` 目录。脚本校验每个文件；已有内容不同会报错，不会静默覆盖。离线用户可在联网机器下载同一版本，拷贝到本机后运行 `--weights --check-only`。源码是历史实验使用的 CoOp 随附版本，不应被任意最新版本替代。

## Original image collections: B2N / CD

Provider links are also recorded in the [pinned APPLeNet README](https://github.com/mainaksingha01/APPLeNet/blob/42bc4654311da9773b59979dc71fe9f60534ab84/README.md#datasets):

| Dataset | Provider / 原发布方入口 | Example relative ID / 相对路径示例 |
|---|---|---|
| PatternNet | [Dataset page](https://sites.google.com/view/zhouwx/dataset) | `oil_well/oilwell107.jpg` |
| RSICD | [RSICD repository](https://github.com/201528014227051/RSICD_optimal) | `center/center_165.jpg` |
| NWPU-RESISC45 | [Dataset information](https://www.tensorflow.org/datasets/catalog/resisc45) | `runway/runway_197.jpg` |
| MLRSNet | [Mendeley release v3](https://data.mendeley.com/datasets/7j9bv9vwsx/3) | `airplane/airplane_00735.jpg` |

Use the image files matching the bundled published split IDs. A loader that renames/re-encodes files, or the original RSICD captioning/multilabel annotations alone, is not sufficient. The resolver supports class-directory and flat-image layouts, but every declared ID must resolve unambiguously. Consult the split files to determine the expected names. For RESISC45 the linked catalog is an information/download entry; do not substitute TFDS-generated filenames for the original released filenames.

需使用与发布划分 ID 匹配的图像。仅有原始描述/多标签标注不足以运行分类协议；不要改名、重编码或重新划分。解析器支持按类别目录及平铺图片布局，但每个 ID 必须唯一定位。RESISC45 的 catalog 是资料入口，不能用转换后产生的新文件名替代原始名称。

## Released v2 images: DG

These are the four image archives linked by APPLeNet, with **16 shared classes**:

| Dataset | Upstream archive / 上游下载 |
|---|---|
| PatternNetv2 | [Google Drive](https://drive.google.com/file/d/1K-GZ2KjQ3hn17JJBrxnmXsTxAFeg2XUT/view) |
| RSICDv2 | [Google Drive](https://drive.google.com/file/d/1uhlTHQCHkE0KD04YGBAKsxPgG14eQez_/view) |
| NWPU-RESISC45v2 | [Google Drive](https://drive.google.com/file/d/1Zfsko5swyQqu5HiuRwZe5jIGoUKfBgxq/view) |
| MLRSNetv2 | [Google Drive](https://drive.google.com/file/d/1OJrAwU1i9hYe7kEsHIIq_TodJDiwnnAz/view) |

Extract each archive and point the binding at the directory resolving IDs such as `baseballfield/baseballfield_225.jpg`. A nested `images/` directory is normal. The bundled pixel ledgers identify the released image bytes; RGB decoding and hash mismatches are errors. Do not generate a new 16-class subset from the original datasets and call it the same v2 cohort.

解压后绑定到能解析对应相对 ID 的图像目录，常见结构包含 `images/`。DG 会按已有像素 ledger 核验字节，不能从原始数据集自行抽出 16 类后直接替代 v2 发布集合。下载链接由上游维护；本仓库提供出处，不声明持续验证其下载可用性。

## Local binding / 本地路径绑定

Copy `image_roots.example.json` to `image_roots.json`, replace placeholders and remove entries for unavailable datasets. All datasets referenced by the selected config must be present. Then run:

```bash
python scripts/bind_external_assets.py --clip-root external/original_clip --clip-checkpoint external/ViT-B-16.pt --image-roots-json image_roots.json --output external_assets.json
```

Windows JSON paths can use forward slashes, for example `D:/datasets/PatternNet`. Quote shell arguments containing spaces. Binding files, images, downloaded weights and run outputs are excluded from Git.

Windows 路径可写成 `D:/datasets/PatternNet`，含空格的命令参数加引号。绑定文件、图像、下载权重和运行输出均被 Git 忽略。

## SPCP checkpoints / SPCP 训练检查点

The training commands produce `final.pt`, which contains trained prompts and method state. Pretrained SPCP checkpoints are not included in this release. Download the frozen CLIP backbone separately using the setup script.

首版暂未托管训练完成的 SPCP 权重。训练命令生成 `final.pt`，保存提示及方法状态；冻结骨干通过外部资产提供。
