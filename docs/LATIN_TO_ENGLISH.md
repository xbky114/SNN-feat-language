# CUB 拉丁名称来源

## 原始信息来源

拉丁名称来自 [Fine-Grained-or-Not 的 Birds_name.xlsx](https://github.com/PRIS-CV/Fine-Grained-or-Not/blob/8a188108bd1484d1bffe0a0a0c766c9f03ef9925/birds/Birds_name.xlsx)，版本 `8a188108bd1484d1bffe0a0a0c766c9f03ef9925`。
第一张表 A、C、D 列分别是物种编号、科名、目名。按物种编号关联 `birds_get_tree_target_2.py`，即可对应 FreeGrainLearning 的目、科编号；原树编号减 1 对应加载器索引。

## 原始拉丁名称

```python
_CUB_ORDERS = (
    'Anseriformes',
    'Apodiformes',
    'Caprimulgiformes',
    'Charadriiformes',
    'Coraciiformes',
    'Cuculiformes',
    'Gaviiformes',
    'Passeriformes',
    'Pelecaniformes',
    'Piciformes',
    'Podicipediformes',
    'Procellariiformes',
    'Suliformes',
)
_CUB_FAMILIES = (
    'Anatidae',
    'Trochilidae',
    'Caprimulgidae',
    'Alcidae',
    'Laridae',
    'Stercorariidae',
    'Alcedinidae',
    'Laridae (Red legged Kittiwake)',
    'Cuculidae',
    'Gaviidae',
    'Alaudidae',
    'Bombycillidae',
    'Cardinalidae',
    'Certhiidae',
    'Corvidae',
    'Fringillidae',
    'Hirundinidae',
    'Icteridae',
    'Icteriidae',
    'Laniidae',
    'Mimidae',
    'Motacillidae',
    'Parulidae',
    'Passerellidae / Emberizidae',
    'Passeridae',
    'Ptilonorhynchidae',
    'Sittidae',
    'Sturnidae',
    'Troglodytidae',
    'Tyrannidae',
    'Vireonidae',
    'Pelecanidae',
    'Picidae',
    'Podicipedidae',
    'Diomedeidae',
    'Procellariidae',
    'Phalacrocoracidae',
    'Fregatidae',
)
```

## 新的信息来源

英文名称使用 [ITIS](https://www.itis.gov/)，查询日期：2026-11-05。
通过 `searchByScientificName` 精确查找拉丁名，再用 `getFullRecordFromTSN` 读取 `commonNameList` 中语言为 `English` 的名称。
缺少直接英文名时，使用下级 ITIS 分类记录中的英文名称构造显示名。目、科英文名称、来源编号（TSN）和派生说明保存在 `data/metadata/cub_names.json`，加载器离线读取。

ITIS 记录链接格式：`https://www.itis.gov/servlet/SingleRpt/SingleRpt?search_topic=TSN&search_value=<TSN>`。

类别编号和层级关系来自 FreeGrainLearning，标签值由加载器的名称映射提供。

# iNat 英文名称

`data/metadata/inat21_names.json` 保存三个层级的英文名称及来源，加载器离线读取。
类别编号和父子关系沿用 FreeGrainLearning 标注。

- 物种：采用 iNaturalist 2021 官方 `common_name`，缺失项再查询 iNaturalist API（`locale=en`）。
- 目、科：采用 ITIS 的英文常用名；存在多个名称时选取最多三个，避免提示词过长。缺失时，从该分组内选择最多三个有英文名的代表物种构造名称。
- 无常用名的物种：参照同属、同科等最近可用类群，构造 `Species related to … (学名)`。这类名称是描述，不是正式常用名。
- 重名类别附加学名；目、科名称包含等级标记。派生名称的记录包含代表物种编号。

原始查询结果缓存在 `data/inat21/name_sources/`。使用已有缓存重新生成：

```bash
python data/metadata/build_inat_names.py
```

来源：[iNaturalist 2021 官方标注](https://ml-inat-competition-datasets.s3.amazonaws.com/2021/val.json.tar.gz)、
[iNaturalist API](https://api.inaturalist.org/v1/docs/)、[ITIS](https://www.itis.gov/)。