# ISIC_MM
### 3.29改动
add fold.csv:数据编号存储为fold.csv，范围0-14，在dataset.py中映射到0-4四个fold
在enet-type中添加fold+、fold++时会使用新的映射构造数据集
