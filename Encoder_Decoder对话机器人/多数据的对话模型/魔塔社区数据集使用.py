from modelscope.msdatasets import MsDataset

_ds = MsDataset.load('Moemuu/Muice-Dataset')

print(_ds)

ds = _ds['train']

print(type(ds))

print(len(ds))

print(ds[0])

# 声明src和tgt 的最大长度
max_src = 0
max_tgt = 0

# 循环 1000 条数据
for i in range(len(ds)):
    d = ds[i]
    src = d['prompt']
    tgt = d['respond']
    # 统计 src 和 tgt 的最大长度
    if len(src) > max_src:
        max_src = len(src)
    if len(tgt) > max_tgt:
        max_tgt = len(tgt)

print(max_src)  # 47
print(max_tgt)  # 337

# 由于 tgt 长度过长，不利于训练，我们决定过滤掉长度大于 50 的 tgt 的数据
