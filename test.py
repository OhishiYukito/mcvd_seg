import torch

# Collate fn for n repeats
def my_collate(batch):
    data, _ = zip(*batch)
    data = torch.stack(data).repeat_interleave(preds_per_test, dim=0)
    return data, torch.zeros(len(data))

dataset = 
dataloader = Dataloader(dataset= , collate_fn= my_collate)


# 評価するときは、1つの正解データに対してpreds_per_test回の予測を行う。
# そして、その平均値をとる。
# なので、正解データが5個であれば、モデルは5*preds_per_test回サンプリングを行い、その平均がスコアとなる。
for batch in dataloader:
    