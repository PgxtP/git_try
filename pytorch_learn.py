import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

PAD_ID = 0
BOS_ID = 2
EOS_ID = 3

SAMPLE_TOKEN_PAIRS = [
    (torch.tensor([11, 12, 13, EOS_ID, PAD_ID, PAD_ID]), torch.tensor([BOS_ID, 21, 22, EOS_ID, PAD_ID, PAD_ID])),
    (torch.tensor([14, 15, EOS_ID, PAD_ID, PAD_ID, PAD_ID]), torch.tensor([BOS_ID, 23, 24, 25, EOS_ID, PAD_ID])),
    (torch.tensor([16, 17, 18, 19, EOS_ID, PAD_ID]), torch.tensor([BOS_ID, 26, 27, 28, EOS_ID, PAD_ID])),
    (torch.tensor([20, 21, 22, EOS_ID, PAD_ID, PAD_ID]), torch.tensor([BOS_ID, 29, 30, EOS_ID, PAD_ID, PAD_ID])),
]


class TranslationDataset(Dataset):
    def __init__(self,samples):
        self.samples=samples
    def __len__(self):
        return len(self.samples)
    def __getitem__(self,num_place):
        return self.samples[num_place]
    def dataloader(self,num_place):
        print (self.samples[num_place],(self.__len__(),len(self.samples[num_place])))
dataset = TranslationDataset(SAMPLE_TOKEN_PAIRS)
assert len(dataset) == len(SAMPLE_TOKEN_PAIRS)
assert torch.equal(dataset[0][0], SAMPLE_TOKEN_PAIRS[0][0])
assert torch.equal(dataset[0][1], SAMPLE_TOKEN_PAIRS[0][1])
print("Dataset checks passed.")

loader = DataLoader(dataset, batch_size=2, shuffle=False)
src_batch, tgt_batch = next(iter(loader))

print("src_batch:", src_batch)
print("src shape:", src_batch.shape)
print("tgt_batch:", tgt_batch)
print("tgt shape:", tgt_batch.shape)


# 以下是训练流程练习所需的固定环境，由 Codex 准备。
torch.manual_seed(42)
train_dataset = TranslationDataset(SAMPLE_TOKEN_PAIRS[:3])
val_dataset = TranslationDataset(SAMPLE_TOKEN_PAIRS[3:])
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)


class TinyTokenTranslator(nn.Module):
    def __init__(self, src_vocab_size=32, tgt_vocab_size=32, d_model=8):
        super().__init__()
        self.embedding = nn.Embedding(src_vocab_size, d_model)
        self.output_layer = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src):
        return self.output_layer(self.embedding(src))


model = TinyTokenTranslator()
loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_ID)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


def compute_batch_loss(current_model, src, tgt):
    logits = current_model(src)
    return loss_fn(logits.transpose(1, 2), tgt)


# TODO：由学习者在此处编写一个 epoch 的训练流程和验证流程。
model.train()
train_loss_sum = 0.0

for src, tgt in train_loader:
    optimizer.zero_grad()

    loss = compute_batch_loss(model, src, tgt)

    loss.backward()
    optimizer.step()

    train_loss_sum += loss.item()

average_train_loss = train_loss_sum / len(train_loader)
print("average train loss:", average_train_loss)
model.eval()
val_loss_sum = 0.0

with torch.no_grad():
    for src, tgt in val_loader:
        loss = compute_batch_loss(model, src, tgt)
        val_loss_sum += loss.item()

average_val_loss = val_loss_sum / len(val_loader)
print("average val loss:", average_val_loss)