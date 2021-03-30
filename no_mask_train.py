# coding: utf-8
from model.transformer import Transformer
import torch
import data.base_data as bd
from utils.dataset import Dataset
import os

save_folder = 'weights'
save_file = "no_mask_model.pkl"
step = 1000
total_loss = -1.
src_sequence_size = 8
tgt_sequence_size = 8


if __name__ == "__main__":
    dataset = Dataset(bd.en_dict, bd.cn_dict, bd.sentence_pair_demo, src_sequence_size, tgt_sequence_size)
    model = Transformer(src_vocab_size=len(bd.en_dict),
                        tgt_vocab_size=len(bd.cn_dict),
                        word_emb_dim=8,
                        tgt_sequence_size=8)
    loss_f = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train()
    for i in range(step):
        optimizer.zero_grad()
        src, tgt_in, tgt_out, _, _ = dataset.get_batch(batch_size=1)
        output = model(src, tgt_in)
        loss = loss_f(torch.log(output), tgt_out)
        if total_loss < 0:
            total_loss = loss.detach().numpy()
        else:
            total_loss = total_loss * 0.95 + loss.detach().numpy() * 0.05
        loss.backward()
        optimizer.step()
        if (i + 1) % 100 == 0:
            print("step: ", i+1, "loss:", total_loss)

    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    save_path = os.path.join(save_folder, save_file)
    torch.save(model, save_path)
    print(f"finished train ! model saved file: {save_path}")

