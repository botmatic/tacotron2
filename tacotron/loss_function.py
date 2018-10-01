from torch import nn
import torch


class Tacotron2Loss(nn.Module):
    def __init__(self, hparams):
        super(Tacotron2Loss, self).__init__()
        self.use_guided_attention = hparams.use_guided_attention
        self.guided_attention_g = hparams.guided_attention_g

    def forward(self, model_output, targets):
        mel_target, gate_target = targets[0], targets[1]
        mel_target.requires_grad = False
        gate_target.requires_grad = False
        gate_target_orig = gate_target
        gate_target = gate_target.view(-1, 1)

        mel_out, mel_out_postnet, gate_out, alignments = model_output
        gate_out = gate_out.view(-1, 1)
        mel_loss = nn.MSELoss()(mel_out, mel_target) + \
            nn.MSELoss()(mel_out_postnet, mel_target)
        gate_loss = nn.BCEWithLogitsLoss()(gate_out, gate_target)

        loss = mel_loss + gate_loss

        if self.use_guided_attention:
            _, length1, length2 = alignments.shape
            i1 = torch.arange(length1).float().cuda().unsqueeze(0)
            i2 = torch.arange(length2).float().cuda().unsqueeze(0)

            length1 = length1 - gate_target_orig.sum(1)
            i1_ratio = i1 / length1.unsqueeze(1)
            length2 = length2 - ((alignments.sum(1) == 0).sum(1).float() - 1)
            i2_ratio = i2 / length2.unsqueeze(1)
            d1 = i1 <= length1.unsqueeze(1)
            d2 = i2 <= length2.unsqueeze(1)
            mask = d1.unsqueeze(2) * d2.unsqueeze(1)

            i1_ratio = i1_ratio.unsqueeze(2)
            i2_ratio = i2_ratio.unsqueeze(1)
            w = 1 - torch.exp(-(i1_ratio - i2_ratio) ** 2 / (2 * (self.guided_attention_g ** 2)))
            
            w[~mask] = 0

            guided_attention_loss = alignments * w.unsqueeze(0)
            loss += guided_attention_loss.sum()

        return loss
