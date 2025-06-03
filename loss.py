import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class Loss(nn.Module):
    def __init__(self, alpha, gamma):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCELoss()
        self.t = 0.5
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.gamma_neg = 0
        self.gamma_pos = 0
        self.clip = 0.05
        self.eps = 1e-9

    def weighted_mse_loss(self, mask_v, data, rec_v):
        n_view = len(rec_v)
        mse1 = mse2 = 0
        for i in range(n_view):
            mse1 += torch.mean(torch.pow(data[i].detach() - rec_v[i][:, i, :], 2) * mask_v[:, i:i + 1])
            for j in range(n_view):
                if i != j:
                    mse2 += torch.mean(
                        torch.pow(data[i].detach() - rec_v[i][:, j, :], 2) * mask_v[:, i:i + 1] * mask_v[:, j:j + 1])
        return self.alpha * mse1 / n_view + self.gamma * mse2 / n_view

    def weighted_contrastive_loss(self, v1, v2, we1, we2):
        mask_miss_inst = we1.mul(we2).bool()
        v1 = v1[mask_miss_inst]
        v2 = v2[mask_miss_inst]
        n = v1.size(0)
        N = 2 * n
        if n == 0:
            return 0
        v1 = F.normalize(v1, p=2, dim=1)
        v2 = F.normalize(v2, p=2, dim=1)
        z = torch.cat((v1, v2), dim=0)
        similarity_mat = torch.matmul(z, z.T) / self.t
        similarity_mat = similarity_mat.fill_diagonal_(0)
        label = torch.cat((torch.tensor(range(n, N)), torch.tensor(range(0, n)))).to(v1.device)
        loss = self.criterion(similarity_mat, label)
        return loss / N

    def weighted_bce_loss(self, label, pred, mask_l):
        return self.bce(pred * mask_l, label * mask_l)

    def asymmetricLoss(self, label, pred, mask_l):
        pred_pos = pred
        pred_neg = 1 - pred

        if self.clip is not None and self.clip > 0:
            pred_neg = (pred_neg + self.clip).clamp(max=1)

        los_pos = label * torch.log(pred_pos.clamp(min=self.eps)) * mask_l
        los_neg = (1 - label) * torch.log(pred_neg.clamp(min=self.eps)) * mask_l
        loss = los_pos + los_neg

        pt0 = pred_pos * label * mask_l
        pt1 = pred_neg * (1 - label) * mask_l  # pt = p if t > 0 else 1-p
        pt = pt0 + pt1
        one_sided_gamma = self.gamma_pos * label + self.gamma_neg * (1 - label)
        one_sided_w = torch.pow(1 - pt, one_sided_gamma)
        torch.set_grad_enabled(True)
        loss *= one_sided_w
        return -loss.sum()
