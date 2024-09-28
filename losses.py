import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb


class ATLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, labels,P_weight=1.0):
        if (torch.isnan(logits).any() or torch.isinf(logits).any() or torch.isnan(labels).any() or torch.isinf(labels).any()):
            pdb.set_trace()
        # TH label
        th_label = torch.zeros_like(labels, dtype=torch.float).to(labels)
        th_label[:, 0] = 1.0
        labels[:, 0] = 0.0

        p_mask = labels + th_label
        n_mask = 1 - labels

        # Rank positive classes to TH
        logit1 = logits - (1 - p_mask) * 1e30
        loss1 = -(F.log_softmax(logit1, dim=-1) * labels).sum(1)

        # Rank TH to negative classes
        logit2 = logits - (1 - n_mask) * 1e30
        loss2 = -(F.log_softmax(logit2, dim=-1) * th_label).sum(1)

        if (torch.isnan(loss1).any() or torch.isinf(loss1).any() or torch.isnan(loss2).any() or torch.isinf(loss2).any()):
            pdb.set_trace()

        # Sum two parts
        loss = P_weight * loss1 + loss2
        loss = loss.mean()
        return loss

    def get_label(self, logits, num_labels=-1):
        th_logit = logits[:, 0].unsqueeze(1)
        output = torch.zeros_like(logits).to(logits)
        mask = (logits > th_logit)
        if num_labels > 0:
            top_v, _ = torch.topk(logits, num_labels, dim=1)
            top_v = top_v[:, -1]
            mask = (logits >= top_v.unsqueeze(1)) & mask
        output[mask] = 1.0
        output[:, 0] = (output.sum(1) == 0.).to(logits)
        return output

class ATLoss11(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, labels):
        # TH label
        th_label = torch.zeros_like(labels, dtype=torch.float).to(labels)
        th_label[:, 0] = 1.0
        labels[:, 0] = 0.0

        p_mask = labels + th_label
        n_mask = 1 - labels

        # Rank positive classes to TH
        logit1 = logits*(-1)
        logit1 = logit1 - (1 - p_mask) * 1e30
        loss1 = -(F.log_softmax(logit1, dim=-1) * th_label).sum(1)

        # Rank TH to negative classes
        logit2 = logits - (1 - n_mask) * 1e30
        loss2 = -(F.log_softmax(logit2, dim=-1) * th_label).sum(1)

        # Sum two parts
        loss = loss1 + loss2
        loss = loss.mean()
        return loss*100

    def get_label(self, logits, num_labels=-1):
        th_logit = logits[:, 0].unsqueeze(1)
        output = torch.zeros_like(logits).to(logits)
        mask = (logits > th_logit)
        if num_labels > 0:
            top_v, _ = torch.topk(logits, num_labels, dim=1)
            top_v = top_v[:, -1]
            mask = (logits >= top_v.unsqueeze(1)) & mask
        output[mask] = 1.0
        output[:, 0] = (output.sum(1) == 0.).to(logits)
        return output


class Focal_loss(nn.Module):
    def __init__(self,gamma=2.0,alpha=0.15):
        super().__init__()
        self.alpha=alpha
        self.gamma=gamma


    def forward(self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ):
      """
      Args:
          inputs: A float tensor of arbitrary shape.
                  The predictions for each example.
          targets: A float tensor with the same shape as inputs. Stores the binary
                  classification label for each element in inputs
                  (0 for the negative class and 1 for the positive class).
          mask:
          alpha: (optional) Weighting factor in range (0,1) to balance
                  positive vs negative examples or -1 for ignore. Default = 0.25
          gamma: Exponent of the modulating factor (1 - p_t) to
                 balance easy vs hard examples.
          reduction: 'none' | 'mean' | 'sum'
                   'none': No reduction will be applied to the output.
                   'mean': The output will be averaged.
                   'sum': The output will be summed.
      Returns:
          Loss tensor with the reduction option applied.
      """

      ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
      p = torch.sigmoid(inputs)
      p_t = p * targets + (1 - p) * (1 - targets)
      loss = ce_loss * ((1 - p_t.detach()) ** self.gamma)
     
      if self.alpha >= 0:
        alpha_t = (1 - self.alpha) * targets + self.alpha * (1 - targets)
        loss = alpha_t * loss
      return loss.mean()