import torch
import torch.nn as nn


class AMSoftmax(nn.Module):
    def __init__(self,
                 in_feats,
                 n_classes=10,
                 m=0.35,
                 s=30):
        super(AMSoftmax, self).__init__()
        self.m = m
        self.s = s
        self.in_feats = in_feats
        self.W = torch.nn.Parameter(torch.randn(in_feats, n_classes), requires_grad=True)
        self.ce = nn.CrossEntropyLoss()
        nn.init.xavier_normal_(self.W, gain=1)

    def forward(self, x, label):
        #print(x.shape, lb.shape, self.in_feats)
        #assert x.size()[0] == label.size()[0]
        #assert x.size()[1] == self.in_feats
        x_norm = torch.norm(x, p=2, dim=1, keepdim=True).clamp(min=1e-12)
        x_norm = torch.div(x, x_norm)
        w_norm = torch.norm(self.W, p=2, dim=0, keepdim=True).clamp(min=1e-12)
        w_norm = torch.div(self.W, w_norm)
        costh = torch.mm(x_norm, w_norm)
        # print(x_norm.shape, w_norm.shape, costh.shape)
        lb_view = label.view(-1, 1).to(x.device)
        delt_costh = torch.zeros(costh.size()).to(x.device).scatter_(1, lb_view, self.m)
        costh_m = costh - delt_costh
        costh_m_s = self.s * costh_m
        loss = self.ce(costh_m_s, label)
        return loss, costh_m_s

    def predict(self, x):
        x_norm = torch.norm(x, p=2, dim=1, keepdim=True).clamp(min=1e-12)
        x_norm = torch.div(x, x_norm)
        w_norm = torch.norm(self.W, p=2, dim=0, keepdim=True).clamp(min=1e-12)
        w_norm = torch.div(self.W, w_norm)
        costh = torch.mm(x_norm, w_norm)
        return costh

class MultiSimilarityLoss(nn.Module):
    def __init__(self):
        super(MultiSimilarityLoss, self).__init__()
        self.thresh = 0.5
        self.margin = 0.1

        self.scale_pos = 2.0
        self.scale_neg = 50.0

    def forward(self, feats, labels):
        #assert feats.size(0) == labels.size(0), \
        #    f"feats.size(0): {feats.size(0)} is not equal to labels.size(0): {labels.size(0)}"
        batch_size = feats.size(0)

        # Feature normalize
        x_norm = torch.norm(feats, p=2, dim=1, keepdim=True).clamp(min=1e-12)
        x_norm = torch.div(feats, x_norm)

        sim_mat = torch.matmul(x_norm, torch.t(x_norm))

        epsilon = 1e-5
        loss = []

        #unique_label, inverse_indices = torch.unique_consecutive(labels, return_inverse=True)

        for i in range(batch_size):
            pos_pair_ = sim_mat[i][labels == labels[i]]
            pos_pair_ = pos_pair_[pos_pair_ < 1 - epsilon]
            neg_pair_ = sim_mat[i][labels != labels[i]]

            #print(pos_pair_)
            #print(neg_pair_)
           
            if len(neg_pair_) >= 1:
                pos_pair = pos_pair_[pos_pair_ - self.margin < max(neg_pair_)]
                if len(pos_pair) >= 1:
                    pos_loss = 1.0 / self.scale_pos * torch.log(
                        1 + torch.sum(torch.exp(-self.scale_pos * (pos_pair - self.thresh))))
                    loss.append(pos_loss)

            if len(pos_pair_) >= 1:
                neg_pair = neg_pair_[neg_pair_ + self.margin > min(pos_pair_)]
                if len(neg_pair) >= 1:
                    neg_loss = 1.0 / self.scale_neg * torch.log(
                        1 + torch.sum(torch.exp(self.scale_neg * (neg_pair - self.thresh))))
                    loss.append(neg_loss)

        #print(labels, len(loss))
        if len(loss) == 0:
            return torch.zeros([], requires_grad=True).to(feats.device)

        loss = sum(loss) / batch_size
        return loss

if __name__ == '__main__':
    criteria = AMSoftmax(20, 5)
    a = torch.randn(10, 20)
    lb = torch.randint(0, 5, (10, ), dtype=torch.long)
    loss = criteria(a, lb)
    loss.backward()

    print(loss.detach().numpy())
    print(list(criteria.parameters())[0].shape)
    print(type(next(criteria.parameters())))
    print(lb)
