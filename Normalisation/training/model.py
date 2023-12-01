#from transformers import BertConfig, BertPreTrainedModel, BertTokenizer, BertModel
from transformers import AutoConfig
from transformers import AutoModelForPreTraining
from transformers import AutoTokenizer
from transformers import AutoModel
from transformers.modeling_utils import SequenceSummary
from torch import nn
import torch.nn.functional as F
import torch
from loss import AMSoftmax
from pytorch_metric_learning import losses, miners
from trans import TransE


class UMLSPretrainedModel(nn.Module):
    def __init__(self, device, model_name_or_path,
                 cui_label_count, rel_label_count, sty_label_count,
                 re_weight=1.0, sty_weight=0.1,
                 cui_loss_type="ms_loss",
                 trans_loss_type="TransE", trans_margin=1.0):
        super(UMLSPretrainedModel, self).__init__()

        self.device = device
        self.model_name_or_path = model_name_or_path
        if self.model_name_or_path.find("large") >= 0:
            self.feature_dim = 1024
        else:
            self.feature_dim = 768
        self.bert = AutoModel.from_pretrained(model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.dropout = nn.Dropout(0.1)

        self.rel_label_count = rel_label_count
        self.re_weight = re_weight

        self.sty_label_count = sty_label_count
        self.linear_sty = nn.Linear(self.feature_dim, self.sty_label_count)
        self.sty_loss_fn = nn.CrossEntropyLoss()
        self.sty_weight = sty_weight

        self.cui_loss_type = cui_loss_type
        self.cui_label_count = cui_label_count

        if self.cui_loss_type == "softmax":
            self.cui_loss_fn = nn.CrossEntropyLoss()
            self.linear = nn.Linear(self.feature_dim, self.cui_label_count)
        if self.cui_loss_type == "am_softmax":
            self.cui_loss_fn = AMSoftmax(
                self.feature_dim, self.cui_label_count)
        if self.cui_loss_type == "ms_loss":
            self.cui_loss_fn = losses.MultiSimilarityLoss(alpha=2, beta=50)
            self.miner = miners.MultiSimilarityMiner(epsilon=0.1)

        self.trans_loss_type = trans_loss_type
        if self.trans_loss_type == "TransE":
            self.re_loss_fn = TransE(trans_margin)
        self.re_embedding = nn.Embedding(
            self.rel_label_count, self.feature_dim)

        self.standard_dataloader = None

        self.sequence_summary = SequenceSummary(AutoConfig.from_pretrained(model_name_or_path)) # Now only used for XLNet

    def softmax(self, logits, label):
        loss = self.cui_loss_fn(logits, label)
        return loss

    def am_softmax(self, pooled_output, label):
        loss, _ = self.cui_loss_fn(pooled_output, label)
        return loss

    def ms_loss(self, pooled_output, label):
        pairs = self.miner(pooled_output, label)
        loss = self.cui_loss_fn(pooled_output, label, pairs)
        return loss

    def calculate_loss(self, pooled_output=None, logits=None, label=None):
        if self.cui_loss_type == "softmax":
            return self.softmax(logits, label)
        if self.cui_loss_type == "am_softmax":
            return self.am_softmax(pooled_output, label)
        if self.cui_loss_type == "ms_loss":
            return self.ms_loss(pooled_output, label)

    def get_sentence_feature(self, input_ids):
        # bert, albert, roberta
        if self.model_name_or_path.find("xlnet") < 0:
            outputs = self.bert(input_ids)
            pooled_output = outputs[1]
            return pooled_output

        # xlnet
        outputs = self.bert(input_ids)
        pooled_output = self.sequence_summary(outputs[0])
        return pooled_output


    # @profile
    def forward(self,
                input_ids_0, input_ids_1, input_ids_2,
                cui_label_0, cui_label_1, cui_label_2,
                sty_label_0, sty_label_1, sty_label_2,
                re_label):
        input_ids = torch.cat((input_ids_0, input_ids_1, input_ids_2), 0)
        cui_label = torch.cat((cui_label_0, cui_label_1, cui_label_2))
        sty_label = torch.cat((sty_label_0, sty_label_1, sty_label_2))
        #print(input_ids.shape, cui_label.shape, sty_label.shape)

        use_len = input_ids_0.shape[0]

        pooled_output = self.get_sentence_feature(
            input_ids)  # (3 * pair) * re_label
        logits_sty = self.linear_sty(pooled_output)
        sty_loss = self.sty_loss_fn(logits_sty, sty_label)

        if self.cui_loss_type == "softmax":
            logits = self.linear(pooled_output)
        else:
            logits = None
        cui_loss = self.calculate_loss(pooled_output, logits, cui_label)

        cui_0_output = pooled_output[0:use_len]
        cui_1_output = pooled_output[use_len:2 * use_len]
        cui_2_output = pooled_output[2 * use_len:]
        re_output = self.re_embedding(re_label)
        re_loss = self.re_loss_fn(
            cui_0_output, cui_1_output, cui_2_output, re_output)

        loss = self.sty_weight * sty_loss + cui_loss + self.re_weight * re_loss
        #print(sty_loss.device, cui_loss.device, re_loss.device)

        return loss, (sty_loss, cui_loss, re_loss)

    """
    def predict(self, input_ids):
        if self.loss_type == "softmax":
            return self.predict_by_softmax(input_ids)
        if self.loss_type == "am_softmax":
            return self.predict_by_amsoftmax(input_ids)        

    def predict_by_softmax(self, input_ids):
        pooled_output = self.get_sentence_feature(input_ids)
        logits = self.linear(pooled_output)
        return torch.max(logits, dim=1)[1], logits

    def predict_by_amsoftmax(self, input_ids):
        pooled_output = self.get_sentence_feature(input_ids)
        logits = self.loss_fn.predict(pooled_output)
        return torch.max(logits, dim=1)[1], logits
    """

    def init_standard_feature(self):
        if self.standard_dataloader is not None:
            for index, batch in enumerate(self.standard_dataloader):
                input_ids = batch[0].to(self.device)
                outputs = self.get_sentence_feature(input_ids)
                normalized_standard_feature = torch.norm(
                    outputs, p=2, dim=1, keepdim=True).clamp(min=1e-12)
                normalized_standard_feature = torch.div(
                    outputs, normalized_standard_feature)
                if index == 0:
                    self.standard_feature = normalized_standard_feature
                else:
                    self.standard_feature = torch.cat(
                        (self.standard_feature, normalized_standard_feature), 0)
            assert self.standard_feature.shape == (
                self.num_label, self.feature_dim), self.standard_feature.shape
        return None

    def predict_by_cosine(self, input_ids):
        pooled_output = self.get_sentence_feature(input_ids)

        normalized_feature = torch.norm(
            pooled_output, p=2, dim=1, keepdim=True).clamp(min=1e-12)
        normalized_feature = torch.div(pooled_output, normalized_feature)
        sim_mat = torch.matmul(normalized_feature, torch.t(
            self.standard_feature))  # batch_size * num_label
        return torch.max(sim_mat, dim=1)[1], sim_mat