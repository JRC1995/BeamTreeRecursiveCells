import torch as T
import torch.nn as nn
import torch.nn.functional as F
from models.layers import *


class classifier_framework(nn.Module):
    def __init__(self, data, config):

        super(classifier_framework, self).__init__()

        self.config = config
        self.out_dropout = config["out_dropout"]
        self.classes_num = data["classes_num"] if data["classes_num"] > 2 else 1
        embedding_data = data["embeddings"]
        self.pad_id = data["PAD_id"]
        self.unk_id = data["UNK_id"]

        self.out_dropout = config["out_dropout"]
        self.in_dropout = config["in_dropout"]
        self.hidden_size = config["hidden_size"]
        self.unk_embed = None

        ATT_PAD = -999999
        self.ATT_PAD = T.tensor(ATT_PAD).float()
        self.zeros = T.tensor(0.0)

        if embedding_data is not None:
            embedding_data = T.tensor(embedding_data)
            self.unk_embed = nn.Parameter(T.randn(embedding_data.size(-1)))
            self.word_embedding = nn.Embedding.from_pretrained(embedding_data,
                                                               freeze=config["word_embd_freeze"],
                                                               padding_idx=self.pad_id)
        else:
            vocab_len = data["vocab_len"]
            self.word_embedding = nn.Embedding(vocab_len, config["embd_dim"],
                                               padding_idx=self.pad_id)

        self.embd_dim = self.word_embedding.weight.size(-1)
        if self.config["initial_transform"]:
            if "input_size" in self.config:
                x = config["input_size"]
            else:
                x = config["hidden_size"]
            self.transform_linear = nn.Linear(self.embd_dim, x)

        encoder_fn = eval(config["encoder_type"])
        self.encoder = encoder_fn(config)


        if not config["global_state_only"]:
            self.attn_linear1 = nn.Linear(config["hidden_size"], config["hidden_size"])
            self.attn_linear2 = nn.Linear(config["hidden_size"], config["hidden_size"])
            if config["global_state_return"]:
                #self.compress_linear = nn.Linear(2 * config["hidden_size"], config["hidden_size"])
                self.global_score_linear1 = nn.Linear(2 * config["hidden_size"], config["hidden_size"])
                self.global_score_linear2 = nn.Linear(config["hidden_size"], config["hidden_size"])

        self.prediction_linear1 = nn.Linear(config["hidden_size"], config["classifier_hidden_size"])
        self.prediction_linear2 = nn.Linear(config["classifier_hidden_size"], self.classes_num)


    # %%

    def embed(self, sequence_idx, input_mask):

        N, S = sequence_idx.size()

        sequence = self.word_embedding(sequence_idx)

        if self.unk_id is not None and self.unk_embed is not None:
            sequence = T.where(sequence_idx.unsqueeze(-1) == self.unk_id,
                               self.unk_embed.view(1, 1, -1).repeat(N, S, 1),
                               sequence)

        assert sequence.size() == (N, S, self.embd_dim)

        if self.config["initial_transform"]:
            sequence = self.transform_linear(sequence)
        sequence = sequence * input_mask.view(N, S, 1)

        return sequence, input_mask


    def extract_features(self, sequence, mask, global_state=None):
        N, S, D = sequence.size()

        mask = mask.view(N, S, 1)

        attention_mask = T.where(mask == 0,
                                 self.ATT_PAD.to(mask.device),
                                 self.zeros.to(mask.device))

        assert attention_mask.size() == (N, S, 1)

        energy = self.attn_linear2(F.gelu(self.attn_linear1(sequence)))

        assert energy.size() == (N, S, D)

        attention = F.softmax(energy + attention_mask, dim=1)

        assert attention.size() == (N, S, D)

        attended_state = T.sum(attention * sequence, dim=1)

        assert attended_state.size() == (N, D)

        if global_state is not None:
            assert global_state.size() == (N, D)
            concated_state = T.cat([attended_state, global_state], dim=-1)
            a = T.sigmoid(self.global_score_linear2(F.gelu(self.global_score_linear1(concated_state))))
            out = (a * global_state) + ((1-a) * attended_state)
            return out
        else:
            return attended_state

    # %%
    def forward(self, batch):

        sequence = batch["sequences_vec"]
        input_mask = batch["input_masks"]
        if "parse_trees" in batch:
            parse_trees = batch["parse_trees"]
        else:
            parse_trees = None

        N = sequence.size(0)

        # EMBEDDING BLOCK
        sequence, input_mask = self.embed(sequence, input_mask)
        sequence = F.dropout(sequence, p=self.in_dropout, training=self.training)

        # ENCODER BLOCK
        if parse_trees is not None:
            sequence_dict = self.encoder(sequence, input_mask, parse_trees)
        else:
            sequence_dict = self.encoder(sequence, input_mask)

        sequence = sequence_dict["sequence"]
        input_mask = sequence_dict["input_mask"]
        if "beam_scores" in sequence_dict:
            beam_scores = sequence_dict["beam_scores"]
        else:
            beam_scores = None

        aux_loss = None
        if "aux_loss" in sequence_dict:
            aux_loss = sequence_dict["aux_loss"]
            if aux_loss is not None:
                aux_loss = aux_loss.mean()

        if self.config["global_state_return"]:
            global_state = sequence_dict["global_state"]
            if global_state is None:
                raise ValueError("Global State cannot be None if 'global_state_return' is set as True")
        else:
            global_state = None

        if not self.config["global_state_only"]:
            global_state = self.extract_features(sequence=sequence, mask=input_mask, global_state=global_state)

        feats = global_state

        feats = F.dropout(feats, p=self.out_dropout, training=self.training)
        feats = F.gelu(self.prediction_linear1(feats))
        feats = F.dropout(feats, p=self.out_dropout, training=self.training)
        logits = self.prediction_linear2(feats)

        """
        if "beam" in self.config["model_name"].lower():
            assert beam_scores is not None
            assert logits.size() == (N * self.config["beam_size"], self.classes_num)
            return {"logits": logits, "aux_loss": aux_loss, "beam_scores": beam_scores}
        else:
        """
        assert logits.size() == (N, self.classes_num)

        if "paths" in sequence_dict:
            paths = sequence_dict["paths"]
            path_scores = sequence_dict["path_scores"]
        else:
            paths = None
            path_scores = None

        return {"logits": logits, "aux_loss": aux_loss, "paths": paths, "path_scores": path_scores}