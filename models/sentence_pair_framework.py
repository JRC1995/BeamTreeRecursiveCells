import torch as T
import torch.nn as nn
import torch.nn.functional as F
from models.layers import *

class sentence_pair_framework(nn.Module):
    def __init__(self, data, config):

        super(sentence_pair_framework, self).__init__()

        self.config = config
        self.classes_num = data["classes_num"] if data["classes_num"] > 2 else 1 # we do sigmoid when "classes_num" == 2
        embedding_data = data["embeddings"]
        self.pad_id = data["PAD_id"]
        self.unk_id = data["UNK_id"]

        self.out_dropout = config["out_dropout"]
        self.in_dropout = config["in_dropout"]
        self.hidden_size = config["hidden_size"]
        self.unk_embed = None

        self.ATT_PAD = T.tensor(-999999).float()
        self.zeros = T.tensor(0.0).float()

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

        self.prediction_linear1 = nn.Linear(4 * config["hidden_size"], config["classifier_hidden_size"])
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


    # %%
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

        sequence1 = batch["sequences1_vec"]
        sequence2 = batch["sequences2_vec"]
        input_mask1 = batch["input_masks1"]
        input_mask2 = batch["input_masks2"]
        if "parse_trees1" in batch:
            parse_trees1 = batch["parse_trees1"]
            parse_trees2 = batch["parse_trees2"]
        else:
            parse_trees1 = None
            parse_trees2 = None


        N = sequence1.size(0)

        # EMBEDDING BLOCK
        sequence1, input_mask1 = self.embed(sequence1, input_mask1)
        sequence2, input_mask2 = self.embed(sequence2, input_mask2)

        sequence1 = F.dropout(sequence1, p=self.in_dropout, training=self.training)
        sequence2 = F.dropout(sequence2, p=self.in_dropout, training=self.training)

        if "batch_pair" in self.config and self.config["batch_pair"]:
            pad = T.zeros(N, 1, self.hidden_size).float().to(sequence1.device)
            zero = T.zeros(N, 1).float().to(sequence1.device)

            max_s = max(sequence1.size(1), sequence2.size(1))

            while sequence1.size(1) < max_s:
                sequence1 = T.cat([sequence1, pad.clone()], dim=1)
                input_mask1 = T.cat([input_mask1, zero.clone()], dim=1)


            while sequence2.size(1) < max_s:
                sequence2 = T.cat([sequence2, pad.clone()], dim=1)
                input_mask2 = T.cat([input_mask2, zero.clone()], dim=1)

            sequence = T.cat([sequence1, sequence2], dim=0)
            input_mask = T.cat([input_mask1, input_mask2], dim=0)
            sequence_dict = self.encoder(sequence, input_mask)

            sequence1_dict = {}
            sequence2_dict = {}
            for key in sequence_dict:
                if sequence_dict[key] is None:
                    sequence1_dict[key] = None
                    sequence2_dict[key] = None
                else:
                    sequence1_dict[key] = sequence_dict[key][0:N]
                    sequence2_dict[key] = sequence_dict[key][N:]

        else:
            # ENCODER BLOCK
            if parse_trees1 is not None:
                sequence1_dict = self.encoder(sequence1, input_mask1, parse_trees1)
                sequence2_dict = self.encoder(sequence2, input_mask2, parse_trees2)
            else:
                sequence1_dict = self.encoder(sequence1, input_mask1)
                sequence2_dict = self.encoder(sequence2, input_mask2)

        sequence1 = sequence1_dict["sequence"]
        sequence2 = sequence2_dict["sequence"]
        input_mask1 = sequence1_dict["input_mask"]
        input_mask2 = sequence2_dict["input_mask"]

        aux_loss = None
        if "aux_loss" in sequence1_dict:
            aux_loss1 = sequence1_dict["aux_loss"]
            aux_loss2 = sequence2_dict["aux_loss"]
            if aux_loss1 is not None and aux_loss2 is not None:
                aux_loss = (aux_loss1 + aux_loss2) / 2
                aux_loss = aux_loss.mean()

        if "paths" in sequence1_dict:
            paths1 = sequence1_dict["paths"]
            path_scores1 = sequence1_dict["path_scores"]
            paths2 = sequence2_dict["paths"]
            path_scores2 = sequence2_dict["path_scores"]
        else:
            paths1 = None
            path_scores1 = None
            paths2 = None
            path_scores2 = None


        if self.config["global_state_return"]:
            global_state1 = sequence1_dict["global_state"]
            global_state2 = sequence2_dict["global_state"]
            if global_state1 is None or global_state2 is None:
                raise ValueError("Global State can not be None if 'global_state_return' is set as True")
        else:
            global_state1 = None
            global_state2 = None

        if self.config["global_state_only"]:
            feats1 = global_state1
            feats2 = global_state2
            if feats1 is None or feats2 is None:
                raise ValueError("Global State cannot be None if 'global_state_only' is set as True")
        else:
            feats1 = self.extract_features(sequence=sequence1, mask=input_mask1, global_state=global_state1)
            feats2 = self.extract_features(sequence=sequence2, mask=input_mask2, global_state=global_state2)

        feats = T.cat([feats1, feats2,
                       feats1 * feats2,
                       T.abs(feats1 - feats2)], dim=-1)

        feats = F.dropout(feats, p=self.out_dropout, training=self.training)
        intermediate = F.gelu(self.prediction_linear1(feats))
        intermediate = F.dropout(intermediate, p=self.out_dropout, training=self.training)
        logits = self.prediction_linear2(intermediate)

        assert logits.size() == (N, self.classes_num)



        return {"logits": logits, "aux_loss": aux_loss,
                "paths1": paths1, "paths2": paths2,
                "path_scores1": path_scores1, "path_scores2": path_scores2}


