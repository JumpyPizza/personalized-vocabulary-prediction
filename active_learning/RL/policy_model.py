import torch
import torch.nn as nn
import torch.nn.functional as F

# TODO: gradient clipping, learning rate scheduling, and model initialization
# What else (in terms of common training tricks) do we need? 


############################
# no sequential sampling
############################
class ReinforceOneShot(nn.Module):
    def __init__(self, device = 'cuda', d_emb=768, 
                 n_heads=4, n_layers=4,
                 n_candidates=10000, K=30,
                 temp=1.0):
        super().__init__()
        # joint embedding for (y,p) in {0…3}
        self.lp_emb = nn.Embedding(4, d_emb).to(device) # label_pred_emb

        # decoder-only Transformer (no pos_emb!)
        layer = nn.TransformerDecoderLayer(d_emb, n_heads, dim_feedforward=2048)
        self.transformer = nn.TransformerDecoder(layer, num_layers=n_layers).to(device) # let's see how GPT forward is implemented?

        # single linear head to n_candidates
        self.to_logits = nn.Linear(d_emb, n_candidates).to(device)

        self.K    = K
        self.temp = temp
        self.device = device
        print("policy model is using device: ", self.device)

    def forward(self, sample_embs, labels, preds):
        sample_embs = sample_embs.to(self.device)
        labels = labels.to(self.device)
        preds = preds.to(self.device)
        # sample_embs: [M, d_emb], labels/preds: [M]
        M, d = sample_embs.shape # M initial samples 

        # 1) add joint label embedding
        pair_idx = labels * 2 + preds             # [M] in 0…3, 00->0, 01->1, 10->2, 11->3
        x = sample_embs + self.lp_emb(pair_idx)    # [M, d_emb]

        # 2) pass through decoder (no memory)
        #    treat “batch”=1, so need shape [M,1,d]
        tgt = x.unsqueeze(1)
        mem = torch.zeros(1, 1, d, device=x.device)
        out = self.transformer(tgt=tgt, memory=mem)  # [M,1,d]
        context = out.mean(dim=0).squeeze(0)          # [d_emb]

        # 3) logits
        logits = self.to_logits(context) / self.temp
        return logits  # [n_candidates]

    # def sample_subset(self, logits): # selection
    #     # Gumbel-TopK
    #     gumbels = -torch.empty_like(logits).exponential_().log()
    #     scores  = logits + gumbels
    #     topk    = scores.topk(self.K)
    #     idxs    = topk.indices                      # [K]
    #     logps   = logits.log_softmax(dim=0)[idxs]   # [K]
    #     return idxs, logps

    def sample_subset(self, logits, initial_ids): # updated selection: exclude initial_ids
        # Create a mask to avoid initial_ids
        mask = torch.ones_like(logits, dtype=torch.bool)
        mask[initial_ids] = False

        # Apply Gumbel noise only to valid (non-masked) logits
        gumbels = -torch.empty_like(logits).exponential_().log()
        gumbels[~mask] = float('-inf')  # Prevent masked indices from being selected

        # Add Gumbel noise to logits
        scores = logits + gumbels

        # Select top-K among valid entries
        topk = scores.topk(self.K)
        idxs = topk.indices  # [K]

        # Compute log-probs only for selected indices
        logps = logits.log_softmax(dim=0)[idxs]  # [K]
        return idxs, logps


class ActorCriticOneShot(nn.Module):
    def __init__(
        self,
        device = 'cuda',
        d_emb=768,
        n_heads=4,
        n_layers=4,
        n_candidates=10000,
        K=30,
        temperature=1.0
    ):
        super().__init__()
        # 1) label prediction embedding
        self.lp_emb = nn.Embedding(4, d_emb).to(device)

        # 2) decoder-only Transformer (no pos emb)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_emb, nhead=n_heads, dim_feedforward=4*d_emb, dropout=0.1
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=n_layers).to(device)

        # 3a) actor head
        self.to_logits = nn.Linear(d_emb, n_candidates).to(device)
        # 3b) critic head
        self.to_value  = nn.Linear(d_emb, 1).to(device)

        self.K           = K
        self.temperature = temperature
        self.device = device
        # init
        nn.init.xavier_uniform_(self.lp_emb.weight)
        nn.init.xavier_uniform_(self.to_logits.weight)
        nn.init.zeros_(self.to_logits.bias)
        nn.init.kaiming_uniform_(self.to_value.weight, a=0)
        nn.init.zeros_(self.to_value.bias)

    def forward(self, sample_embs, true_labels, pred_labels):
        """
        sample_embs: [M, d_emb]
        true_labels: [M], pred_labels: [M]
        returns:
          logits:  [n_candidates]
          value:   []         scalar
          context: [d_emb]    for debugging, if you like
        """
        sample_embs = sample_embs.to(self.device)
        true_labels = true_labels.to(self.device)
        pred_labels = pred_labels.to(self.device)
        M, d = sample_embs.shape
        pair_idx = true_labels * 2 + pred_labels     # in {0..3}
        x = sample_embs + self.lp_emb(pair_idx)      # [M, d_emb]

        # TransformerDecoder
        tgt    = x.unsqueeze(1)                      # [M,1,d_emb]
        memory = torch.zeros(1, 1, d, device=x.device)
        out    = self.transformer(tgt=tgt, memory=memory)  # [M,1,d_emb]

        # pool → context
        context = out.mean(dim=0).squeeze(0)         # [d_emb]

        # actor & critic
        logits = self.to_logits(context) / self.temperature  # [n_candidates]
        value  = self.to_value(context).squeeze(0)          # []

        return logits, value

    def sample_subset(self, logits, initial_ids): # updated selection: exclude initial_ids
        # Create a mask to avoid initial_ids
        mask = torch.ones_like(logits, dtype=torch.bool)
        mask[initial_ids] = False

        # Apply Gumbel noise only to valid (non-masked) logits
        gumbels = -torch.empty_like(logits).exponential_().log()
        gumbels[~mask] = float('-inf')  # Prevent masked indices from being selected

        # Add Gumbel noise to logits
        scores = logits + gumbels

        # Select top-K among valid entries
        topk = scores.topk(self.K)
        idxs = topk.indices  # [K]

        # Compute log-probs only for selected indices
        logps = logits.log_softmax(dim=0)[idxs]  # [K]
        return idxs, logps



######################################################
# sequential sampling, rewards only after the rollout
######################################################
def causal_mask(sz: int, device) -> torch.Tensor:
    # (sz × sz) mask with -inf above diagonal
    return torch.triu(torch.full((sz, sz), float('-inf'), device=device), diagonal=1)


class ReinforceSequential(nn.Module):
    def __init__(
        self,
        d_emb: int = 768,
        n_heads: int = 4,
        n_layers: int = 4,
        n_candidates: int = 10000,
        K: int = 30,
        temperature: float = 1.0,
    ):
        super().__init__()
        # joint embedding for (true_label, pred_label) → 0..3
        self.lp_emb = nn.Embedding(4, d_emb)
        self.d_emb = d_emb
        # 2) Decoder: autoregressive, cross-attend to encoder memory
        layer = nn.TransformerDecoderLayer(d_emb, n_heads, dim_feedforward=2048)
        self.transformer = nn.TransformerDecoder(layer, num_layers=n_layers)


        # 4) Final head to score all candidates
        self.to_logits = nn.Linear(d_emb, n_candidates)

        self.K           = K
        self.temperature = temperature
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.lp_emb.weight)
        nn.init.xavier_uniform_(self.to_logits.weight)
        nn.init.zeros_(self.to_logits.bias)
        # Transformer defaults are OK

    def forward(self, previous_states, previous_true_labels, previous_pred_labels,
                current_state, current_true_label, current_pred_label):

        if current_state.numel() == 0:
            x = previous_states                     # [M, d_emb]
            lp = previous_true_labels * 2 + previous_pred_labels
        else:
            x = torch.cat([previous_states, current_state], dim=0)        # [M+t, d_emb]
            lp = torch.cat([
                previous_true_labels * 2 + previous_pred_labels,
                current_true_label * 2 + current_pred_label
            ], dim=0)          
                                           # [M+t]
         # 2) add label, prediction information
        x = x + self.lp_emb(lp)                            # [M+t, d_emb]

        # 3) causal transformer stack
        #    we treat "batch"=1, so add an axis:
        x = x.unsqueeze(1)                                   # [M+t,1,d_emb]
            
        mem = torch.zeros(1, 1, self.d_emb, device=x.device)
        out = self.transformer(tgt=x, memory=mem)
        # 4) use last token’s representation to score
        # or should we use mean pooling? 
        context = out[-1]                                          # [d_emb]
        logits = self.to_logits(context) / self.temperature      # [n_candidates]
        return logits

class TransformerPolicySequential(nn.Module):
    def __init__(self, d_emb, n_candidates, n_heads=4,  n_layers=4,  temperature=1.0):
        super().__init__()
        self.d_emb = d_emb

        self.lp_emb = nn.Embedding(4, d_emb)
        self.d_emb = d_emb
        # 2) Decoder: autoregressive, cross-attend to encoder memory
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_emb,
            nhead=n_heads,
            dim_feedforward=2048,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)


        # 4) Final head to score all candidates
        self.to_logits = nn.Linear(d_emb, n_candidates)
        self.to_value  = nn.Linear(d_emb, 1)

        self.temperature = temperature
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.lp_emb.weight)
        nn.init.xavier_uniform_(self.to_logits.weight)
        nn.init.zeros_(self.to_logits.bias)
        nn.init.kaiming_uniform_(self.to_value.weight, a=0)
        nn.init.zeros_(self.to_value.bias)

    def forward(self, previous_states, previous_true_labels, previous_pred_labels,
                current_state, current_true_label, current_pred_label):
        if current_state.numel() == 0:
            x = previous_states                     # [M, d_emb]
            lp = previous_true_labels * 2 + previous_pred_labels
        else:
            x = torch.cat([previous_states, current_state], dim=0)        # [M+t, d_emb]
            lp = torch.cat([
                previous_true_labels * 2 + previous_pred_labels,
                current_true_label * 2 + current_pred_label
            ], dim=0)         
        x = x + self.lp_emb(lp)
        # x = x.unsqueeze(1) # [M+t, 1, d_emb]
        x = x.unsqueeze(0) # [1, M+t, d_emb], batch first
        out = self.transformer(x)
        # context = out[-1].squeeze(0) # [d_emb]
        context =  out[0, -1] # [ d_emb]
        # actor & critic
        logits = self.to_logits(context) / self.temperature  # [n_candidates]
        value  = self.to_value(context).squeeze()          # []

        return logits, value

######################################################
# sequential, how to adjust policy at each step based on the actual label? 
######################################################






