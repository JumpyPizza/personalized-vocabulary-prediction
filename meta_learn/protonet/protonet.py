import sys
sys.path.append("../meta_learn")
from bert_model import BertTokenTask   

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
import numpy as np
from tqdm import tqdm

def l2n(x: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    # L2 Normalization 
    return x / (x.norm(dim=dim, keepdim=True) + eps)

def first_labeled_index(labels_row: torch.Tensor) -> Optional[int]:
    # find the idx that is non zero in labels, this is the idx for the token 
    idx = torch.nonzero(labels_row != -100, as_tuple=False)
    if idx.numel() == 0:
        return None
    return int(idx[0].item())

#########################
# Encode the tokens
#########################
class BertTokenEncoder(nn.Module):
    """
    Returns per-token projected vectors [B, T, D].
    Use last hidden states by default; optional scalar-mix over layers.
    """
    def __init__(self, model_name: str, proj_dim: int = 128, dropout: float = 0.1, use_scalar_mix: bool = False):
        super().__init__()
        if not torch.cuda.is_available():
            raise ValueError("No GPU found")
        self.backbone = AutoModel.from_pretrained(model_name, output_hidden_states=use_scalar_mix)
        self.use_scalar_mix = use_scalar_mix
        H = self.backbone.config.hidden_size
        if use_scalar_mix:
            L = self.backbone.config.num_hidden_layers + 1  # embeddings + each layer
            self.layer_weights = nn.Parameter(torch.zeros(L))  # learned scalar mix (softmaxed)

        self.proj = nn.Sequential(
            nn.Linear(H, proj_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.out_dim = proj_dim  # D for downstream

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        if self.use_scalar_mix:
            # hidden_states: tuple len L, each [B,T,H]
            hs = torch.stack(out.hidden_states, dim=0)             # [L,B,T,H]
            weights = torch.softmax(self.layer_weights, dim=0)     # [L]
            token_reps = (weights.view(-1, 1, 1, 1) * hs).sum(0)   # [B,T,H]
        else:
            token_reps = out.last_hidden_state                     # [B,T,H]

        z = self.proj(token_reps)                                  # [B,T,D]
        return z

#########################
# Each user is regarded as a task
# each user is embedded and get a user embedding e_u
#########################
class LabelProjector(nn.Module):
    # an embedding layer to encode token labels
    def __init__(self, num_classes: int, label_dim: int = 64):
        super().__init__()
        self.emb = nn.Embedding(num_classes, label_dim)
        nn.init.xavier_uniform_(self.emb.weight)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        return self.emb(y)  # [N=len(y), label_dim]


class UserEmbedder(nn.Module):
    """
    Multi-class user embedding.
    Suppose user profile is represented by the small set of support samples, 
    the labels distinguish different users 

    Builds e_u from:
      - class prototypes in token space: prototypes [C, D] C = class_num, D = dim
      - centered deltas: Δ_c = μ_c - mean_present(μ)
      - concat token vec and label emb [v_i ; label_emb(y_i)] -> s
      - final: e_u = MLP([ vec(prototypes) ; vec(deltas) ; s ])

    Notes:
      - If a class is missing in support, its prototype is zeros here; the mean_present is computed
        over PRESENT classes only, so missing classes get Δ_c = -mean_present.
    """
    def __init__(
        self,
        embed_dim: int,
        num_classes: int = 2,
        label_dim: int = 64,
        user_hidden: int = 256,
        user_dim: int = 128,
        dropout: float = 0.1
    ):
        super().__init__()
        self.num_classes = num_classes
        self.label_proj = LabelProjector(num_classes, label_dim)

        # DeepSets /phi to pool (vec, label_emb) pairs into s
        self.phi = nn.Sequential(
            nn.Linear(embed_dim + label_dim, user_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(user_hidden, user_dim),
            nn.ReLU(inplace=True),
        )

        # MLP over concatenated [vec(prototypes); vec(deltas); s]
        concat_dim = (num_classes * embed_dim) + (num_classes * embed_dim) + user_dim
        self.mlp = nn.Sequential(
            nn.Linear(concat_dim, 2 * user_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(2 * user_dim, user_dim)
        )
        self.out_dim = user_dim

    def forward(
        self,
        vecs: torch.Tensor,          # [N, D] (support token vectors)
        y: torch.Tensor,             # [N]    (class ids) 
        prototypes: torch.Tensor,    # [C, D] (class prototypes; zeros if missing)
        present: torch.Tensor        # [C] bool (which classes are present in support)
    ) -> torch.Tensor:
        C, D = prototypes.shape

        # DeepSets pooling over (vec, label_emb)
        lab = self.label_proj(y)                      # [N, L]
        h   = self.phi(torch.cat([vecs, lab], -1))    # [N, U]
        s   = h.mean(0, keepdim=True)                 # [1, U]

        # Centered deltas: Δ_c = μ_c - mean_present(μ)
        if present.any():
            mean_present = prototypes[present].mean(0)     # [D]
        else:
            mean_present = torch.zeros(D, device=prototypes.device)
        deltas = prototypes - mean_present.unsqueeze(0)    # [C, D]

        # Normalize per-part to keep scales balanced
        protos_flat = l2n(prototypes).reshape(1, C * D)    # [1, C*D]
        deltas_flat = l2n(deltas).reshape(1, C * D)        # [1, C*D]
        s_norm      = l2n(s)                                # [1, U]

        z = torch.cat([protos_flat, deltas_flat, s_norm], dim=-1)  # [1, 2*C*D + U]
        e_u = l2n(self.mlp(z))                                     # [1, U]
        return e_u


class UserProtoMemory:
    """
    Save and update user embeddings during training; 
    Search for similar users based on user embeddings 
    """
    def __init__(self, momentum: float = 0.9, device: str = "cuda"):
        self.m = momentum
        self.device = device
        self.user_e: Dict[str, torch.Tensor] = {}   # [user_dim]
        self.user_mu: Dict[str, torch.Tensor] = {}  # [C, proto_dim]

    @torch.no_grad()
    def update(self, user_id: str, e_u: torch.Tensor, prototypes: torch.Tensor):
        # e_u: [1,U], prototypes: [C,D]
        eu = e_u.squeeze(0).detach().to(self.device)
        mu = prototypes.detach().to(self.device)
        if user_id not in self.user_e:
            self.user_e[user_id] = eu
            self.user_mu[user_id] = mu
            return
        self.user_e[user_id] = self.m * self.user_e[user_id] + (1 - self.m) * eu
        self.user_mu[user_id] = self.m * self.user_mu[user_id] + (1 - self.m) * mu

    @torch.no_grad()
    def neighbors(self, user_id: str, e_u: torch.Tensor, k: int = 5) -> List[Tuple[str, float]]:
        if not self.user_e:
            return []
        keys = list(self.user_e.keys())
        E = torch.stack([self.user_e[k] for k in keys], dim=0)  # [U, Du]
        sims = (l2n(e_u) @ l2n(E).T).squeeze(0)                 # [U]
        if user_id in keys:
            sims[keys.index(user_id)] = -1e9
        topk = torch.topk(sims, k=min(k, len(keys))).indices.tolist()
        return [(keys[i], float(sims[i].item())) for i in topk]


# ProtoNet 
@dataclass
class UserProtoConfig:
    num_classes: int = 2
    proj_dim: int = 128            # token embedding dim D
    label_dim: int = 64
    user_dim: int = 128
    encoder_lr: float = 2e-5
    embedder_lr: float = 1e-4
    weight_decay: float = 0.0
    dropout: float = 0.1
    distance: str = "cosine"       # for prototype logits
    tau: float = 0.9               # neighbor softmax temp
    lam: float = 0.7               # own vs neighbors mix  
    max_support_subbatch: int = 256
    max_query_subbatch: int = 512
    # beta_consistency: float = 0.1
    # margin: float = 0.2
    k_neighbors: int = 10
    use_scalar_mix: bool = False


class ProtoNetWrapper(nn.Module):
    """
    User-as-task prototypical meta-learner
      - Build class prototypes /mu from support (embedding-only) for C classes 
      - Build label-aware user embedding e_u 
      - Mix in neighbor prototypes via memory (attention over neighbor users)
      - Train with CE + consistency (between two supports from the same user) 
    """
    def __init__(self, model_name: str, cfg: UserProtoConfig = None, wandb_run = None):
        super().__init__()
        if not torch.cuda.is_available():
            raise ValueError("No GPU found")
        if cfg is None: 
            print("cfg initialized from default")
            cfg = UserProtoConfig()

        encoder = BertTokenEncoder(
            model_name=model_name,
            proj_dim=cfg.proj_dim,
            dropout=cfg.dropout,
            use_scalar_mix=cfg.use_scalar_mix
        )

        self.encoder = nn.DataParallel(encoder).to("cuda:0") if torch.cuda.device_count() > 1 else encoder.to("cuda")
        self.embedder = UserEmbedder(
            embed_dim=cfg.proj_dim,
            num_classes=cfg.num_classes,
            label_dim=cfg.label_dim,
            user_hidden=2 * cfg.user_dim,
            user_dim=cfg.user_dim,
            dropout=cfg.dropout
        ).to(next(self.encoder.parameters()).device)

        # two-param-group optimizer
        self.optimizer = torch.optim.AdamW([
            {"params": self._enc().parameters(), "lr": cfg.encoder_lr},
            {"params": self.embedder.parameters(), "lr": cfg.embedder_lr},
        ], weight_decay=cfg.weight_decay)

        self.cfg = cfg
        self.wandb_run = wandb_run
        self.memory = UserProtoMemory(momentum=0.9, device=self.device)

    @property
    def device(self):
        return next(self.encoder.parameters()).device

    def _enc(self) -> BertTokenEncoder:
        return self.encoder.module if isinstance(self.encoder, nn.DataParallel) else self.encoder

    
    def _token_vecs_labels(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract a single token vector (the labeled token) and its label from each sequence in the batch.
        Returns:
            V: [N, D], y: [N]
        """
        input_ids = batch["input_ids"].to(self.device)
        attn = batch["attention_mask"].to(self.device)
        labels = batch["labels"].to(self.device)  # mapped labels with -100 for ignored
   
        Z = self.encoder(input_ids, attn)
       

        vecs, cls = [], []
        B, T, D = Z.shape
        for b in range(B):
            j = first_labeled_index(labels[b]) # get the idx of the target token in this batch
            if j is None:
                continue
            vecs.append(Z[b, j]) # get the token embedding of the target token 
            cls.append(int(labels[b, j].item()))  # get the label for the target token
        if not vecs:
            raise RuntimeError("no valid token embedding in batch")
            # return torch.empty(0, self.cfg.proj_dim, device=self.device), torch.empty(0, dtype=torch.long, device=self.device)
        V = l2n(torch.stack(vecs, dim=0))  # [N,D] 
        y = torch.tensor(cls, dtype=torch.long, device=self.device) #[N]
        return V, y

    def _prototypes(self, vecs: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute per-class prototypes μ_c and presence mask.
        Returns:
            prototypes: [C,D], present: [C] bool
        """
        C, D = self.cfg.num_classes, self.cfg.proj_dim
        prototypes = torch.zeros(C, D, device=self.device)
        present = torch.zeros(C, dtype=torch.bool, device=self.device)
        for c in range(C):
            m = (y == c)
            if m.any():
                prototypes[c] = vecs[m].mean(0)
                present[c] = True
        # if not present.all():
        #     raise RuntimeError("require all the classes to be present")
        return prototypes, present

    def _logits(self, Q: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
        if self.cfg.distance == "cosine":
            return l2n(Q) @ l2n(P).T
        elif self.cfg.distance == "euclidean":
            q2 = (Q ** 2).sum(1, keepdim=True)
            p2 = (P ** 2).sum(1, keepdim=True).T
            cross = Q @ P.T
            d2 = q2 + p2 - 2 * cross
            return -d2
        else:
            raise ValueError(f"Unknown distance: {self.cfg.distance}")

    # Meta update 
    def meta_update(self, tasks: List, support_batch_size: int, query_batch_size: int) -> Dict[str, float]:
        """
        One meta-iteration over a list of user tasks (BertTokenTask).
        Draw two supports for consistency; build user embedding; mix neighbor prototypes; compute losses.
        """
        self.train()
        self.optimizer.zero_grad()

        ce_sum = cons_sum = ctr_sum = 0.0
        used = 0

        for task in tasks:
            uid = getattr(task, "task_id", None)
            if uid is None:
                uid = getattr(task, "user_name", str(id(task)))
            if uid is None:
                raise RunTimeError("No user ID")

            # two supports (A for memory + mixing, B for consistency + queries)
            sup_a, _, _ = task.get_data(support_batch_size=support_batch_size, query_batch_size=query_batch_size, ensure_non_zero=True)
            sup_b, qry, _ = task.get_data(support_batch_size=support_batch_size, query_batch_size=query_batch_size, ensure_non_zero=True)

            Sa, ya = self._token_vecs_labels(sup_a)
            Sb, yb = self._token_vecs_labels(sup_b)
            Q,  yq = self._token_vecs_labels(qry)

            if Sa.numel() == 0 or Sb.numel() == 0 or Q.numel() == 0:
                continue

            prot_a, pres_a = self._prototypes(Sa, ya)   # [C,D], [C]
            prot_b, pres_b = self._prototypes(Sb, yb)

            # user embedding from A (multi-class)
            e_u = self.embedder(Sa, ya, prot_a, pres_a)  # [1, U]

            # recored user embedding
            self.memory.update(uid, e_u, prot_a)

            # get k neighbors 
            neigh = self.memory.neighbors(uid, e_u, k=self.cfg.k_neighbors)
            if len(neigh) > 0:
                keys, sims = zip(*neigh)
                sims_t = torch.tensor(sims, device=self.device).view(1, -1)
                alpha = torch.softmax(sims_t / self.cfg.tau, dim=-1).squeeze(0)  # [K]
                neigh_mu = torch.stack([self.memory.user_mu[k] for k in keys], dim=0)  # [K,C,D]
                prot_neigh = (alpha.view(-1, 1, 1) * neigh_mu).sum(0)                 # [C,D]
            else:
                prot_neigh = torch.zeros_like(prot_a)

            prot_comb = self.cfg.lam * prot_a + (1.0 - self.cfg.lam) * prot_neigh  # [C,D]

            # CE on combined prototypes
            logits = self._logits(Q, prot_comb)             # [Nq,C]
            ce = F.cross_entropy(logits, yq)

            # consistency between prot_a and prot_b (only over classes present in both)
            # prot_a and prot_b should come from the same user
            ##### SKIP the consistency loss ###
            # both_present = pres_a & pres_b
            # if both_present.any():
            #     diffs = prot_a[both_present] - prot_b[both_present]      # [K,D]
            #     cons = torch.norm(diffs, p=2, dim=1).mean()              # average over shared classes
            # else:
            #     cons = torch.tensor(0.0, device=self.device)


            # loss = ce + self.cfg.beta_consistency * cons 
            loss = ce
            loss.backward()

            ce_sum += float(ce.item())
            # cons_sum += float(cons.item())
           
            used += 1

        if used > 0:
            self.optimizer.step()

        logs = {
            "meta/ce": ce_sum / max(1, used),
            # "meta/cons": cons_sum / max(1, used),
            # "meta/ctr": ctr_sum / max(1, used),
            # "meta/tasks": used
        }
        if self.wandb_run is not None:
            self.wandb_run.log(logs)
        return logs


    def _token_vecs_labels_batched(
        self,
        batch: Dict[str, torch.Tensor],
        max_subbatch: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Like _token_vecs_labels but runs the encoder in chunks to handle very large sets.
        Returns:
            V: [N, D] (L2-normalized token vectors)
            y: [N]
        """
        input_ids = batch["input_ids"].to(self.device)
        attn      = batch["attention_mask"].to(self.device)
        labels    = batch["labels"].to(self.device)

        B = input_ids.size(0)
        vecs, cls = [], []

        with torch.no_grad():
            for i in range(0, B, max_subbatch):
                j = min(i + max_subbatch, B)
                Z = self.encoder(input_ids[i:j], attn[i:j])  # [b, T, D]
                # pick labeled token per sequence
                for b in range(i, j):
                    row = labels[b]
                    idx = (row != -100).nonzero(as_tuple=False)
                    if idx.numel() == 0:
                        continue
                    t = int(idx[0].item())
                    # Z is aligned to [i:j], so index relative
                    vecs.append(Z[b - i, t])
                    cls.append(int(row[t].item()))

        # if not vecs:
        #     return torch.empty(0, self.cfg.proj_dim, device=self.device), torch.empty(0, dtype=torch.long, device=self.device)
        if not vecs:
            raise RuntimeError("no valid token embedding in batch")
        V = l2n(torch.stack(vecs, 0))
        y = torch.tensor(cls, dtype=torch.long, device=self.device)
        return V, y
        
    @torch.no_grad()
    def classify_with_support(
        self,
        support_input: Dict[str, torch.Tensor],
        query_input: Dict[str, torch.Tensor],
        mix_neighbors: bool = True
    ):
        """
        Based on the prototypes of each class, 
        retrieve similar user using e_u, mix prototypes with similar users' prototypes 
        classify using mixed prototypes 
        """
        self.eval()

        # support is small;
        S, y = self._token_vecs_labels(support_input)
        if S.numel() == 0:
            return [], []

        prototypes, present = self._prototypes(S, y)

        # neighbor mixing (fallback to own prototypes if memory empty)
        if mix_neighbors:
            e_u = self.embedder(S, y, prototypes, present)
            neigh = self.memory.neighbors("inference_user", e_u, k=self.cfg.k_neighbors)
            if len(neigh) > 0:
                keys, sims = zip(*neigh)
                sims_t = torch.tensor(sims, device=self.device).view(1, -1)
                alpha = torch.softmax(sims_t / self.cfg.tau, dim=-1).squeeze(0)  # [K]
                neigh_mu = torch.stack([self.memory.user_mu[k] for k in keys], dim=0)  # [K,C,D]
                prot_neigh = (alpha.view(-1, 1, 1) * neigh_mu).sum(0)                 # [C,D]
                prototypes = self.cfg.lam * prototypes + (1.0 - self.cfg.lam) * prot_neigh
      
        # --- large query set: batch both encoding and logits ---
        Q, yq = self._token_vecs_labels_batched(query_input, max_subbatch=self.cfg.max_query_subbatch)
        if Q.numel() == 0:
            return [], []

        preds, labels = [], []
        N = Q.size(0)
      
        chunk = self.cfg.max_query_subbatch
        for i in range(0, N, chunk):
            j = min(i + chunk, N)
            logits = self._logits(Q[i:j], prototypes)  # [chunk, C]
            preds.extend(logits.argmax(dim=1).tolist())
            labels.extend(yq[i:j].tolist())

        return preds, labels

    

    @torch.no_grad()
    def evaluate_task(self, task, mix_neighbors: bool = True):
        evaluation_set = task.get_evaluation_set(task.query_set) 
        # for inference task, the support ration = number of support set
        support_set = task.get_evaluation_set(task.support_set) 
        preds, labels = self.classify_with_support(support_set, evaluation_set, mix_neighbors = mix_neighbors)
        return preds, labels


    def save(self, path: str):
        torch.save({
            "encoder": self._enc().state_dict(),
            "embedder": self.embedder.state_dict(),
            "cfg": self.cfg.__dict__,
        }, path)

    def load(self, path: str, strict: bool = True):
        
        ckpt = torch.load(path, map_location=self.device)
        self._enc().load_state_dict(ckpt["encoder"], strict=strict)
        self.embedder.load_state_dict(ckpt["embedder"], strict=strict)
        # self.cfg = ckpt["cfg"] #TODO load properly
        print(f"ckpt loaded from {path}")
    

    @torch.no_grad()
    def rebuild_memory_from_users(
        self,
        support_user_data: list,          # list of user dicts compatible with BertTokenTask
        tokenizer,
        support_batch_size: int = 30,
        shots_per_user: int = 3,
        seed: int = 0,
    ):
        """
        Recompute UserProtoMemory from support users.
        """
       
        rng = np.random.RandomState(seed)

        # reset memory
        self.memory = UserProtoMemory(momentum=0.9, device=self.device)

        for u_idx, udata in tqdm(enumerate(support_user_data), desc="building support user memory"):
            task = BertTokenTask(udata, tokenizer, support_ratio=0.9)
            # give a deterministic id if not present
            uid = getattr(task, "task_id", None)
            if uid is None:
                uid = getattr(task, "user_name", f"support_{u_idx}")

            # aggregate across multiple sampled supports to stabilize memory
            agg_protos = torch.zeros(self.cfg.num_classes, self.cfg.proj_dim, device=self.device)
            agg_count  = 0
            agg_eu     = torch.zeros(1, self.cfg.user_dim, device=self.device)

            for _ in range(shots_per_user):
                sup, _, _ = task.get_data(support_batch_size=support_batch_size, query_batch_size=0, ensure_non_zero=True)
                S, y = self._token_vecs_labels(sup)
                if S.numel() == 0:
                    continue
                prototypes, present = self._prototypes(S, y)
                e_u = self.embedder(S, y, prototypes, present)  # [1, U]
                agg_protos += prototypes
                agg_eu     += e_u
                agg_count  += 1

            if agg_count == 0:
                # no valid support for this user; skip
                raise RuntimeError("No Valid Count")
                # continue

            prototypes = (agg_protos / agg_count).detach()
            e_u        = (agg_eu / agg_count).detach()
            self.memory.update(uid, e_u, prototypes)
