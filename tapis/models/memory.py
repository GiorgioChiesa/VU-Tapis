
from torch import nn
import torch
import math

class CustomPositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=5000, encoding_type="learnable"):
        super(CustomPositionalEncoding, self).__init__()
        
        # Let's define the encoding matrix based on the desired encoding type
        self.encoding_type = encoding_type
        
        if encoding_type == "sinusoidal":
            # Sinusoidal Encoding
            self.register_buffer('positional_encoding', self._get_sinusoidal_encoding(embed_dim, max_len))
        elif encoding_type == "linear":
            # Linear Encoding (values increase linearly with position)
            self.register_buffer('positional_encoding', self._get_linear_encoding(embed_dim, max_len))
        elif encoding_type == "learnable":
            # Learnable Encoding (the model learns the encoding)
            self.positional_encoding = nn.Parameter(torch.zeros(max_len, embed_dim))
            nn.init.xavier_uniform_(self.positional_encoding)  # Initialize the learnable parameters

    def forward(self, x):
        # Ensure the positional encoding matrix aligns with the input's sequence length
        return self.positional_encoding[x, :].unsqueeze(0)

    def _get_sinusoidal_encoding(self, embed_dim, max_len):
        # Generates a sinusoidal positional encoding matrix
        pos_encoding = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * -(math.log(10000.0) / embed_dim))
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        return pos_encoding

    def _get_linear_encoding(self, embed_dim, max_len):
        # Generates a linear positional encoding matrix
        return torch.linspace(0, 1, steps=max_len).unsqueeze(1).expand(max_len, embed_dim)


class MemoryTokenizer(nn.Module):
    def __init__(self, d_model, max_len, num_classes):
        super().__init__()
        self.mem_cls = nn.Parameter(torch.randn(1, 1, d_model))
        # Encoding temporale relativo: posizione -max_len ... -1
        self.pos_embed = CustomPositionalEncoding(d_model, max_len + 1, encoding_type="linear")
        self.input_proj = nn.Linear(d_model + num_classes, d_model)
        self.max_len = max_len
    
    def forward(self, memory_bank_items, current_step, device='cuda'):
        """
        memory_bank_items: lista di dict {'embedding': tensor, 'logit': tensor, 'step': int}
        """
        # Handle empty memory bank
        if not memory_bank_items:
            cls = self.mem_cls.expand(1, -1, -1)  # [1, 1, d_model]
            return cls
        
        tokens = []
        positions = []
        
        for item in memory_bank_items:
            # Proietta embedding+logit → d_model
            combined = torch.cat([item['embedding'].squeeze(0), item['logit']], dim=-1).to(device)  # [d_model + num_classes]
            token = self.input_proj(combined)  # [d_model]
            tokens.append(token)
            
            # Posizione relativa (quanti step fa era questo token)
            rel_pos = current_step - item['step']
            rel_pos_clamped = min(rel_pos, self.max_len) # clamp
            positions.append(rel_pos_clamped)
        
        tokens = torch.stack(tokens, dim=0).unsqueeze(0)  # [1, L, d_model]
        pos_enc = self.pos_embed(torch.tensor(positions).to(device))  # [L, d_model]
        tokens = tokens + pos_enc  # aggiungi encoding temporale
        
        # Prependi [MEM_CLS]
        cls = self.mem_cls.expand(tokens.size(0), -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)  # [1, L+1, d_model]
    
        return tokens
    

class LightweightMemoryEncoder(nn.Module):
    """
    Comprime la sequenza variabile di ricordi in un contesto fisso.
    Usa solo 2-3 layer Transformer encoder.
    """
    def __init__(self, d_model=768, nhead=8, num_layers=2, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,  # lightweight: 2x invece del solito 4x
            dropout=dropout,
            batch_first=True,
            norm_first=True  # Pre-LN: più stabile
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, d_model)
    
    def forward(self, memory_tokens, src_key_padding_mask=None):
        """
        memory_tokens: [B, L, d_model]  (L = lunghezza memoria, anche variabile con mask)
        src_key_padding_mask: [B, L] - True dove ci sono token di padding
        
        Returns: [B, d_model] - il CLS token come rappresentazione globale della memoria
        """
        out = self.encoder(memory_tokens, src_key_padding_mask=src_key_padding_mask)
        # Estrai il primo token (MEM_CLS) come summary della memoria
        # memory_context = self.output_proj(out[:, 0, :])  # [B, d_model]
        memory_context = self.output_proj(out)  # [B, d_model]
        return memory_context


class MemoryAugmentedClassifier(nn.Module):
    """
    Combina il feature del frame corrente con il contesto di memoria
    tramite cross-attention, poi classifica.
    """
    def __init__(self, d_model=768, nhead=8, tasks=[""], num_classes=[40, 40], device=""):
        super().__init__()
        
        # Cross-attention: frame corrente (query) attende sulla memoria (key, value)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=0.1,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 2, d_model)
        )
        if not device:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tasks = tasks
        assert len(tasks) == len(num_classes), "tasks e num_classes devono avere la stessa lunghezza"

        for task, num in zip(tasks, num_classes):
            setattr(self, f"{task}_head", nn.Linear(d_model, num, device=device))
        self.classification_heads = [nn.Linear(d_model, num, device=device) for num in num_classes]  # una testa per ogni task
    
    def forward(self, current_feat, memory_context):
        """
        current_feat: [B, d_model] - embedding CLS del batch corrente da MViT
        memory_context: [B, L, d_model] - tutti i token di memoria (non solo il CLS)
                        oppure [B, 1, d_model] se usi solo il CLS della memoria
        """
        # Porta current_feat a [B, 1, d_model] per la cross-attention
        # query = current_feat.unsqueeze(1)  # [B, 1, d_model]
        query = current_feat # [B, 1, d_model]
        if len(memory_context.shape) == 2:
            memory_context = memory_context.unsqueeze(1)  # [B, 1, d_model]
        # Cross-attention: il frame corrente "interroga" la memoria
        attended, _ = self.cross_attn(query, memory_context, memory_context)
        
        # Residual + Norm
        x = self.norm1(query + attended)  # [B, 1, d_model]
        x = self.norm2(x + self.ffn(x))
        x = x.squeeze(1)  # [B, d_model]
        
        # for i in 
        # phase_logits = self.phase_head(x)
        # step_logits = self.step_head(x)
        logits_list = {}
        for task in self.tasks:
            head = getattr(self, f"{task}_head")
            logits_list[task] = head(x)

        # logits_list = [head(x) for head in self.classification_heads]

        return logits_list
    

from collections import deque
import torch

class MemoryBank:
    """
    Salva (embedding, logit, step_index) ogni saving_rate passi.
    Politica FIFO: quando piena, rimuove il token più vecchio.
    """
    def __init__(self, max_len=100, saving_rate=5, d_model=768, num_classes=40):
        self.max_len = max_len
        self.saving_rate = saving_rate
        self.d_model = d_model
        self.num_classes = num_classes
        self.bank = deque(maxlen=max_len)  # FIFO automatico con maxlen!
        self._step_counter = 0
    
    def update(self, embedding: torch.Tensor, logit: torch.Tensor):
        """
        Chiama questo ad ogni forward pass.
        Salva solo ogni saving_rate passi.
        
        embedding: [d_model] - CLS token del MViT
        logit: [num_classes] - output del modello (prima del softmax)
        """
        self._step_counter += 1
        
        if self._step_counter % self.saving_rate == 0:
            self.bank.append({
                'embedding': embedding.detach().cpu(),
                'logit': logit.detach().cpu(),
                'step': self._step_counter
            })
    
    def get_list(self, max_len=None, device='cuda'):
        """Restituisce la lista di item nella memoria (per debug)."""
        items = list(self.bank)
        if max_len is not None:
            items = items[-max_len:]
        return items

    def get_tokens(self, max_len=None, device='cuda'):
        """
        Restituisce i token di memoria come tensore.
        
        Returns:
            tokens: [L, d_model + num_classes]
        """
        max_len = max_len or self.max_len
        
        if len(self.bank) == 0:
            # Nessuna memoria: restituisce un token zero
            empty = torch.zeros(1, self.d_model + self.num_classes).to(device)
            return empty, torch.zeros(1, dtype=torch.bool).to(device)
        
        items = list(self.bank)
        tokens = torch.stack([
            torch.cat([item['embedding'], item['logit']], dim=-1)
            for item in items
        ], dim=0).to(device)  # [L, d_model + num_classes]
        
        return tokens
    
    def reset(self):
        """Reset all'inizio di un nuovo video."""
        self.bank.clear()
        self._step_counter = 0
    
    def __len__(self):
        return len(self.bank)