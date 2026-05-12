# Integrazione MixtureOfExpertHead nel MViT

## Panoramica

Il modello MViT supporta ora l'uso di `MixtureOfExpertHead` come testa di classificazione alternativa a `TransformerBasicHead`.

## Integrazione nel Codice

La logica di selezione si trova in [tapis/models/video_model_builder.py](tapis/models/video_model_builder.py#L684-L705):

```python
if hasattr(cfg, 'TASKS') and hasattr(cfg.TASKS, 'HEAD_TYPE') and cfg.TASKS.HEAD_TYPE == 'moe':
    # Use Mixture of Experts head
    extra_head = head_helper.MixtureOfExpertHead(...)
else:
    # Use default Transformer head
    extra_head = head_helper.TransformerBasicHead(...)
```

## Configurazione nel YAML

Per abilitare MixtureOfExpertHead nel tuo config YAML (es. `configs/Orsi/TAPIS/TAPIS_LONG.yaml`):

```yaml
TASKS:
  HEAD_TYPE: "moe"          # Abilita Mixture of Experts
  MOE_NUM_EXPERTS: 3        # Numero di esperti (default: 3)
  MOE_EXPERT_DIM_HIDDEN: 512  # Dimensione hidden per esperto (default: 512)
  
  # ... altri parametri
  TASKS: ["steps", "phases"]
  NUM_CLASSES: [33, 15]
  HEAD_ACT: ["softmax", "softmax"]
```

## Utilizzo da Linea di Comando

Puoi anche passare i parametri direttamente da linea di comando:

```bash
python -B tools/run_net.py --cfg configs/Orsi/TAPIS/TAPIS_LONG.yaml \
    TRAIN.ENABLE True VAL.ENABLE True \
    TASKS.HEAD_TYPE "moe" \
    TASKS.MOE_NUM_EXPERTS 3 \
    TASKS.MOE_EXPERT_DIM_HIDDEN 512 \
    OUTPUT_DIR outputs/orsi/LONG/moe_run1
```

## Parametri Configurabili

| Parametro | Descrizione | Default | Note |
|-----------|-------------|---------|------|
| `TASKS.HEAD_TYPE` | Tipo di head: "transformer" o "moe" | "transformer" | Richiesto per abilitare MoE |
| `TASKS.MOE_NUM_EXPERTS` | Numero di reti esperte | 3 | Aumentare per più diversità |
| `TASKS.MOE_EXPERT_DIM_HIDDEN` | Dimensione hidden layer per esperto | 512 | Trade-off parametri/capacità |

## Compatibilità

- ✅ MViT: **Supportato** con TransformerBasicHead
- ✅ MViT con cls_embed: **Supportato**
- ⚠️ SlowFast: Non raccomandato (usa ResNetBasicHead)
- ✅ MultiGPU/DistributedDataParallel: Supportato

## Comparazione Testa Transformer vs MoE

### TransformerBasicHead (Default)
```
Input (batch, seq_len, embed_dim)
    ↓
Media su sequenza
    ↓
Linear Layer (embed_dim → num_classes)
    ↓
Activation
    ↓
Output (batch, num_classes)

Parametri: embed_dim × num_classes ≈ 768 × 33 ≈ 25K
```

### MixtureOfExpertHead
```
Input (batch, seq_len, embed_dim)
    ↓
Media su sequenza
    ├→ [Expert 1] → logits
    ├→ [Expert 2] → logits
    └→ [Expert 3] → logits
    ↓
Gating Network → pesi esperti
    ↓
Weighted Sum → Final Projection
    ↓
Output (batch, num_classes)

Parametri: 3 × (embed_dim × hidden) + (embed_dim × num_experts) + (num_classes × num_classes)
           ≈ 3 × (768 × 512) + (768 × 3) + (33 × 33) ≈ 1.2M
```

## Vantaggi e Trade-offs

### Vantaggi di MoE
- 🎯 **Specializzazione**: Ogni esperto impara diverse rappresentazioni
- 🔄 **Routing adattivo**: La gating network apprende a pesare dinamicamente gli esperti
- 📈 **Potenziale accuratezza**: Può catturare pattern complessi meglio di una singola testa

### Trade-offs
- ⚠️ **Parametri**: ~50x più parametri rispetto a TransformerBasicHead
- 🐢 **Velocità**: Inference leggermente più lenta (esecuzione di N esperti)
- 🎓 **Training**: Convergenza potrebbe richiedere più epoch

## Example: Configurazione Completa

**File: configs/Orsi/TAPIS/TAPIS_MOE.yaml**

```yaml
# Copia TAPIS_LONG.yaml e modifica:

TASKS:
  TASKS: ["steps", "phases"]
  NUM_CLASSES: [33, 15]
  HEAD_ACT: ["softmax", "softmax"]
  LOSS_FUNC: ["cross_entropy", "cross_entropy"]
  WEIGHT_LOSS_BY_CLASS: [steps_distribution.csv, phases_distribution.csv]
  
  # Abilita MoE
  HEAD_TYPE: "moe"
  MOE_NUM_EXPERTS: 4
  MOE_EXPERT_DIM_HIDDEN: 768

# ... resto della configurazione
```

**Esecuzione:**

```bash
bash run_files/orsi_long.sh TAPIS_MOE.yaml
```

## Monitoraggio Training

Durante il training, puoi monitorare:

1. **Loss convergence**: Comparabile a TransformerBasicHead
2. **Per-expert activation**: Osservare quale esperto è più attivato (se implementato)
3. **Gating weights**: Verificare che la gating network impari pesi diversi

## Troubleshooting

### Errore: "TASKS.HEAD_TYPE not found"
**Soluzione**: Assicurati che il parametro sia in maiuscolo nel YAML

### MoE più lento di Transformer
**Soluzione**: Riduci `MOE_NUM_EXPERTS` o `MOE_EXPERT_DIM_HIDDEN`

### Convergenza lenta
**Soluzione**: MoE ha più parametri, potrebbe richiedere:
- Learning rate più basso
- Più epoch di training
- Warmup più lungo

## Prossimi Sviluppi

1. **Sparse activation**: Solo top-K esperti attivi
2. **Load balancing**: Bilanciare carico tra esperti
3. **Per-task experts**: Esperti diversi per ogni task
4. **Mixture of Mixture**: MoE nested per feature extraction più complessa

## Riferimenti

- [tapis/models/head_helper.py](tapis/models/head_helper.py) - Implementazione MixtureOfExpertHead
- [tapis/models/video_model_builder.py](tapis/models/video_model_builder.py) - Integrazione nel MViT
- [MOE_HEAD_GUIDE.md](MOE_HEAD_GUIDE.md) - Guida tecnica dettagliata
