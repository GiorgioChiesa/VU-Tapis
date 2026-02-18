# ORSI to COCO Dataset Converter

Convertitore per trasformare annotazioni di dataset ORSI (formato RARP) nel formato COCO compatibile con GraSP/TAPIS.

## 📋 Indice

- [Panoramica](#panoramica)
- [Caratteristiche](#caratteristiche)
- [Requisiti](#requisiti)
- [Installazione](#installazione)
- [Utilizzo](#utilizzo)
- [Formato Input](#formato-input)
- [Formato Output](#formato-output)
- [Configurazioni](#configurazioni)
- [Esempi](#esempi)
- [Dettagli tecnici](#dettagli-tecnici)

## 🎯 Panoramica

Questo script converte le annotazioni ORSI (nel formato RARP01.json) nel formato COCO standard utilizzato da GraSP/TAPIS. La conversione gestisce:

- **Fasi chirurgiche** (PHASES) → `phases_categories`
- **Step/Eventi** (EVENTS) → `steps_categories`
- **Timestamp temporali** → Frame numbers con interpolazione temporale
- **Instant events** → Eventi istantanei attivi per una durata specificata
- **Frame con annotazioni** → Forzati nel dataset estratto

## ✨ Caratteristiche

### 1. **Differenziazione Instant Events**
- Identifica automaticamente gli eventi istantanei (point-in-time)
- Li distingue dagli eventi range (start/end)
- Mantiene gli instant events attivi per una durata configurabile (default: 3 secondi)

### 2. **Forced Annotation Frames**
- Tutti i frame con evento/fase sono inclusi nel dataset estratto
- Aggiunge frame adiacenti per maggior contesto
- Combina campionamento regolare con frame annotati

### 3. **Conversione Temporale**
- Converte timestamp (secondi) → frame numbers
- Usa FPS configurabile per conversione accurata
- Supporta video con framerate diversi

### 4. **Batch Processing**
- Elabora più file di annotazione contemporaneamente
- Genera CSV combinati per training
- Può processare interi dataset

### 5. **Generazione CSV**
- Crea file CSV compatibili con GraSP
- Formato: `VIDEO_NAME PARTITION_IDX FRAME_IDX FILE_PATH`
- Supporta CSV singoli per video e CSV combinati

### 6. **Lettura Dimensioni Immagine (opzionale)**
- Legge width e height effettive dai file immagine se disponibili
- Usa PIL/Pillow se installato
- Fallback a dimensioni default se immagini non disponibili
- Caching delle dimensioni per performance

## 📦 Requisiti

- Python 3.7+
- Librerie standard: `json`, `csv`, `os`, `pathlib`, `collections`
- **Opzionale:** PIL/Pillow (per leggere dimensioni effettive dalle immagini)

Nessuna dipendenza esterna **richiesta** per il funzionamento base!

## 🚀 Installazione

```bash
# Posizionarsi nella directory del converter
cd /scratch/Video_Understanding/GraSP/TAPIS/data/GraSP/conversion_to_Orsidataset

# Il file orsi2coco.py è già disponibile
# Non richiede installazione di dipendenze aggiuntive
```

## 📖 Utilizzo

### Conversione di un singolo file

```bash
python orsi2coco.py \
  --input RARP01.json \
  --output RARP01_coco.json \
  --fps 25 \
  --frame-step 46 \
  --instant-duration 3.0 \
  --csv \
  --csv-output train.csv
```

### Batch processing (più file)

```bash
python orsi2coco.py \
  --input /path/to/annotations_directory/ \
  --output /path/to/output_directory/ \
  --batch \
  --fps 25 \
  --frame-step 46 \
  --instant-duration 3.0 \
  --csv
```

### Visualizzare gli argomenti disponibili

```bash
python orsi2coco.py --help
```

## 📥 Formato Input

### RARP01.json

Struttura del file di annotazione ORSI:

```json
{
  "REMARKS": "Descrizione del video",
  "EVENTS": {
    "Categoria_Evento_1": {
      "Nome_Evento_1": [timestamp1, timestamp2, ...],
      "Nome_Evento_2": [timestamp1, timestamp2, ...],
      ...
    },
    "Categoria_Evento_2": {
      ...
    }
  },
  "PHASES": {
    "Nome_Fase_1": {
      "START": [timestamp_inizio],
      "END": [timestamp_fine]
    },
    "Nome_Fase_2": {
      "START": [timestamp_inizio],
      "END": [timestamp_fine]
    },
    ...
  }
}
```

**Note:**
- I timestamp sono in **secondi** dall'inizio del video
- EVENTS possono avere più timestamp per lo stesso evento
- PHASES sono definite da coppie START-END
- Eventi senza END corrispondente sono considerati instant events

### Esempio RARP01.json (parte)

```json
{
  "EVENTS": {
    "General Events": {
      "Out of body": [32.98, 341.99, 1178.3, ...],
      "Back inside body": [61.8, 355.42, 1195.57, ...]
    },
    "Bladder detachment": {
      "Incision peritoneum - left": [367.53],
      "Adhesion removal": [135.97]
    }
  },
  "PHASES": {
    "Port insertion and surgical access": {
      "START": [135.97],
      "END": [367.53]
    },
    "Bladder detachment": {
      "START": [367.53],
      "END": [984.06]
    }
  }
}
```

## 📤 Formato Output

### RARP01_coco.json

Formato COCO standard compatibile con GraSP:

```json
{
  "info": {
    "description": "ORSI Dataset",
    "url": "https://example.com",
    "version": "1",
    "year": "2024",
    "contributor": "ORSI"
  },
  "phases_categories": [
    {
      "id": 0,
      "name": "Idle",
      "description": "Idle",
      "supercategory": "phase"
    },
    {
      "id": 1,
      "name": "Apical_dissection",
      "description": "Apical dissection",
      "supercategory": "phase"
    },
    ...
  ],
  "steps_categories": [
    {
      "id": 0,
      "name": "Idle",
      "description": "Idle",
      "supercategory": "step"
    },
    {
      "id": 1,
      "name": "Prostate_bagging",
      "description": "Prostate bagging",
      "supercategory": "step"
    },
    ...
  ],
  "images": [
    {
      "id": 1,
      "file_name": "RARP01/000000000.jpg",
      "width": 1280,
      "height": 800,
      "date_captured": "",
      "license": 1,
      "coco_url": "",
      "flickr_url": "",
      "video_name": "RARP01",
      "frame_num": 0
    },
    ...
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "image_name": "RARP01/000000000.jpg",
      "phases": 0,
      "steps": 45
    },
    ...
  ]
}
```

### train.csv

Formato CSV compatibile con GraSP frame lists:

```
VIDEO_NAME PARTITION_IDX FRAME_IDX FILE_PATH
RARP01 1 0 RARP01/000000000.jpg
RARP01 1 1 RARP01/000000001.jpg
RARP01 1 2 RARP01/000000002.jpg
...
```

**Colonne:**
- `VIDEO_NAME`: Nome del video/caso
- `PARTITION_IDX`: Indice partizione (1 per training, 0 per test, ecc.)
- `FRAME_IDX`: Indice sequenziale del frame nel dataset
- `FILE_PATH`: Path relativo al frame (es. CASO001/000000000.jpg)

## ⚙️ Configurazioni

### Argomenti da linea di comando

| Argomento | Default | Descrizione |
|-----------|---------|-------------|
| `--input` | - | Input JSON file o cartella con file JSON |
| `--output` | - | Output JSON file o cartella |
| `--fps` | 25 | Frame rate del video (conversione secondi→frame) |
| `--frame-step` | 46 | Estrai frame ogni N frame (≈1.84s @ 25fps) |
| `--width` | 1280 | Larghezza frame video |
| `--height` | 800 | Altezza frame video |
| `--instant-duration` | 3.0 | Durata (secondi) degli instant events |
| `--csv` | False | Genera file CSV |
| `--csv-output` | - | Path file CSV combinato |
| `--frame-root` | - | Directory radice dei frame (legge dimensioni reali dalle immagini) |
| `--batch` | False | Modalità batch (processa cartella) |

### Configurazione nel codice

È possibile modificare le costanti globali nel file:

```python
FPS = 25                      # Frame rate default
INSTANT_EVENT_DURATION = 3    # Durata instant events (secondi)
```

## 📝 Esempi

### Esempio 1: Conversione singolo file con default

```bash
python orsi2coco.py \
  --input RARP01.json \
  --output RARP01_coco.json
```

Output:
```
Converted: RARP01.json
  Total frames: 150255
  Extracted frames: 8985
  Frames with annotations: 5925
  Instant events: 72
  Phases: 17
  Steps: 73
  Output: RARP01_coco.json
```

### Esempio 2: Batch processing con CSV

```bash
python orsi2coco.py \
  --input ~/dataset/annotations/ \
  --output ~/dataset/coco_format/ \
  --batch \
  --fps 30 \
  --frame-step 30 \
  --csv
```

Output:
```
Found 5 annotation files
Converted: RARP01.json
  Total frames: 150255
  Extracted frames: 8985
  ...
Converted: RARP02.json
  ...
Generated combined CSV: ~/dataset/coco_format/train.csv
Total rows in CSV: 45230
```

### Esempio 3: Video con FPS diverso

```bash
python orsi2coco.py \
  --input RARP50.json \
  --output RARP50_coco.json \
  --fps 30 \
  --instant-duration 2.5 \
  --csv
```

### Esempio 4: Frame sampling aggressivo

```bash
python orsi2coco.py \
  --input RARP01.json \
  --output RARP01_lite.json \
  --frame-step 100 \
  --csv
```

(Estrae frame ogni 4 secondi a 25fps)

### Esempio 5: Lettura dimensioni effettive dalle immagini

```bash
python orsi2coco.py \
  --input RARP01.json \
  --output RARP01_coco.json \
  --frame-root /path/to/frames/root \
  --csv
```

(Legge width e height reali dai file immagine JPEG/PNG)

## 🔧 Dettagli Tecnici

### Algoritmo di Identificazione Instant Events

Un evento è considerato **instant** se:
1. Non ha un evento "END" corrispondente
2. Non ha pattern start/end/begin nel nome
3. È isolato rispetto ad altri eventi

```python
# Esempio di instant events individuati
- "Hemostatic metal clip placement"
- "Prostate bagging"
- "Transection of the urethra"

# Esempio di range events (NON instant)
- "Out of body" ↔ "Back inside body"
- "Start dissection" ↔ "Continue posterior dissection"
```

### Conversione Temporale

```
frame_num = int(timestamp_secondi * fps)

Esempio:
- timestamp: 32.98 secondi
- fps: 25
- frame_num: int(32.98 * 25) = 824
```

### Collezione Frame Annotati

Per ogni evento/fase viene estratto:
- **Instant events**: Tutti i frame da `t` a `t + instant_duration`
- **Non-instant events**: Frame al timestamp + 4 frame successivi
- **Fasi**: Frame di inizio e fine di ogni fase
- **Sampling regolare**: Ogni `frame_step` frame

Tutti questi frame vengono fusi e ordinati nel dataset finale.

### Gestione ID COCO

- **image_id**: Sequenziale (1, 2, 3, ...)
- **annotation_id**: Sequenziale (1, 2, 3, ...)
- **phases id**: Assegnato in ordine alfabetico (0=Idle, 1..n=altri)
- **steps id**: Assegnato in ordine alfabetico (0=Idle, 1..n=altri)

## 🎬 Integrazione con TAPIS

### Utilizzo con il dataloader di GraSP/TAPIS

```python
from tapis.datasets.grasp import GraSPDataset

# Caricare il dataset convertito
dataset = GraSPDataset(
    json_path='RARP01_coco.json',
    frame_root='path/to/frames/',
    split='train'
)

# Usare nel training
dataloader = DataLoader(dataset, batch_size=32)
```

### Struttura directory consigliata

```
my_dataset/
├── RARP01/
│   ├── 000000000.jpg
│   ├── 000000001.jpg
│   └── ...
├── RARP02/
│   ├── 000000000.jpg
│   └── ...
├── annotations/
│   ├── RARP01_coco.json
│   ├── RARP02_coco.json
│   └── train.csv
└── RARP01.json
└── RARP02.json
```

## 📊 Statistiche Conversione RARP01

Basato sul test di conversione:

```
Input: RARP01.json
- Durata video: 6010.23 secondi (≈1.67 ore)
- Total frames: 150,255 (@ 25fps)
- Events (instant): 72
- Phases: 16

Output: RARP01_coco.json
- Extracted frames: 8,985 (5.98% del totale)
- Annotated frames: 5,925 (3.94% del totale)
- Categorie fasi: 17 (incluso Idle)
- Categorie steps: 73 (incluso Idle)

File size:
- JSON: 4.4 MB
- CSV: 315 KB
```

## 🐛 Troubleshooting

### Errore: "File not found"
```bash
# Verificare il path assoluto
python orsi2coco.py \
  --input /path/completo/RARP01.json \
  --output /path/completo/RARP01_coco.json
```

### CSV non generato
```bash
# Aggiungere flag --csv
python orsi2coco.py \
  --input RARP01.json \
  --output RARP01_coco.json \
  --csv  # <-- importante
```

### FPS non accurato
Verificare il FPS del video:
```bash
# Con ffprobe
ffprobe -v error -select_streams v:0 -show_entries \
  stream=r_frame_rate -of default=noprint_wrappers=1:nokey=1:nokey=1 video.mp4

# Impostare il valore corretto
python orsi2coco.py --input RARP01.json --output out.json --fps 30
```

## 📄 Licenza

Converter sviluppato per il dataset GraSP/TAPIS

## 📧 Contatti e Support

Per issues e domande sui formati ORSI/COCO, consultare:
- Dataset GraSP: https://cinfonia.uniandes.edu.co
- TAPIS repository: [repository_link]

---

**Ultima modifica:** 2024-02-17
**Versione:** 1.0.0
