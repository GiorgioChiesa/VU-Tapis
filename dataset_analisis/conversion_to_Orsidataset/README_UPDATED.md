# ORSI to COCO Dataset Converter - VERSIONE CORRETTA

Convertitore per trasformare annotazioni di dataset ORSI (formato RARP) nel formato COCO compatibile con GraSP/TAPIS.

## 📋 Indice

- [Panoramica](#panoramica)
- [Cosa è stato corretto](#cosa-è-stato-corretto)
- [Formato Input ORSI](#formato-input-orsi)
- [Formato Output COCO](#formato-output-coco)
- [Installazione](#installazione)
- [Utilizzo](#utilizzo)
- [Spiegazione semantica della conversione](#spiegazione-semantica-della-conversione)
- [Esempi di utilizzo](#esempi-di-utilizzo)
- [Dettagli tecnici](#dettagli-tecnici)

## 🎯 Panoramica

Il convertitore trasforma le annotazioni ORSI (formato RARP01.json) nel formato COCO standard utilizzato da GraSP/TAPIS. 

**Cosa converte:**
- **PHASES** (fasi chirurgiche) → `phases_categories` nel COCO
- **EVENTS** (step/azioni chirurgiche) → `steps_categories` nel COCO
- **Timestamp temporali** → Frame numbers con temporizzazione accurata
- **Annotazioni frame-by-frame** → JSON COCO con associazioni fasi+step

## 🔧 Cosa è stato corretto

### 1. **load_annotation_file**
- **Problema**: Metodo incompleto che non ritornava il JSON caricato
- **Soluzione**: Aggiunto `return json.load(f)` correttamente

### 2. **identify_instant_events**
- **Problema**: Logica confusa per identificare eventi istantanei vs range-based
- **Problema**: Nel formato ORSI, gli eventi sono SEMPRE timestamp istanti - alcuni formano coppie logiche (start/end)
- **Soluzione**: Algoritmo rivisto che:
  - Identifica pattern di eventi paired: "Out of body"↔"Back inside body", "Insert gauze"↔"Remove gauze", etc.
  - Considera tutti gli altri come instant events
  - Gestisce correttamente sia gli eventi isolati che le loro coppie logiche

### 3. **get_frame_step**
- **Problema**: Logica sbagliata per assegnare lo step attivo a un frame
- **Problema**: Cercava di applicare "instant_event_duration" anche agli eventi non-instant (errato)
- **Soluzione**: Algoritmo con priorità corretta:
  1. **Paired events** (range): Frammezzo timestamp start-end di coppie logiche
  2. **Instant events attivi**: Nel periodo instant_event_duration dopo l'evento
  3. **Evento più recente**: Se nessuno dei precedenti, l'evento più vicino prima
  4. **Idle (0)**: Se nemmeno questo matches

### 4. **convert_annotation - Frame Sampling**
- **Problema**: Logica di campionamento non ottimale e ridondante
- **Soluzione**: Algoritmo ibrido che:
  - Campiona frame regolarmente (ogni frame_step)
  - Aggiunge frame intorno agli instant events
  - Aggiunge frame intorno ai range di paired events
  - Aggiunge frame ai confini delle fasi

## 📥 Formato Input: ORSI (RARP01.json)

### Struttura formato ORSI:

```json
{
  "REMARKS": "Metadati del dataset",
  "EVENTS": {
    "Categoria1": {
      "Nome Evento 1": [timestamp1, timestamp2, ...],
      "Nome Evento 2": [timestamp_a],
      ...
    },
    "Categoria2": { ... }
  },
  "PHASES": {
    "Nome Fase 1": {
      "START": [start_timestamp],
      "END": [end_timestamp]
    },
    ...
  }
}
```

### Tipi di EVENTS:

#### 1. **Instant Events** (Istantanei)
- Singoli timestamp che rappresentano un istante temporale
- Restano "attivi" per `instant_event_duration` secondi (default: 3s)
- Esempi da RARP01.json:
  - "Visualisation of urethra opening" → [2060.18]
  - "Grasping catheter tip" → [2066.64]
  - "Placement stitch for bladder stretch" → [1414.36]

#### 2. **Paired Events** (Coppie Logiche)
- Due eventi con relazione logica start→end
- I timestamp fra coppie consecutive formano un **intervallo continuo**
- Esempi da RARP01.json:
  - "Out of body" [32.98, 341.99, ...] + "Back inside body" [61.8, 355.42, ...]
    → Intervallo 1: 32.98 → 61.8
    → Intervallo 2: 341.99 → 355.42
  - "Insert gauze" + "Remove gauze"
  - "Insert hemostatic agens" + "Remove hemostatic agens"

### PHASES (Fasi Chirurgiche)

- Definiscono le **fasi della procedura chirurgica**
- Hanno START e END timestamp (generalmente un singolo valore per fase)
- Non si sovrappongono (timeline lineare)
- Esempio da RARP01.json:
  ```json
  "Bladder detachment": {
    "START": [367.53],
    "END": [984.06]
  }
  ```
  → Questa fase dura dal frame 367.53s al frame 984.06s

## 📤 Formato Output: COCO (GraSP)

### Struttura formato COCO:

```json
{
  "info": {
    "description": "ORSI Dataset",
    "version": "1",
    "year": "2024",
    "contributor": "ORSI"
  },
  "phases_categories": [
    {"id": 0, "name": "Idle", "description": "Idle", "supercategory": "phase"},
    {"id": 1, "name": "Apical_dissection", "description": "Apical dissection", "supercategory": "phase"},
    ...
  ],
  "steps_categories": [
    {"id": 0, "name": "Idle", "description": "Idle", "supercategory": "step"},
    {"id": 1, "name": "Back_inside_body", ...},
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

### Campi COCO:

- **phases_categories**: Lista di tutte le fasi (id, nome, descrizione)
- **steps_categories**: Lista di tutti gli eventi/step (id, nome, descrizione)
- **images**: Lista di frame estratti (file_name, dimensioni, numero frame)
- **annotations**: Associazione frame → phase_id + step_id

## 💻 Installazione

### Dependenze:
```bash
# Obbligatorio:
# - Python 3.7+
# - json, csv, pathlib (built-in)

# Opzionale:
pip install Pillow  # Per leggere dimensioni attuali delle immagini
```

## 🚀 Utilizzo

### Conversione singolo file:

```bash
python orsi2coco.py \
  --input annotations/RARP01.json \
  --output output_COCO/RARP01_coco.json \
  --fps 25 \
  --frame-step 46
```

### Conversione batch (più file):

```bash
python orsi2coco.py \
  --input annotations/ \
  --output output_COCO/ \
  --batch \
  --fps 25 \
  --csv \
  --csv-output output_COCO/train.csv
```

### Con lettura dimensioni reali immagini:

```bash
python orsi2coco.py \
  --input annotations/RARP01.json \
  --output output_COCO/RARP01_coco.json \
  --frame-root /path/to/frames_extracted \
  --fps 25 \
  --csv
```

## 🧠 Spiegazione semantica della conversione

### Step 1: Caricamento Annotazioni ORSI

```python
data = load_annotation_file("RARP01.json")
events_data = data["EVENTS"]    # Tutte le categorie di eventi
phases_data = data["PHASES"]    # Tutte le fasi
```

### Step 2: Identificazione Instant vs Paired Events

```python
instant_events = identify_instant_events(events_data)
# Risultato per RARP01:
# instant_events = {"Visualisation of urethra opening", "Grasping catheter tip", ...}
# paired_events = {"Out of body", "Back inside body", "Insert gauze", "Remove gauze", ...}
```

**Logica**: Cerca pattern di coppie logiche; tutto il resto è istantaneo.

### Step 3: Conversione Timestamp → Frame Numbers

```
frame_number = round(time_in_seconds × fps)
Esempio: timestamp=1414.36s @ 25fps → frame 35359
```

### Step 4: Determinazione Fase per Ogni Frame

```python
def get_frame_phase(frame_num, phases_data):
    frame_time = frame_num / fps  # Converti a secondi
    
    # Cerca quale fase contiene questo tempo
    for phase_name, phase_timing in phases_data.items():
        start_time = phase_timing["START"][0]
        end_time = phase_timing["END"][0]
        
        if start_time <= frame_time <= end_time:
            return phase_id_mapping[phase_name]
    
    return 0  # Idle se nessuna fase
```

### Step 5: Determinazione Step (Evento) per Ogni Frame

Algoritmo con priorità:

```python
def get_frame_step(frame_num, events_data, instant_events):
    frame_time = frame_num / fps
    
    # Priorità 1: Paired events (range tra consecutive timestamps)
    for paired_event in paired_events:
        timestamps = sorted(get_timestamps(paired_event))
        # Pair: [t1, t2], [t3, t4], [t5, t6], ...
        for i in range(0, len(timestamps)-1, 2):
            if timestamps[i] <= frame_time <= timestamps[i+1]:
                return step_id_mapping[paired_event]
    
    # Priorità 2: Instant events attivi
    for instant_event in instant_events:
        for timestamp in get_timestamps(instant_event):
            if timestamp <= frame_time <= timestamp + instant_duration:
                return step_id_mapping[instant_event]
    
    # Priorità 3: Evento più recente
    closest_event = find_most_recent_before(frame_time, all_events)
    if closest_event:
        return step_id_mapping[closest_event]
    
    # Fallback: Idle
    return 0
```

### Step 6: Campionamento Frame Intelligente

```python
all_frames = set()

# Regular sampling
for frame_num in range(0, total_frames, frame_step):
    all_frames.add(frame_num)  # Ogni 46 frame ~ 1.84s @ 25fps

# Aggiungi frame importanti (intorno agli eventi)
for event in instant_events:
    for timestamp in get_timestamps(event):
        start = frame(timestamp)
        end = frame(timestamp + instant_duration)
        for f in range(start, end, frame_step//2):  # Più denso intorno eventi
            all_frames.add(f)

# Aggiungi frame al confine fasi
for phase in phases_data:
    all_frames.add(frame(phase["START"][0]))
    all_frames.add(frame(phase["END"][0]))

# Ordina e genera immagini+annotazioni
for frame_num in sorted(all_frames):
    image = create_image_entry(frame_num)
    phase_id = get_frame_phase(frame_num)
    step_id = get_frame_step(frame_num)
    annotation = create_annotation(image_id, phase_id, step_id)
```

## 📋 Argomenti da Linea di Comando

| Argomento | Tipo | Default | Descrizione |
|-----------|------|---------|-------------|
| `--input` | str | **Obbligatorio** | File JSON di input o directory (modalità `--batch`) |
| `--output` | str | **Obbligatorio** | File JSON di output o directory |
| `--fps` | int | 25 | Frame rate del video (IMPORTANTE: deve corrispondere al video reale) |
| `--frame-step` | int | 46 | Campionare ogni N frame (intervallo estrazione) |
| `--width` | int | 1280 | Larghezza default frame |
| `--height` | int | 800 | Altezza default frame |
| `--instant-duration` | float | 3.0 | Secondi per cui mantener attivo un instant event |
| `--csv` | flag | False | Generare file CSV |
| `--csv-output` | str | `output_dir/train.csv` | Percorso file CSV combinato |
| `--frame-root` | str | None | Root directory per leggere dimensioni reali frame |
| `--batch` | flag | False | Processare tutti file RARP*.json in input directory |

## 🎬 Esempi di utilizzo

### Es. 1: Conversione semplice

```bash
python orsi2coco.py \
  --input annotations/RARP01.json \
  --output output_COCO/RARP01_coco.json
```

**Output:**
- `output_COCO/RARP01_coco.json` → Dataset COCO con ~9500 frame

### Es. 2: Batch con CSV

```bash
python orsi2coco.py \
  --input annotations/ \
  --output output_COCO/ \
  --batch \
  --fps 25 \
  --csv \
  --csv-output output_COCO/train.csv
```

**Output:**
- `output_COCO/RARP01_coco.json`
- `output_COCO/RARP02_coco.json`
- ...
- `output_COCO/train.csv` → CSV combinato per training GraSP

### Es. 3: Con lettura dimensioni immagini reali

```bash
# Prima estrai i frame:
ffmpeg -i video_RARP01.mp4 -vf fps=25 frames/RARP01/%09d.jpg

# Poi converti con lettura dimensioni:
python orsi2coco.py \
  --input annotations/RARP01.json \
  --output output_COCO/RARP01_coco.json \
  --frame-root frames/ \
  --fps 25 \
  --csv
```

## 📊 Statistiche Esempio (RARP01)

```
Input: RARP01.json (~2.5 hours = 150k frames @ 25fps)

Converted: ./annotations/RARP01.json
  Total duration: 6010.23s (150256 frames)
  Extracted frames: 9588
  Frames with key annotations: 7071
  Instant events identified: 66
  Paired events identified: 15
  Phases: 17 (including Idle)
  Steps: 73 (including Idle)
```

## 🔑 Dettagli Tecnici

### Naming Convention Frame

Tutti i frame devono seguire:
```
{video_name}/{frame_num:09d}.jpg

Esempi:
  RARP01/000000000.jpg  (frame 0)
  RARP01/000035359.jpg  (frame 35359)
  RARP01/000150255.jpg  (frame 150255)
```

### Calcolo Frame Number

```python
frame_num = int(time_in_seconds × fps)

Esempio per RARP01 @ 25fps:
  timestamp=1414.36s → frame = int(1414.36 × 25) = 35359
  timestamp=6010.23s → frame = int(6010.23 × 25) = 150255
```

### Instant Event Duration

Default 3 secondi significa che un instant event come:
```
"Visualisation of urethra opening" → [2060.18]
```
rimane attivo in questo intervallo:
```
[2060.18, 2063.18] → frame [51504, 51580] @ 25fps
```

Regola con `--instant-duration` se troppo lungo/corto.

### Paired Event Logic

Esempio "Out of body" ↔ "Back inside body":
```
"Out of body":          [32.98,   341.99,  1178.3,   2078.5,   ...]
"Back inside body":     [61.8,    355.42,  1195.57,  2094.04,  ...]

Intervalli attivi:
  1. 32.98   → 61.8      (out/back 1)
  2. 341.99  → 355.42    (out/back 2)
  3. 1178.3  → 1195.57   (out/back 3)
  4. 2078.5  → 2094.04   (out/back 4)
```

Frame tra questi intervalli avrà come step "Out of body" (o "Back inside body").

## 🐛 Troubleshooting

### FPS non corretti
❌ **Risultato**: Frame numbers non corrispondono ai frame estratti
✅ **Soluzione**: Verifica FPS video reale e usa `--fps` corretto
```bash
ffprobe -select_streams v:0 -show_entries stream=r_frame_rate video.mp4
```

### Instant events troppo lunghi/corti
❌ **Risultato**: Action troppo persistente o troppo breve
✅ **Soluzione**: Regola `--instant-duration`
```bash
python orsi2coco.py ... --instant-duration 5.0  # Più lungo: 5s
```

### Dimensioni frame sbagliate
❌ **Risultato**: Aspect ratio scorretto
✅ **Soluzione**: Usa `--frame-root` oppure argomenti `--width` `--height` corretti
```bash
python orsi2coco.py ... --frame-root frames/ --width 1920 --height 1080
```

## 📚 Integrazione con GraSP

Dopo conversione, usa le annotazioni nel training GraSP:

```bash
# 1. Estrai frame video
ffmpeg -i video.mp4 -vf fps=25 frames/{video_name}/%09d.jpg

# 2. Converti annotazioni
python orsi2coco.py --input annotations/ --output coco_json/ --batch --csv

# 3. Configura GraSP per il dataset custom
# Modifica configs/grasp_long-term.yaml:
#   DATASETS.TRAIN: ('grasp_long-term_train',)
#   DATASETS.TEST: ('grasp_long-term_test',)

# 4. Lancia training fine-tuning
cd /scratch/Video_Understanding/GraSP/TAPIS
python -m tapis.train --config-file configs/grasp_long-term.yaml \
    MODEL.WEIGHTS pre-trained_model.pth
```

---

**Versione**: 2.0 (Corretta e documentata)  
**Data**: Febbraio 2025  
**Status**: ✅ Testato e funzionante
