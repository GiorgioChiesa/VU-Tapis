#!/usr/bin/env python3
"""
Script per unire tutti i file JSON COCO presenti in output_COCO.
Gestisce correttamente gli indici di immagini e annotazioni.
"""

import json
import os
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import argparse

def parse_arguments():
    """Analizza gli argomenti della riga di comando."""
    parser = argparse.ArgumentParser(
        description='Unisce tutti i file JSON COCO presenti in una directory.'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='output_COCO',
        help='Directory contenente i file JSON COCO (default: output_COCO)'
    )
    return parser.parse_args()

args = parse_arguments()
output_dir = args.output_dir


def load_json_file(file_path):
    """Carica un file JSON."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json_file(data, file_path):
    """Salva un file JSON con formattazione."""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"✓ File salvato: {file_path}")


def merge_categories(all_categories):
    """
    Unisce liste di categorie rimuovendo duplicati.
    Mantiene l'ordine degli ID.
    """
    seen = {}
    merged = []
    
    for category in all_categories:
        key = (category['id'], category['name'])
        if key not in seen:
            seen[key] = True
            merged.append(category)
    
    return sorted(merged, key=lambda x: x['id'])


if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 MERGE DI FILE JSON COCO")
    print("="*60)
    
    """
    Unisce tutti i file JSON COCO nella cartella specificata.
    """
    # Get the directory path relative to the current file
    script_dir = Path(__file__).parent.absolute()
    output_coco_dir = script_dir / output_dir
    
    if not output_coco_dir.exists():
        print(f"❌ Directory non trovata: {output_coco_dir}")
        exit(1)
    
    # Trova tutti i file JSON
    json_files = sorted(output_coco_dir.glob('*_coco.json'))
    
    if not json_files:
        print(f"❌ Nessun file JSON trovato in {output_coco_dir}")
        exit(1)
    
    print(f"\n📂 Trovati {len(json_files)} file JSON:")
    for f in json_files:
        print(f"   - {f.name}")
    
    # Carica tutti i file
    datasets = []
    file_sizes = {}
    
    for json_file in json_files:
        print(f"\n📥 Caricamento: {json_file.name}...")
        data = load_json_file(json_file)
        datasets.append(data)
        file_sizes[json_file.stem] = {
            'images': len(data.get('images', [])),
            'annotations': len(data.get('annotations', []))
        }
        print(f"   Images: {file_sizes[json_file.stem]['images']}")
        print(f"   Annotations: {file_sizes[json_file.stem]['annotations']}")
    
    # Unisci i dataset
    print("\n🔄 Unione dei dataset in corso...")
    
    merged_data = {
        'info': {},
        'phases_categories': [],
        'steps_categories': [],
        'images': [],
        'annotations': []
    }
    
    # Unisci info (prendi la prima e aggiungi metadati di merge)
    merged_data['info'] = datasets[0]['info'].copy()
    merged_data['info']['merged_from'] = [f.name for f in json_files]
    merged_data['info']['merge_date'] = datetime.now().isoformat()
    merged_data['info']['total_files_merged'] = len(json_files)
    
    # Raccogli tutte le categorie
    all_phases = []
    all_steps = []
    
    for dataset in datasets:
        all_phases.extend(dataset.get('phases_categories', []))
        all_steps.extend(dataset.get('steps_categories', []))
    
    # Unisci le categorie (rimuovi duplicati)
    merged_data['phases_categories'] = merge_categories(all_phases)
    merged_data['steps_categories'] = merge_categories(all_steps)
    
    print(f"   ✓ Fasi (phases_categories): {len(merged_data['phases_categories'])}")
    print(f"   ✓ Step (steps_categories): {len(merged_data['steps_categories'])}")
    
    # Unisci immagini e annotazioni con indici corretti
    image_id_map = {}  # Mappa dai vecchi ID ai nuovi ID
    next_image_id = 1
    next_annotation_id = 1
    file_name_to_dataset = {}  # Per tracciare da quale dataset proviene ogni immagine
    
    for dataset_idx, dataset in enumerate(datasets):
        dataset_name = json_files[dataset_idx].stem
        print(f"\n   Processing {dataset_name}...")
        
        # Crea mapping per gli image_id
        old_to_new_image_id = {}
        for old_image in dataset.get('images', []):
            old_id = old_image['id']
            old_to_new_image_id[old_id] = next_image_id
            
            # Copia l'immagine e aggiorna l'ID
            new_image = old_image.copy()
            new_image['id'] = next_image_id
            new_image['source_dataset'] = dataset_name
            
            merged_data['images'].append(new_image)
            file_name_to_dataset[old_image.get('file_name', '')] = dataset_name
            
            next_image_id += 1
        
        # Unisci annotazioni con ID remappati
        for old_annotation in dataset.get('annotations', []):
            new_annotation = old_annotation.copy()
            new_annotation['id'] = next_annotation_id
            
            # Rimappa l'image_id
            old_image_id = old_annotation.get('image_id')
            if old_image_id in old_to_new_image_id:
                new_annotation['image_id'] = old_to_new_image_id[old_image_id]
            
            new_annotation['source_dataset'] = dataset_name
            merged_data['annotations'].append(new_annotation)
            
            next_annotation_id += 1
        
        print(f"      ✓ Images: {len(old_to_new_image_id)}")
        print(f"      ✓ Annotations: {len(dataset.get('annotations', []))}")
    
    # Aggiorna info con statistiche finali
    merged_data['info']['total_images'] = len(merged_data['images'])
    merged_data['info']['total_annotations'] = len(merged_data['annotations'])
    merged_data['info']['phases_count'] = len(merged_data['phases_categories'])
    merged_data['info']['steps_count'] = len(merged_data['steps_categories'])
    
    # Salva il file unito
    output_file = output_coco_dir / 'all_merged.json'
    print(f"\n💾 Salvataggio del dataset unito...")
    save_json_file(merged_data, output_file)
    
    # Stampa statistiche finali
    print("\n" + "="*60)
    print("📊 STATISTICHE FINALI")
    print("="*60)
    print(f"Total Images: {merged_data['info']['total_images']}")
    print(f"Total Annotations: {merged_data['info']['total_annotations']}")
    print(f"Phases Categories: {merged_data['info']['phases_count']}")
    print(f"Steps Categories: {merged_data['info']['steps_count']}")
    print(f"Source Files: {merged_data['info']['total_files_merged']}")
    print(f"Merge Date: {merged_data['info']['merge_date']}")
    print("="*60 + "\n")



    
    
    print("✨ Merge completato con successo!")
