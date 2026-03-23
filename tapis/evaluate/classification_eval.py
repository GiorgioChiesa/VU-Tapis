import itertools
import numpy as np
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_recall_curve, balanced_accuracy_score ,average_precision_score, f1_score, roc_auc_score
from tqdm import tqdm
import wandb
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
import os
import logging
from collections import Counter
import json
import cv2

def eval_classification(task, coco_anns, preds, **kwargs):
    if kwargs.get("class_names", None) is not None:
        classes = kwargs["class_names"]
    elif isinstance(coco_anns,dict) and f'{task}_categories' in coco_anns:
        classes = coco_anns[f'{task}_categories']
    else:
        raise ValueError(f"Class names not found for task {task} in coco annotations. Please provide class_names in kwargs.")
    classes = sorted(classes, key=lambda x: x['id'])
    num_classes = len(classes)
    bin_labels = []
    bin_preds = []
    ann_preds_dict = {}
    image_not_found_count = 0
    
    if isinstance(coco_anns, dict) and "annotations" in coco_anns and "images" in coco_anns:
        # Create annotation lookup by image name
        ann_by_image = {}
        for idx, ann in enumerate(coco_anns["annotations"]):
            img_name = ann["image_name"]
            if img_name not in ann_by_image:
                ann_by_image[img_name] = []
            ann_by_image[img_name].append((idx, ann))
        
        # Loop through predictions (typically fewer than annotations)
        for img_name, pred_data in tqdm(preds.items()):
            if img_name not in ann_by_image:
                continue
                
            these_probs = pred_data[f'{task}_score_dist']
            if len(these_probs) == 0:
                print(f"Prediction not found for image {img_name}")
                these_probs = np.zeros((1, num_classes))
            
            video = img_name.split('/')[0]
            
            for idx, ann in ann_by_image[img_name]:
                ann_class = int(ann[task])
                bin_labels.append(label_binarize([ann_class], classes=list(range(0, num_classes)))[0])
                bin_preds.append(these_probs)
                
                if video in ann_preds_dict:
                    ann_preds_dict[video].append((ann_class, np.argmax(these_probs)))
                else:
                    ann_preds_dict[video] = [(ann_class, np.argmax(these_probs))]
        
        image_not_found_count = len(coco_anns["annotations"]) - len(bin_labels)
        print(f"Total images not found in predictions: {image_not_found_count}")
        print(f"Total for calculating metrics: {len(bin_labels)}")
    else:
        bin_labels = label_binarize(coco_anns, classes=list(range(0, num_classes)))
        bin_preds = preds
    # Count missing annotations

    
    bin_labels = np.array(bin_labels)
    bin_preds = np.array(bin_preds)
    precision = {}
    recall = {}
    threshs = {}
    ap = {}
    auc = {}
    acc = {}
    for c in range(0, num_classes):
        precision[c], recall[c], threshs[c] = precision_recall_curve(bin_labels[:, c], bin_preds[:,c]  )
        auc[c] = roc_auc_score(bin_labels[:, c], bin_preds[:, c])
        acc[c] = balanced_accuracy_score(bin_labels[:, c], bin_preds[:, c]>0.5)
        ap[c] = average_precision_score(bin_labels[:, c], bin_preds[:, c])

    mAP = np.nanmean(average_precision_score(bin_labels, bin_preds))
    best_preds = np.argmax(bin_preds, axis=1)
    best_labels = np.argmax(bin_labels, axis=1)
    mACC = np.nanmean(balanced_accuracy_score(best_labels, best_preds))
    mAUC = np.nanmean(roc_auc_score(bin_labels, bin_preds))
    mf1 = np.nanmean(f1_score(best_labels, best_preds, average='macro'))

    cat_names = [f"{cat['name']}-AP" for cat in classes]
    cat_res_dict = dict(zip(cat_names,list(ap.values())))

    cat_res_dict.update({f"{cat['name']}-AUC": auc[c] for c, cat in enumerate(classes)})
    cat_res_dict.update({f"{cat['name']}-ACC": acc[c] for c, cat in enumerate(classes)})
    

    f1_dict = {}
    auc_dict = {}
    acc_dict = {}
    for video, anns_preds in ann_preds_dict.items():
        np_anns_preds = np.array(anns_preds)
        anns = np_anns_preds[:,0]
        preds = np_anns_preds[:,1]
        f1_dict[video] = f1_score(anns, preds, average='macro')
        acc_dict[video] = balanced_accuracy_score(anns, preds)
    
    # f1 = np.nanmean(list(f1_dict.values()))
    # acc = np.nanmean(list(acc_dict.values()))
    
    cat_res_dict.update({f"{task}_mAP":mAP})
    cat_res_dict.update({f"{task}_f1_score":mf1})
    cat_res_dict.update({f"{task}_balanced_accuracy":mACC})
    cat_res_dict.update(f1_dict)

    # create confusion matrix
    save_confusion_matrix(preds=bin_preds, 
                          labels=bin_labels, 
                          class_name=classes, 
                          path=os.path.join(kwargs.get("output_dir", "./temp"), f"confusion_matrix_{task}.png"),
                          normalize='true' if task == "steps" else None)
    
    # save_missmatches(preds=best_preds, 
    #                  labels=best_labels, 
    #                  task=task, 
    #                  class_name=classes, 
    #                  output_dir=kwargs.get("output_dir", "./temp"),
    #                  img_ann_dict=kwargs.get("img_ann_dict", None),
    #                  imgs_folder=kwargs.get("imgs_folder", None))
    
    
    return mAP, cat_res_dict


def save_missmatches(preds, labels, task, class_name, output_dir="./temp", img_ann_dict:list=None, imgs_folder:str=None, **kwargs):
    if img_ann_dict is None:
        print("Image annotation dictionary not provided, cannot save mismatches with image names.")
        return
    
    preds = np.array(preds, dtype=np.int32)
    if len(preds.shape) == 2 and preds.shape[1] > 1:
        preds = np.argmax(preds, axis=1)
    labels = np.array(labels, dtype=np.int32)

    mismatches = []
    mismatches_by_video = {}
    consecutive_groups={}
    prev_status = None
    start_video = 0

    for i, (pred_class, true_class, img_path) in enumerate(sorted(zip(preds, labels, img_ann_dict), key=lambda x: x[2])):
        video = img_path.split('/')[0]
        if video not in consecutive_groups:
            consecutive_groups[video] = []
        if video not in mismatches_by_video:
            mismatches_by_video[video] = set()
        if pred_class != true_class:
            mismatches.append({
                "idx":i,
                "image_name": img_path,
                "video": video,
                "predicted_class": class_name[pred_class]['name'] if 0 <= pred_class < len(class_name) else "Unknown",
                "true_class": class_name[true_class]['name'] if 0 <= true_class < len(class_name) else "Unknown",
                "predicted_class_id": int(pred_class),
                "true_class_id": int(true_class)
            })

            mismatches_by_video[video].add(i)
        
        curr_status = (pred_class == true_class, video)
        if prev_status is None:
            prev_status = curr_status
            start = 0
            length = 1
        elif curr_status != prev_status :
            consecutive_groups[prev_status[1]].append((start, length, prev_status[0]))
            if prev_status[1] != curr_status[1]:
                start_video = i
            start = i - start_video
            length = 1
            prev_status = curr_status
        length +=1     
            
    mismatch_path = os.path.join(output_dir,"missmatches", f"{task}.json")
    os.makedirs(os.path.dirname(mismatch_path), exist_ok=True)
    with open(mismatch_path, 'w') as f:
        json.dump(mismatches, f, indent=4)
    
    plot_video_missmatches(mismatches_by_video, task, output_dir, max_idx=i)
    if imgs_folder is not None:
        max_save_video = kwargs.get("max_save_video", 0)
        save_missmatches_videos(mismatches, output_dir, imgs_folder, max_save_video)
    
    
def save_missmatches_videos(mismatches, output_dir, imgs_folder, max_video:int=0):
    """
    Save mismatch frames as videos organized by error type.
    """
    if imgs_folder is None or mismatches is None:
        return
    if max_video <= 0:
        return
    
    video_saved = 0
    
    # Group mismatches by error type (pred_class, true_class)
    errors_by_type = {}
    for mismatch in mismatches:
        error_key = f"{mismatch['predicted_class']}--{mismatch['true_class']}"
        if error_key not in errors_by_type:
            errors_by_type[error_key] = []
        errors_by_type[error_key].append(mismatch)
    
    # Process each error type
    for error_type, error_mismatches in errors_by_type.items():
        error_dir = os.path.join(output_dir, "missmatches", error_type)
        os.makedirs(error_dir, exist_ok=True)
        
        # Group mismatches by video
        by_video = {}
        for mismatch in error_mismatches:
            video = mismatch['video']
            if video not in by_video:
                by_video[video] = []
            by_video[video].append(mismatch)
        
        # Process each video
        for video, video_mismatches in by_video.items():
            # Sort by index
            video_mismatches = sorted(video_mismatches, key=lambda x: x['idx'])
            
            # Group consecutive frames
            video_groups = []
            current_group = [video_mismatches[0]]
            
            for mismatch in video_mismatches[1:]:
                if mismatch['idx'] - current_group[-1]['idx'] <= 1:
                    current_group.append(mismatch)
                else:
                    video_groups.append(current_group)
                    current_group = [mismatch]
            video_groups.append(current_group)
            
            # Create video for each group
            for group in video_groups:
                start_idx = group[0]['idx']
                end_idx = group[-1]['idx']
                
                # Build frame list with padding
                frame_indices = []
                
                # Add 16 frames before if exist
                padding_start = 0
                frame_indices.extend(range(start_idx - padding_start, start_idx))
                
                # Add actual frames
                frame_indices.extend(range(start_idx, end_idx + 1))
                
                # Repeat first frame 16 times
                frame_indices = [idx if idx >= 0 else 0 for idx in frame_indices]  # Handle negative indices:
                
                # Load frames and create video
                frames = []
                for f, frame_idx in enumerate(frame_indices):
                    #TODO: ricerca del frame è sbagliata
                    frame_path = os.path.join(imgs_folder, group[f]['image_name'])
                    try:
                        frame = cv2.imread(frame_path)
                        if frame is not None:
                            frames.append(frame)
                        else:
                            frame_path = frame_path.replace("/nas_private/","/").replace("/orsi/","/orsi_tensors/").replace(".jpg", ".pt")
                    except Exception as e:
                        logging.warning(f"Could not load frame {frame_path}: {e}")

                # Save video if frames exist
                if frames and len(frames):
                    video_name = f"{video}_{start_idx}_{end_idx}.mp4"
                    video_path = os.path.join(error_dir, video_name)
                    
                    h, w = frames[0].shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out = cv2.VideoWriter(video_path, fourcc, 1.0, (w, h))
                    
                    for frame in frames:
                        out.write(frame)
                    out.release()
                    video_saved +=1
                    if max_video and video_saved >= max_video:
                        return
    


def plot_video_missmatches(mismatches_by_video:dict, task:str, output_dir:str="./temp", max_idx:int=1, **kwargs):
        # Create bar plot visualization
        # Get all videos and create figure
        videos = sorted(mismatches_by_video.keys())
        fig, ax = plt.subplots(figsize=(16, max(4, len(videos) * 0.8)))
        
        # Plot each video as a horizontal bar
        for i, video in enumerate(videos):
            mismatch_indices = mismatches_by_video[video]
            # max_idx = max([m['idx'] for m in mismatches if m['video'] == video])
            
            # Create array: 1 for correct, 0 for mismatch
            frame_status = np.ones(max_idx + 1)
            for idx in mismatch_indices:
                frame_status[idx] = 0
            
            # Plot consecutive frames with same status as single bars
            # Group consecutive frames by status
            consecutive_groups = []
            j = 0
            while j < len(frame_status):
                start = j
                current_status = frame_status[j]
                while j < len(frame_status) and frame_status[j] == current_status:
                    j += 1
                consecutive_groups.append((start, j - start, current_status))
            
            # Plot each group as a single bar
            for start, length, status in consecutive_groups:
                color = 'red' if status == 0 else 'green'
                alpha = 0.7 if status == 0 else 0.3
                ax.barh(i, length, left=start, color=color, alpha=alpha, height=0.6)
        
        ax.set_yticks(range(len(videos)))
        ax.set_yticklabels(videos)
        ax.set_xlabel('Frame Index')
        ax.set_title(f'Frame Classification Results by Video ({task})')
        ax.legend(['Correct', 'Mismatch'], loc='upper right')
        
        # Save plot
        plot_path = os.path.join(output_dir, "missmatches", f"{task}_frames_plot.png")
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        plt.savefig(plot_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
        
def save_confusion_matrix(preds, labels, class_name:list=[],path:str="./temp/confusion_matrix.png", **kwargs):
    if len(preds) == 0 or len(labels) == 0:
        print("No predictions or labels provided for confusion matrix.")
        return None
    if preds.shape[0] != labels.shape[0]:
        print("Number of predictions and labels do not match for confusion matrix.")
        return None
    if len(preds.shape) == 2 and preds.shape[1] > 1:
        preds = np.argmax(preds, axis=1)
    if len(labels.shape) == 2 and labels.shape[1] > 1:
        labels = np.argmax(labels, axis=1)
    cm = confusion_matrix(labels, preds , normalize=kwargs.get("normalize", None))
    
    if len(class_name) >0 and isinstance(class_name[0], dict):
        class_names = [cla['name'] for cla in class_name if 'name' in cla]
    else:
        class_names = [str(i) for i in range(cm.shape[0])]
    
    # Count occurrences of each class in predictions and labels
    all_classes = np.concatenate([np.argmax(preds, axis=1) if len(preds.shape) == 2 and preds.shape[1] > 1 else preds,
                                   np.argmax(labels, axis=1) if len(labels.shape) == 2 and labels.shape[1] > 1 else labels])
    class_counts = Counter(all_classes)
    
    # Remove class names not present in data
    class_names = [name for i, name in enumerate(class_names) if i in class_counts]

    # Pad confusion matrix to match number of classes
    # padded_cm = np.zeros((len(class_names), len(class_names)))
    # padded_cm[:cm.shape[0], :cm.shape[1]] = cm
    # cm = padded_cm
    
    assert cm.shape[0] == len(class_names), "Confusion matrix should be square"
    
    fig, ax = plt.subplots(figsize=(cm.shape[1]+5, cm.shape[0]))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    # Aggiungi ticks e labels
    ax.set(xticks=np.arange(cm.shape[1]),
            yticks=np.arange(cm.shape[0]),
            xticklabels=[i for i, _ in enumerate(class_names)], 
            yticklabels=[i for i, _ in enumerate(class_names)],
            title=f"Confusion Matrix",
            ylabel='True label',
            xlabel='Predicted label')

    # # Ruota i tick labels
    # plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
    #             rotation_mode="anchor")
    
    # Aggiungi legenda per i nomi delle classi
    legend_labels = [f"{i}: {name}" for i, name in enumerate(class_names)]
    ax.text(1.25, 1, "\n".join(legend_labels), transform=ax.transAxes, 
            fontsize=15, verticalalignment='top', bbox=dict(boxstyle='round', 
            facecolor='wheat', alpha=0.5))


    fig.tight_layout()

    # Crea la cartella se non esiste
    cm_dir = os.path.dirname(path)
    os.makedirs(cm_dir, exist_ok=True)

    # Salva l'immagine
    plt.savefig(path)
    plt.close(fig)
    
def eval_presence(task, coco_anns, preds, img_ann_dict, **kwargs):
    classes = coco_anns[f'{task}_categories']
    num_classes = len(classes)
    bin_labels = np.zeros((len(coco_anns["images"]), num_classes))
    bin_preds = np.zeros((len(coco_anns["images"]), num_classes))
    evaluated_frames = []
    for idx, img in tqdm(enumerate(coco_anns["images"]), total=len(coco_anns["images"])):
        binary_task_label = np.zeros(num_classes+1, dtype='uint8')
        label_list = [coco_anns["annotations"][idx][task] for idx in img_ann_dict[img['file_name']]]
        assert all(type(label_list[0])==type(lab_item) for lab_item in label_list), f'Inconsistent label type {label_list} in frame {img["file_name"]}'
        
        if isinstance(label_list[0], list):
            label_list = list(set(itertools.chain(*label_list)))
        elif isinstance(label_list[0], int):
            label_list = list(set(label_list))
        else:
            raise ValueError(f'Do not support annotation {label_list[0]} of type {type(label_list[0])} in frame {img["file_name"]}')
        binary_task_label[label_list] = 1
        ann_classes = binary_task_label[1:].tolist()
        bin_labels[idx, :] = ann_classes
        

        if  img["file_name"] in preds.keys():            
            these_probs = preds[img["file_name"]]['{}_score_dist'.format(task)]
            
            if len(these_probs) == 0:
                print("Prediction not found for image {}".format(img["file_name"]))
                these_probs = np.zeros((1, num_classes))
            else:
                evaluated_frames.append(idx)
            bin_preds[idx, :] = these_probs
            
        else:
            print("Image {} not found in predictions lists".format(img["file_name"]))
            
    bin_labels = bin_labels[evaluated_frames]
    bin_preds = bin_preds[evaluated_frames]
    
    precision = {}
    recall = {}
    threshs = {}
    ap = {}
    for c in range(0, num_classes):
        precision[c], recall[c], threshs[c] = precision_recall_curve(bin_labels[:, c], bin_preds[:, c])
        ap[c] = average_precision_score(bin_labels[:, c], bin_preds[:, c])

    mAP = np.nanmean(list(ap.values()))
    
    cat_names = [f"{cat['name']}-AP" for cat in classes]
    cat_res_dict = dict(zip(cat_names,list(ap.values())))
            
    return mAP, cat_res_dict