#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Train a video classification model."""

from asyncio import tasks
import sys
from pathlib import Path

# Add parent directory to path so tapis can be imported
if Path(__file__).parent.parent not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent))
if Path(__file__).parent.parent.joinpath('detectron2') not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent.joinpath('detectron2')))

import random
import numpy as np
import shutil
import os
import pprint
import torch

import tapis.models.losses as losses
import tapis.models.optimizer as optim
import tapis.utils.checkpoint as cu
import tapis.utils.distributed as du
import tapis.utils.logging as logging
import tapis.utils.misc as misc

from tapis.datasets import loader
from tapis.models import build_model
from tapis.utils.meters import EpochTimer, SurgeryMeter
import torch.backends.cudnn as cudnn
import torch.backends.cudnn
import wandb
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from pyinstrument import Profiler


logger = logging.get_logger(__name__)
wanbrun = None

def wandgb_log(stats):
    global wanbrun
    if wanbrun is not None:
        tasks_map = [k for k, v in stats.items() if isinstance(v,dict)]
        if len(tasks_map)==0:
            stats = {f'{stats["mode"]}_{k}': v for k, v in stats.items()}
            wanbrun.log(stats)
            return
        if stats["mode"].lower() in ['test', 'val']:
            cur_epoch = int(stats["cur_epoch"])
            stat = {}
            for tasks in tasks_map:
                stat = {}
                for k, v in stats[tasks].items():
                    stat[f'{stats["mode"]}_{k}'] = v
                # stats ={f'{stats["mode"]}_{k}': v for k, v in stats["phases_map"].items()}
                stat["cur_epoch"] = int(cur_epoch)
                stat[f"{tasks}_cm"] = wandb.Image(os.path.join(stats["output_dir"], f"confusion_matrix_{tasks.split('_')[0]}.png"))
                wanbrun.log(stat)
            
        elif stats["mode"].lower() == 'train':
            cur_epoch = int(stats["cur_epoch"])
            stats = {f'{stats["mode"]}_{k}': v for k, v in stats.items()}
            stats["cur_epoch"] = int(cur_epoch)
            wanbrun.log(stats)
        
        


def log_confusion_matrix_wandb(meter, task, mode, epoch, mean_map, name_wandb, cfg):
    """
    Crea e salva una confusion matrix come immagine, poi la logga su wandb come artifact.

    Args:
        pred (list): Lista di liste con le probabilità per ogni classe.
                     Formato: [[prob_class0, prob_class1, ...], ...]
        labels (list): Lista con le etichette vere per ogni campione.
                       Formato: [class_idx, class_idx, ...]
        task (str): Nome del task.
        mode (str): Modalità di training ('train', 'val', 'test').
        epoch (int): Numero dell'epoca corrente.
        cfg: Configurazione del modello.
    """
    global wanbrun

    try:
        pred = meter.all_preds[task]
        labels = meter.all_labels[task]
        
        # Converti le probabilità in predizioni tramite argmax
        pred_classes = np.argmax(np.array(pred), axis=1)

        # Calcola la confusion matrix
        cm = confusion_matrix(labels, pred_classes )

        # Crea una lista di nomi delle classi
        num_classes = cm.shape[0]
        # class_names = list(set([v.split("-")[0] for v in meter.full_map]))
        class_names = [f"Class {i}" for i in range(num_classes)]
        # class_names = []
        # for name in meter.full_map[task]:
        #     if "RARP" in name:
        #         continue
        #     class_names.append("-".join(name.split("-")[:-1]))
        # class_names = [n for n in set(class_names) if n]

        # Crea la figura
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)

        # Aggiungi ticks e labels
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=class_names, yticklabels=class_names,
               title=f"Confusion Matrix - {task} ({mode}) - Epoch {epoch} - mAP: {mean_map:.4f}",
               ylabel='True label',
               xlabel='Predicted label')

        # Ruota i tick labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")

        # # Aggiungi i valori nella matrice
        # thresh = cm.max() / 2.
        # for i in range(cm.shape[0]):
        #     for j in range(cm.shape[1]):
        #         ax.text(j, i, format(cm[i, j], 'd'),
        #                 ha="center", va="center",
        #                 color="white" if cm[i, j] > thresh else "black")

        fig.tight_layout()

        # Crea la cartella se non esiste
        cm_dir = os.path.join(cfg.OUTPUT_DIR, "confusion_matrix")
        os.makedirs(cm_dir, exist_ok=True)

        # Salva l'immagine
        cm_path = os.path.join(cm_dir, f"cm_{task}_{mode}_epoch_{epoch}.png")
        plt.savefig(cm_path)
        plt.close(fig)

        # Log come artifact su wandb
        if name_wandb and wanbrun is not None:
            artifact_name = f"confusion_matrix_{task}_{mode}_{name_wandb}_epoch_{epoch}"
            # artifact = wandb.Artifact(name=artifact_name, type="confusion_matrix")
            # artifact.add_file(cm_path)
            # wanbrun.log_artifact(artifact)
            wandb.save(artifact_name, base_path=cm_dir)

    except Exception as e:
        logger.warning(f"Errore nel logging della confusion matrix per task {task}: {e}")


def train_epoch(
    train_loader,
    model,
    optimizer,
    scaler,
    train_meter,
    cur_epoch,
    cfg,
):
    """
    Perform the video training for one epoch.
    Args:
        train_loader (loader): video training loader.
        model (model): the video model to train.
        optimizer (optim): the optimizer to perform optimization on the model's
            parameters.
        train_meter (TrainMeter): training meters to log the training performance.
        cur_epoch (int): current epoch of training.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    # Enable train mode.
    model.train()
    train_meter.iter_tic()
    data_size = len(train_loader)
    tasks = cfg.TASKS.TASKS
    loss_funs = cfg.TASKS.LOSS_FUNC
    
    weight = {task: losses.get_weight_from_csv(cfg.TASKS.WEIGHT_LOSS_BY_CLASS[t_id], cfg.TASKS.NUM_CLASSES[t_id]) for t_id, task in enumerate(tasks)}
    if cfg.NUM_GPUS:
        weight = {task: weight[task].to("cuda") if weight[task] is not None else None for task in weight}
    loss_dict = {task:losses.get_loss_func(loss_funs[t_id])( weight=weight[task], reduction=cfg.SOLVER.REDUCTION) for t_id,task in enumerate(tasks)}
    type_dict = {task:losses.get_loss_type(loss_funs[t_id],cfg.MODEL.PRECISION) for t_id,task in enumerate(tasks)}
    loss_weights = cfg.TASKS.LOSS_WEIGHTS
    if cfg.REGIONS.ENABLE and cfg.TASKS.PRESENCE_RECOGNITION:
        pres_loss_dict = {f'{task}_presence':losses.get_loss_func('bce')(reduction=cfg.SOLVER.REDUCTION) for task in cfg.TASKS.PRESENCE_TASKS}
        pres_type_dict = {f'{task}_presence':losses.get_loss_type('bce') for task in cfg.TASKS.PRESENCE_TASKS}
        loss_dict.update(pres_loss_dict)
        type_dict.update(pres_type_dict)
        loss_weights += cfg.TASKS.PRESENCE_WEIGHTS

    with tqdm(total=data_size, desc=f"Train Epoch {cur_epoch+1}/{cfg.SOLVER.MAX_EPOCH}", unit="it") as t:     
        for cur_iter, (inputs, labels, data, image_names) in enumerate(train_loader):
            t.update(1)
            # Transfer the data to the current GPU device.
            if cfg.NUM_GPUS:
                inputs = [input.cuda(non_blocking=True) for input in inputs]
                if cfg.MODEL.PRECISION == 64:
                    inputs[0] = inputs[0].double()

                for key, val in data.items():
                    data[key] = val.cuda(non_blocking=True)
                    if cfg.MODEL.PRECISION == 64:
                        data[key]  = data[key].double()

                for key, val in labels.items():
                    labels[key] = val.cuda(non_blocking=True)
                    if cfg.MODEL.PRECISION == 64:
                        labels[key]  = labels[key].double()
                
                if cfg.NUM_GPUS>1:
                    image_names = image_names.cuda(non_blocking=True)
                        
            # Update the learning rate.
            lr = optim.get_epoch_lr(cur_epoch + float(cur_iter) / data_size, cfg)
            optim.set_lr(optimizer, lr)

            train_meter.data_toc()

            with torch.cuda.amp.autocast(enabled=cfg.TRAIN.MIXED_PRECISION):
                rpn_ftrs = data["rpn_features"] if cfg.FEATURES.ENABLE else None
                boxes_mask = data["boxes_mask"] if cfg.REGIONS.ENABLE else None
                boxes = data["boxes"] if cfg.REGIONS.ENABLE else None
                images = data["images"] if cfg.FEATURES.USE_RPN else None
                preds = model(inputs, bboxes=boxes, features=rpn_ftrs, boxes_mask=boxes_mask, images=images)

                # Explicitly declare reduction to mean and compute the loss for each task.
                loss = []
                for task in loss_dict:
                    loss_fun = loss_dict[task]
                    target_type = type_dict[task]
                    loss.append(loss_fun(preds[task], labels[task].to(target_type))) 

            if len(loss_dict) >1:
                final_loss = losses.compute_weighted_loss(loss, loss_weights)
            else:
                final_loss = loss[0]
                
            # check Nan Loss.
            if cur_iter % 50 == 0:
                misc.check_nan_losses(final_loss.item())
            # Perform the backward pass.
            # scaler.scale(final_loss).backward()

            # Unscales the gradients of optimizer's assigned params in-place
            # scaler.unscale_(optimizer)

            # Clip gradients if necessary
            if cfg.SOLVER.CLIP_GRAD_VAL:
                torch.nn.utils.clip_grad_value_(
                    model.parameters(), cfg.SOLVER.CLIP_GRAD_VAL
                )
            elif cfg.SOLVER.CLIP_GRAD_L2NORM:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.SOLVER.CLIP_GRAD_L2NORM
                )
                
            scaler.scale(final_loss / cfg.TRAIN.ACCUM_STEPS).backward()  # normalizza il loss

            if (cur_iter + 1) % cfg.TRAIN.ACCUM_STEPS == 0:
                scaler.unscale_(optimizer)
                if cfg.SOLVER.CLIP_GRAD_VAL:
                    torch.nn.utils.clip_grad_value_(model.parameters(), cfg.SOLVER.CLIP_GRAD_VAL)
                elif cfg.SOLVER.CLIP_GRAD_L2NORM:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.SOLVER.CLIP_GRAD_L2NORM)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            if cfg.NUM_GPUS > 1:
                final_loss = du.all_reduce([final_loss])[0]
            final_loss = final_loss.item()

            # Update and log stats.
            train_meter.update_stats(None, None, None, final_loss, loss, lr)
            train_meter.iter_toc()  # measure allreduce for this meter
            stats = train_meter.log_iter_stats(cur_epoch, cur_iter)
            train_meter.iter_tic()
            t.set_postfix(stats.items())

            if cfg.SOLVER.MAX_ITER and cur_iter+1 >= cfg.SOLVER.MAX_ITER:
                break
            
    t.close()
    # Log epoch stats.
    train_meter.log_epoch_stats(cur_epoch)

    # Log confusion matrix for each task
    if cfg.WANDB_ENABLE:
        wandgb_log(stats)
    train_meter.reset()

@torch.no_grad()
def eval_epoch(val_loader, model, val_meter, cur_epoch, cfg):
    """
    Evaluate the model on the val set.
    Args:
        val_loader (loader): data loader to provide validation data.
        model (model): model to evaluate the performance.
        val_meter (ValMeter): meter instance to record and calculate the metrics.
        cur_epoch (int): number of the current epoch of training.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """

    # Evaluation mode enabled. The running stats would not be updated.
    model.eval()
    val_meter.iter_tic()
    complete_tasks = cfg.TASKS.TASKS
    region_tasks = {task for task in cfg.TASKS.TASKS if cfg.REGIONS.ENABLE and task in cfg.ENDOVIS_DATASET.REGION_TASKS}
    if cfg.REGIONS.ENABLE:
        if cfg.TASKS.PRESENCE_RECOGNITION and cfg.TASKS.EVAL_PRESENCE:
            pres_tasks = [f'{task}_presence' for task in cfg.TASKS.PRESENCE_TASKS]
            complete_tasks += pres_tasks
    with tqdm(total=len(val_loader), desc=f"Eval Epoch {cur_epoch+1}/{cfg.SOLVER.MAX_EPOCH}", unit="it") as t:
        for cur_iter, (inputs, labels, data, image_names) in enumerate(val_loader):
            t.update(1)
            if cfg.NUM_GPUS:
                inputs[0] = inputs[0].cuda(non_blocking=True)
                if cfg.MODEL.PRECISION == 64:
                    inputs[0] = inputs[0].double()

                for key, val in data.items():
                    data[key] = val.cuda(non_blocking=True)
                    if cfg.MODEL.PRECISION == 64:
                        data[key]  = data[key].double()

                for key, val in labels.items():
                    labels[key] = val.cuda(non_blocking=True)
                    if cfg.MODEL.PRECISION == 64:
                        labels[key]  = labels[key].double()
                
                if cfg.NUM_GPUS>1:
                    image_names = image_names.cuda(non_blocking=True)
                        
            val_meter.data_toc()

            rpn_ftrs = data["rpn_features"] if cfg.FEATURES.ENABLE else None
            boxes_mask = data["boxes_mask"] if cfg.REGIONS.ENABLE else None
            ori_boxes = data["ori_boxes"] if cfg.REGIONS.ENABLE else None
            boxes_idxs = data["ori_boxes_idxs"] if cfg.REGIONS.ENABLE else None
            boxes = data["boxes"] if cfg.REGIONS.ENABLE else None
            images = data["images"] if cfg.FEATURES.USE_RPN else None

            assert (not (cfg.REGIONS.ENABLE and cfg.FEATURES.ENABLE)) or len(rpn_ftrs)==len(image_names)==len(boxes), f'Inconsistent lenghts {len(rpn_ftrs)} & {len(image_names)} & {len(boxes)}'

            preds = model(inputs, bboxes=boxes, features=rpn_ftrs, boxes_mask=boxes_mask, images=images)
            # breakpoint()
            if cfg.NUM_GPUS:
                preds = {task: preds[task].to("cpu", non_blocking=True) for task in preds}
                # Accumula tutti i risultati prima di accedervi
                torch.cuda.synchronize()  # una sola sync alla fine
                ori_boxes = ori_boxes.cpu() if cfg.REGIONS.ENABLE else None
                boxes_idxs = boxes_idxs.cpu() if cfg.REGIONS.ENABLE else None
                boxes_mask = boxes_mask.cpu() if cfg.REGIONS.ENABLE else None

                if cfg.NUM_GPUS>1:
                    image_names = image_names.cpu()
                    image_names = torch.cat(du.all_gather_unaligned(image_names),dim=0).tolist()

                    preds = {task: torch.cat(du.all_gather_unaligned(preds[task]), dim=0) for task in preds}

                    if cfg.REGIONS.ENABLE:
                        ori_boxes = torch.cat(du.all_gather_unaligned(ori_boxes), dim=0)
                        boxes_mask = torch.cat(du.all_gather_unaligned(boxes_mask), dim=0)
                        idxs_gather = du.all_gather_unaligned(boxes_idxs)
                        for i in range(len(idxs_gather)):
                            idxs_gather[i]+= torch.tensor((cfg.TEST.BATCH_SIZE/cfg.NUM_GPUS)*i).long()

                        boxes_idxs = torch.cat(idxs_gather, dim=0)

            val_meter.iter_toc()

            if cfg.REGIONS.ENABLE:
                for task in region_tasks:
                    preds[task] = [preds[task][boxes_idxs==idx].tolist() for idx in range(len(boxes_mask))]
                if 'masks' in preds:
                    ori_boxes = [preds['boxes'][boxes_mask][boxes_idxs==idx].numpy().tolist() for idx in range(len(boxes_mask))]
                    preds['masks'] = [preds['masks'][boxes_mask][boxes_idxs==idx].numpy() for idx in range(len(boxes_mask))]
                else:
                    ori_boxes = [ori_boxes[boxes_idxs==idx].tolist() for idx in range(len(boxes_mask))]
            for task in complete_tasks:
                if task not in region_tasks:
                    preds[task] = preds[task].tolist()

            # Update and log stats.
            val_meter.update_stats(preds, image_names, ori_boxes, labels=labels)
            stats = val_meter.log_iter_stats(cur_epoch, cur_iter)
            val_meter.iter_tic()
            t.set_postfix(stats.items())
            
            # if cfg.SOLVER.MAX_ITER and cur_iter+1 >= cfg.SOLVER.MAX_ITER:
            #     break


    t.close()
    if cfg.NUM_GPUS > 1:
        if du.is_master_proc():
            task_map, mean_map, out_files, stats, early_stop = val_meter.log_epoch_stats(cur_epoch)
        else:
            task_map, mean_map, out_files, stats, early_stop =  [0, 0, 0, {}, False]
        torch.distributed.barrier()
    else:
        task_map, mean_map, out_files, stats, early_stop = val_meter.log_epoch_stats(cur_epoch)

    if cfg.WANDB_ENABLE and cfg.NUM_GPUS <= 1:
        wandgb_log(stats)

    return task_map, mean_map, out_files, early_stop


def train(cfg):
    """
    Train a video model for many epochs on train set and evaluate it on val set.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    
    profiler = Profiler()
    profiler.start()
    # Set up environment.
    du.init_distributed_training(cfg)
    # Set random seed from configs.
    random.seed(cfg.RNG_SEED)
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)
    cudnn.benchmark = True
    cudnn.deterministic = True

    # Setup logging format.
    logging.setup_logging(cfg.OUTPUT_DIR)

    # Print config.
    logger.info("Train with config:")
    logger.info(pprint.pformat(cfg))
    if cfg.WANDB_ENABLE:
        global wanbrun
        wandb.login()
        wanbrun = wandb.init(project=cfg.WANDB_PROJECT, 
                         name=cfg.NAME,
                        entity=cfg.WANDB_ENTITY,
                        config= misc.flatten_dict(cfg))

    # Build the video model and print model statistics.
    model = build_model(cfg)
    if cfg.MODEL.PRECISION == 64:
        model = model.double()
    if cfg.TRAIN.FREEZE_ENCODER:
        model.freeze_encoder()

    # Calculating model info (param & flops). 
    # Remove if it is not working
    try:
        if du.is_master_proc() and cfg.LOG_MODEL_INFO:
            misc.log_model_info(model, cfg, use_train_input=True)
            for task in cfg.TASKS.TASKS:
                head = getattr(model, "extra_heads_{}".format(task))
                misc.log_model_info(head, cfg, use_train_input=False)
    except Exception as e:
        logger.info(f'Error while trying to calculate model parameters and FLOPs:\n{e}')

    # Construct the optimizer.
    optimizer = optim.construct_optimizer(model, cfg)
    
    # Create a GradScaler for mixed precision training
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.TRAIN.MIXED_PRECISION)

    # Load a checkpoint to resume training if applicable.
    start_epoch = cu.load_train_checkpoint(
        cfg, model, optimizer, scaler if cfg.TRAIN.MIXED_PRECISION else None
    )

    # Create the video train and val loaders.
    train_loader = loader.construct_loader(cfg, "train")
    val_loader = loader.construct_loader(cfg, "val")
    
    # Create meters.
    train_meter = SurgeryMeter(len(train_loader), cfg, mode="train")
    val_meter = SurgeryMeter(len(val_loader), cfg, mode="val")

    # Perform final test
    if cfg.TEST.ENABLE:
        logger.info("Evaluating epoch: {}".format(start_epoch))
        map_task, mean_map, out_files, _ = eval_epoch(val_loader, model, val_meter, start_epoch-1, cfg)
        val_meter.reset()
        if not cfg.TRAIN.ENABLE:
            return
    elif cfg.TRAIN.ENABLE:
        # Perform the training loop.
        logger.info("Start epoch: {}".format(start_epoch + 1))
        
    # Stats for saving checkpoint:
    complete_tasks = cfg.TASKS.TASKS
    best_task_map = {task: 0 for task in complete_tasks}
    best_mean_map = 0
    epoch_timer = EpochTimer()
    
    for cur_epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCH):

        # Shuffle the dataset.
        loader.shuffle_dataset(train_loader, cur_epoch)

        # Train for one epoch.
        epoch_timer.epoch_tic()
        train_epoch(
            train_loader,
            model,
            optimizer,
            scaler,
            train_meter,
            cur_epoch,
            cfg,
        )
        epoch_timer.epoch_toc()
        logger.info(
            f"Epoch {cur_epoch} takes {epoch_timer.last_epoch_time():.2f}s. Epochs "
            f"from {start_epoch} to {cur_epoch} take "
            f"{epoch_timer.avg_epoch_time():.2f}s in average and "
            f"{epoch_timer.median_epoch_time():.2f}s in median."
        )
        logger.info(
            f"For epoch {cur_epoch}, each iteraction takes "
            f"{epoch_timer.last_epoch_time()/len(train_loader):.2f}s in average. "
            f"From epoch {start_epoch} to {cur_epoch}, each iteraction takes "
            f"{epoch_timer.avg_epoch_time()/len(train_loader):.2f}s in average."
        )

        is_checkp_epoch = cu.is_checkpoint_epoch(
            cfg,
            cur_epoch,
            None
        )
        is_eval_epoch = misc.is_eval_epoch(
            cfg, cur_epoch, None
        )

        _ = misc.aggregate_sub_bn_stats(model)

        # Save a checkpoint.
        if is_checkp_epoch:
            cu.save_checkpoint(
                cfg.OUTPUT_DIR,
                model,
                optimizer,
                cur_epoch,
                cfg,
                scaler if cfg.TRAIN.MIXED_PRECISION else None,
            )
        
        if not cfg.MODEL.KEEP_ALL_CHECKPOINTS:
            del_fil = os.path.join(cfg.OUTPUT_DIR,'checkpoints', 'checkpoint_epoch_{0:05d}.pyth'.format(cur_epoch-1))
            if os.path.exists(del_fil):
                os.remove(del_fil)
            
        # Evaluate the model on validation set.
        if is_eval_epoch:
            map_task, mean_map, out_files, early_stop = eval_epoch(val_loader, model, val_meter, cur_epoch, cfg)
            if (cfg.NUM_GPUS > 1 and du.is_master_proc()) or cfg.NUM_GPUS == 1:
                main_path = os.path.split(list(out_files.values())[0])[0]
                fold = main_path.split('/')[-1]
                best_preds_path = main_path.replace(fold, fold+'/best_predictions')
                if not os.path.exists(best_preds_path):
                    os.makedirs(best_preds_path)
                old_best_mean_map = best_mean_map
                # Save best results
                if mean_map > best_mean_map:
                    best_mean_map = mean_map
                    logger.info("Best mean map at epoch {}".format(cur_epoch))
                    cu.save_best_checkpoint(
                        cfg.OUTPUT_DIR,
                        model,
                        optimizer,
                        'mean',
                        cfg,
                        scaler if cfg.TRAIN.MIXED_PRECISION else None,
                        )
                    for task in complete_tasks:
                        file = out_files[task].split('/')[-1]
                        copy_path = os.path.join(best_preds_path, file.replace('epoch', 'best_all') )
                        shutil.copyfile(out_files[task], copy_path)
                
                for task in complete_tasks:
                    if list(map_task[task].values())[0] > best_task_map[task]:
                        best_task_map[task] = list(map_task[task].values())[0]
                        logger.info("Best {} map at epoch {}".format(task, cur_epoch))
                        file = out_files[task].split('/')[-1]
                        copy_path = os.path.join(best_preds_path, file.replace('epoch', 'best') )
                        shutil.copyfile(out_files[task], copy_path)
                        cu.save_best_checkpoint(
                            cfg.OUTPUT_DIR,
                            model,
                            optimizer,
                            task,
                            cfg,
                            scaler if cfg.TRAIN.MIXED_PRECISION else None,
                        )
                # # Log confusion matrix for best or last epoch
                # is_best = mean_map > old_best_mean_map
                # if cfg.WANDB_ENABLE and cfg.NUM_GPUS <= 1:
                #     name_wandb = "best_epoch" if is_best else "last_epoch"
                #     for task in complete_tasks:
                #         if len(val_meter.all_preds[task]) > 0 and len(val_meter.all_labels[task]) > 0:
                #             log_confusion_matrix_wandb(
                #                 val_meter,
                #                 task,
                #                 "val",
                #                 cur_epoch,
                #                 mean_map,
                #                 name_wandb,
                #                 cfg
                #             )
                
                # Early stopping check
                if early_stop:
                    logger.info(f"Early stopping triggered after {cur_epoch + 1} epochs: no improvement >= {cfg.SOLVER.EARLY_STOP_ep_th[1]} for {cfg.SOLVER.EARLY_STOP_ep_th[0]} epochs")
                    break
            val_meter.reset()
    cu.save_checkpoint(
            cfg.OUTPUT_DIR,
            model,
            optimizer,
            cur_epoch,
            cfg,
            scaler if cfg.TRAIN.MIXED_PRECISION else None,
            )
    
    
    profiler.stop()
    profiler.print()
    profiler.output_html()
