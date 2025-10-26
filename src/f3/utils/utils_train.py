import torch
from tqdm import tqdm
from copy import deepcopy

from f3.utils import ev_to_grid, tp_tn_fp_fn, acc_f1


def train_fixed_time(args, eff, train_loader, optimizer, scheduler, epoch,
                     logger=None, accelerator=None, iters_to_accumulate=1):
    train_loss, train_acc, train_f1 = 0, 0, 0
    tp, tn, fp, fn, iter_loss = 0, 0, 0, 0, 0
    pred_frame_size = deepcopy(args.frame_sizes) + [args.time_pred // args.bucket]

    # Update pred_frame_size if cropping/resizing is enabled
    crop_resize_training = getattr(args, 'random_crop_resize', None)
    if crop_resize_training is not None:
        resize_size = crop_resize_training.get('resize', None)
        if resize_size is not None:
            pred_frame_size = resize_size + [args.time_pred // args.bucket]
        else:
            crop_size = crop_resize_training['crop']
            pred_frame_size = crop_size + [args.time_pred // args.bucket]

    for idx, data in tqdm(enumerate(train_loader), total=len(train_loader), disable=not accelerator.is_local_main_process):
        ff_events, pred_events, ff_counts, pred_counts, valid_mask = data # (N,3) or (N,4), (B), (B,W,H,2) or (B,W,H,T,2), (B, W, H) #! T: max prediction time bins time_pred//bucket

        if not args.polarity[0]: ff_events = ff_events[..., :3]
        if not args.polarity[1]: pred_events = pred_events[..., :3]

        pred_event_grid = ev_to_grid(pred_events, pred_counts, *pred_frame_size)

        with accelerator.accumulate(eff):
            predlogits, loss = eff(ff_events, ff_counts, pred_event_grid, valid_mask) # (B,N,3) -> (B,W,H,1) or (B,W,H,T)
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        if accelerator.is_local_main_process:
            train_loss += loss.item() / iters_to_accumulate
            iter_loss += loss.item() / iters_to_accumulate

            predictions = torch.sigmoid(predlogits)
            tp, tn, fp, fn = [sum(x) for x in zip((tp, tn, fp, fn), tp_tn_fp_fn(predictions, pred_event_grid))]

            if (idx+1) % iters_to_accumulate == 0:
                acc, f1 = acc_f1(tp, tn, fp, fn)
                train_acc += acc
                train_f1 += f1
                logger.info(f"Epoch: {epoch}, Idx: {(idx+1)//iters_to_accumulate}, Loss: {iter_loss}, " +\
                            f"Acc: {acc}, pos_acc: {tp/(tp+fn)}, neg_acc: {tn/(tn+fp)}")
                if args.wandb:
                    accelerator.log({"acc": acc, "loss": iter_loss, "neg_acc": tn/(tn+fp),
                                     "pos_acc": tp/(tp+fn), "lr": scheduler.get_last_lr()[0]})
                tp, tn, fp, fn, iter_loss = 0, 0, 0, 0, 0

    if accelerator.is_local_main_process:
        # normalize separately because the effective batch size is still based on train_batch
        train_loss /= (len(train_loader) // iters_to_accumulate)
        train_acc /= (len(train_loader) // iters_to_accumulate)
        train_f1 /= (len(train_loader) // iters_to_accumulate)

        logger.info("#"*50)
        logger.info(f"Training: Epoch: {epoch}, Loss: {train_loss}, Acc: {train_acc}, F1: {train_f1}")
        logger.info("#"*50)

        if args.wandb:
            accelerator.log({"train_acc": train_acc, "train_loss": train_loss, "train_f1": train_f1, "epoch": epoch})
