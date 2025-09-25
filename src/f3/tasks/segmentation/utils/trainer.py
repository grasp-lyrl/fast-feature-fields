import torch
import wandb
from tqdm import tqdm

from f3.utils import crop_params


def train_fixed_time_segmentation(args, model, train_loader, optimizer, scheduler, loss_fn,
                                  epoch, logger=None, scaler=None, iters_to_accumulate=1):
    model.train()
    train_loss, train_miou, train_acc = 0, 0, 0
    iter_loss, iter_miou, iter_acc,  = 0, 0, 0
    for idx, data in tqdm(enumerate(train_loader), total=len(train_loader)):
        # [(B,N,3) or (B,N,4)], [(B,W,H,2) or (B,W,H,T,2)], [(B,H,W,1)] #! T: max prediction time bins time_pred//bucket
        ff_events, event_counts, _, semantic_labels, src_ofst_res = data
        
        ff_events = ff_events.cuda()
        event_counts = event_counts.cuda()
        semantic_labels = semantic_labels.cuda().long()

        if not args.polarity[0]: ff_events = ff_events[..., :3]

        # i, j, i+H, j+W (H, W) are the crop sizes
        cparams = crop_params(src_ofst_res, torch.tensor(args.random_crop['size'], dtype=torch.int32))
        
        semantic_labels = torch.stack([
            semantic_labels[i, cparams[i, 0]:cparams[i, 2], cparams[i, 1]:cparams[i, 3]]
            for i in range(semantic_labels.shape[0])
        ])

        with torch.autocast(device_type="cuda", enabled=args.amp, dtype=torch.float16):
            seg_logits = model(ff_events, event_counts, cparams)[0] # (B,N,3) -> (B,C,H,W)
            loss = loss_fn(seg_logits, semantic_labels)
            loss /= iters_to_accumulate
        scaler.scale(loss).backward()

        train_loss += loss.item()
        iter_loss += loss.item()

        if (idx+1) % iters_to_accumulate == 0:
            train_acc += iter_acc
            train_miou += iter_miou

            logger.info(f"Epoch: {epoch}, Idx: {(idx+1)//iters_to_accumulate}, Loss: {iter_loss}, " +\
                        f"Acc: {iter_acc}, MIoU: {iter_miou}")

            if args.wandb:
                wandb.log({"acc": iter_acc, "loss": iter_loss, "miou": iter_miou,
                           "lr": scheduler.get_last_lr()[0]})
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            iter_loss, iter_miou, iter_acc,  = 0, 0, 0
    # normalize separately because the effective batch size is still based on train_batch
    train_loss /= (len(train_loader) // iters_to_accumulate)
    train_acc /= (len(train_loader) // iters_to_accumulate)
    train_miou /= (len(train_loader) // iters_to_accumulate)

    logger.info("#"*50)
    logger.info(f"Training: Epoch: {epoch}, Loss: {train_loss}, Acc: {train_acc}, MIoU: {train_miou}")
    logger.info("#"*50)

    if args.wandb:
        wandb.log({"train_acc": train_acc, "train_loss": train_loss,
                   "train_miou": train_miou, "epoch": epoch})
