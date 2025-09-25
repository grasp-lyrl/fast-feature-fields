import torch
import wandb
from tqdm import tqdm

from f3.utils import crop_params


def train_fixed_time_disparity(args, model, train_loader, optimizer, scheduler, loss_fn,
                           epoch, logger=None, scaler=None, iters_to_accumulate=1, pseudo_model=None):
    # Pseudo Model is used for gradient matching if loss function is SiLogGradLoss
    model.train()
    train_loss, iter_loss = 0, 0
    for idx, data in tqdm(enumerate(train_loader), total=len(train_loader)):
        # [(B,N,3) or (B,N,4)], [(B,W,H,2) or (B,W,H,T,2)], [(B,H,W,1)] #! T: max prediction time bins time_pred//bucket
        ff_events, event_counts, disparity, src_ofst_res = data

        ff_events = ff_events.cuda()
        event_counts = event_counts.cuda()
        disparity = disparity.cuda().float() # (B,H,W)

        if not args.polarity[0]: ff_events = ff_events[..., :3]

        # 720x1280 ff with features in the center -> 480x480 crop from the 480x640 center. The model resizes it to 518x518.
        src_0ofst_res = src_ofst_res.clone() # (xofst, yofst, h, w)
        src_0ofst_res[:, :2] = 0
        cparams = crop_params(src_0ofst_res, src_ofst_res[0, 2]) + src_ofst_res[:, :2].repeat(1, 2)

        disparity = torch.stack([
            disparity[i, cparams[i, 0]:cparams[i, 2], cparams[i, 1]:cparams[i, 3]]
            for i in range(disparity.shape[0])
        ])

        with torch.autocast(device_type="cuda", enabled=args.amp, dtype=torch.float16):
            disparity_pred = model(ff_events, event_counts, cparams)[0] # (B,N,3) -> (B,C,H,W)
            disparity_valid_mask = disparity < args.max_disparity # don't want asburdly high disparities
            if args.loss == "siloggrad":
                with torch.no_grad():
                    grad = pseudo_model(ff_events, event_counts, cparams)[0] # (B,N,3) -> (B,H,W)
                loss = loss_fn(disparity_pred, disparity, grad, disparity_valid_mask)
            else:
                loss = loss_fn(disparity_pred, disparity, disparity_valid_mask)
            loss = loss / iters_to_accumulate
        scaler.scale(loss).backward()

        train_loss += loss.item()
        iter_loss += loss.item()

        if (idx+1) % iters_to_accumulate == 0:
            logger.info(f"Epoch: {epoch}, Idx: {(idx+1)//iters_to_accumulate}, Loss: {iter_loss}")

            if args.wandb:
                wandb.log({"loss": iter_loss, "lr": scheduler.get_last_lr()[0]})

            if args.clip_grad > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            iter_loss = 0
    # normalize separately because the effective batch size is still based on train_batch
    train_loss /= (len(train_loader) // iters_to_accumulate)

    logger.info("#"*50)
    logger.info(f"Training: Epoch: {epoch}, Loss: {train_loss}")
    logger.info("#"*50)

    if args.wandb:
        wandb.log({"train_loss": train_loss, "epoch": epoch})
