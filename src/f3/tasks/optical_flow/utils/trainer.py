import torch
import wandb
from tqdm import tqdm


def train_fixed_time_optical_flow(args, model, train_loader, optimizer, scheduler,
                                  epoch, logger=None, scaler=None, iters_to_accumulate=1):
    model.train()
    train_loss, train_phto_loss, train_smooth_loss = 0, 0, 0
    iter_loss, iter_phto_loss, iter_smooth_loss = 0, 0, 0
    for idx, data in tqdm(enumerate(train_loader), total=len(train_loader)):
        # [(B,N,3) or (B,N,4)], [(B,W,H,2) or (B,W,H,T,2)], [(B,H,W,1)] #! T: max prediction time bins time_pred//bucket
        events_old, events_new, events_flow, counts_old, counts_new, counts_flow, src_ofst_res = data

        events_old, events_new, events_flow = events_old.cuda(), events_new.cuda(), events_flow.cuda()
        counts_old, counts_new, counts_flow = counts_old.cuda(), counts_new.cuda(), counts_flow.cuda()

        cparams = torch.cat([
            src_ofst_res[:, :2],
            src_ofst_res[:, :2] + src_ofst_res[:, 2:]
        ], dim=1).int()

        with torch.autocast(device_type="cuda", enabled=args.amp, dtype=torch.float16):
            _, _, _, _, loss_dict = model(events_flow, counts_flow,
                                          events_old, counts_old,
                                          events_new, counts_new, cparams)
            loss = loss_dict["loss"]
            loss /= iters_to_accumulate
        scaler.scale(loss).backward()

        photometric_loss = loss_dict["photometric_loss"]
        smoothness_loss = loss_dict["smoothness_loss"]

        train_loss += loss.item()
        iter_loss += loss.item()
        train_phto_loss += photometric_loss.item()
        iter_phto_loss += photometric_loss.item()
        train_smooth_loss += smoothness_loss.item()
        iter_smooth_loss += smoothness_loss.item()

        if (idx+1) % iters_to_accumulate == 0:
            logger.info(f"Epoch: {epoch}, Idx: {(idx+1)//iters_to_accumulate}, Loss: {iter_loss}, " +\
                        f"Photometric Loss: {iter_phto_loss}, Smoothness Loss: {iter_smooth_loss}")

            if args.wandb:
                wandb.log({"loss": iter_loss, "photometric_loss": iter_phto_loss,
                           "smoothness_loss": iter_smooth_loss, "lr": scheduler.get_last_lr()[0]})
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            iter_loss, iter_phto_loss, iter_smooth_loss = 0, 0, 0
    # normalize separately because the effective batch size is still based on train_batch
    train_loss /= (len(train_loader) // iters_to_accumulate)
    train_phto_loss /= (len(train_loader) // iters_to_accumulate)
    train_smooth_loss /= (len(train_loader) // iters_to_accumulate)

    logger.info("#"*50)
    logger.info(f"Training: Epoch: {epoch}, Loss: {train_loss}, Photometric Loss: {train_phto_loss}, " +\
                f"Smoothness Loss: {train_smooth_loss}")
    logger.info("#"*50)

    if args.wandb:
        wandb.log({"train_loss": train_loss, "train_photometric_loss": train_phto_loss,
                   "train_smoothness_loss": train_smooth_loss, "epoch": epoch})
