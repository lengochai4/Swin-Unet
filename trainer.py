import logging
import os
import random
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from utils import DiceLoss


def trainer_synapse(args, model, snapshot_path):
    from datasets.dataset_synapse import Synapse_dataset, RandomGenerator

    # -----------------------
    # Prepare output dir + logging
    # -----------------------
    os.makedirs(snapshot_path, exist_ok=True)

    logging.basicConfig(
        filename=os.path.join(snapshot_path, "log.txt"),
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d] %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu

    # -----------------------
    # Datasets
    # -----------------------
    db_train = Synapse_dataset(
        base_dir=args.root_path,
        list_dir=args.list_dir,
        split="train",
        transform=transforms.Compose(
            [RandomGenerator(output_size=[args.img_size, args.img_size])]
        ),
    )
    db_val = Synapse_dataset(
        base_dir=args.root_path,
        list_dir=args.list_dir,
        split="val",
        transform=transforms.Compose(
            [RandomGenerator(output_size=[args.img_size, args.img_size])]
        ),
    )

    logging.info("The length of train set is: {}".format(len(db_train)))
    logging.info("The length of val set is: {}".format(len(db_val)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    # -----------------------
    # DataLoaders (FIXED)
    # -----------------------
    train_loader = DataLoader(
        db_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )

    # NOTE: val should use db_val (not db_train)
    val_loader = DataLoader(
        db_val,
        batch_size=1,  # val thường batch 1 để ổn định
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )

    if args.n_gpu > 1:
        model = nn.DataParallel(model)

    model.train()
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)

    optimizer = optim.SGD(
        model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001
    )
    writer = SummaryWriter(os.path.join(snapshot_path, "log"))

    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(train_loader)
    logging.info(
        "{} iterations per epoch. {} max iterations ".format(
            len(train_loader), max_iterations
        )
    )

    best_loss = 1e10
    epoch_bar = tqdm(range(max_epoch), ncols=70)

    for epoch_num in epoch_bar:
        # -----------------------
        # Train
        # -----------------------
        model.train()
        sum_ce = 0.0
        sum_dice = 0.0

        for _, sampled_batch in tqdm(
            enumerate(train_loader),
            desc=f"Train: {epoch_num}",
            total=len(train_loader),
            leave=False,
        ):
            image_batch, label_batch = sampled_batch["image"], sampled_batch["label"]
            image_batch = image_batch.cuda(non_blocking=True)
            label_batch = label_batch.cuda(non_blocking=True)

            outputs = model(image_batch)

            loss_ce = ce_loss(outputs, label_batch.long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            loss = 0.4 * loss_ce + 0.6 * loss_dice

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            # poly LR schedule
            lr_ = base_lr * (1.0 - float(iter_num) / float(max_iterations)) ** 0.9
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr_

            iter_num += 1
            writer.add_scalar("info/lr", lr_, iter_num)
            writer.add_scalar("info/total_loss", loss.item(), iter_num)
            writer.add_scalar("info/loss_ce", loss_ce.item(), iter_num)
            writer.add_scalar("info/loss_dice", loss_dice.item(), iter_num)

            sum_ce += loss_ce.item()
            sum_dice += loss_dice.item()

            # -----------------------
            # Tensorboard images (FIXED SHAPES)
            # add_image expects CHW (3D). We'll log (1,H,W).
            # -----------------------
            if iter_num % 20 == 0:
                b = 1 if image_batch.shape[0] > 1 else 0

                img = image_batch[b, 0:1, :, :]  # (1,H,W)
                img = (img - img.min()) / (img.max() - img.min() + 1e-8)
                writer.add_image("train/Image", img, iter_num)

                pred = torch.argmax(torch.softmax(outputs, dim=1), dim=1)  # (B,H,W)
                writer.add_image("train/Prediction", pred[b, ...].unsqueeze(0) * 50, iter_num)

                gt = label_batch[b, ...].unsqueeze(0)  # (1,H,W)
                writer.add_image("train/GroundTruth", gt * 50, iter_num)

        mean_ce = sum_ce / max(1, len(train_loader))
        mean_dice = sum_dice / max(1, len(train_loader))
        train_loss = 0.4 * mean_ce + 0.6 * mean_dice
        logging.info(
            "Train epoch: %d : loss : %f, loss_ce: %f, loss_dice: %f"
            % (epoch_num, train_loss, mean_ce, mean_dice)
        )

        # -----------------------
        # Val
        # -----------------------
        if (epoch_num + 1) % args.eval_interval == 0:
            model.eval()
            v_sum_ce = 0.0
            v_sum_dice = 0.0

            with torch.no_grad():
                for _, sampled_batch in tqdm(
                    enumerate(val_loader),
                    desc=f"Val: {epoch_num}",
                    total=len(val_loader),
                    leave=False,
                ):
                    image_batch, label_batch = sampled_batch["image"], sampled_batch["label"]
                    image_batch = image_batch.cuda(non_blocking=True)
                    label_batch = label_batch.cuda(non_blocking=True)

                    outputs = model(image_batch)
                    loss_ce = ce_loss(outputs, label_batch.long())
                    loss_dice = dice_loss(outputs, label_batch, softmax=True)

                    v_sum_ce += loss_ce.item()
                    v_sum_dice += loss_dice.item()

            v_mean_ce = v_sum_ce / max(1, len(val_loader))
            v_mean_dice = v_sum_dice / max(1, len(val_loader))
            val_loss = 0.4 * v_mean_ce + 0.6 * v_mean_dice

            logging.info(
                "Val epoch: %d : loss : %f, loss_ce: %f, loss_dice: %f"
                % (epoch_num, val_loss, v_mean_ce, v_mean_dice)
            )

            # Save best / last
            if val_loss < best_loss:
                best_loss = val_loss
                save_mode_path = os.path.join(snapshot_path, "best_model.pth")
            else:
                save_mode_path = os.path.join(snapshot_path, "last_model.pth")

            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

    writer.close()
    return "Training Finished!"
