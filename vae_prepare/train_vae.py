import os
os.environ["CUDA_VISIBLE_DEVICES"] = '4,5,6,7'
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import yaml
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from accelerate import Accelerator
from tqdm import tqdm
from diffusers.optimization import get_scheduler
import lpips
from dataloader import AMCDataset
from diffusers import AutoencoderKLTemporalDecoder
from utils import count_num_params, save_orig_and_generated_gifs
from modules import PatchGAN, init_weights
from modules import LPIPS as mylpips
from einops import rearrange

# OMP_NUM_THREADS=4  python -m torch.distributed.run --nproc_per_node=1 --master_port=33371 train_vae.py
### Load Arguments ###
def experiment_config_parser():
    parser = argparse.ArgumentParser(description="Experiment Configuration")

    parser.add_argument("--experiment_name",
                        help="Name of Experiment being Launched",
                        # required=True,
                        type=str,
                        default="VAETrainer",
                        metavar="experiment_name")

    parser.add_argument("--wandb_run_name",
                        # required=True,
                        type=str,
                        default="vae_0903",
                        metavar="wandb_run_name")

    parser.add_argument("--working_directory",
                        help="Working Directory where checkpoints and logs are stored, inside a \
                        folder labeled by the experiment name",
                        # required=True,
                        type=str,
                        default="/ssd2/AMC_zstack_2_patches/vae_0903",
                        metavar="working_directory")

    parser.add_argument("--log_wandb",
                        default=True,
                        action=argparse.BooleanOptionalAction,
                        help="Do you want to log to WandB?")

    parser.add_argument("--resume_from_checkpoint",
                        help="Pass name of checkpoint folder to resume training from",
                        default=None,
                        type=str,
                        metavar="resume_from_checkpoint")

    parser.add_argument("--training_config",
                        help="Path to config file for all training information",
                        # required=True,
                        default="./configs/stage1_vae_train.yaml",
                        type=str,
                        metavar="training_config")

    parser.add_argument("--pretrained_model_name_or_path",
                        help="Path to config file for all model information",
                        # required=True,
                        default="stabilityai/stable-video-diffusion-img2vid-xt",
                        type=str,
                        metavar="model_config")

    parser.add_argument("--path_to_dataset",
                        help="Root directory of dataset",
                        default='/ssd1/AMC_zstack_2_patches_warp/pngs_mid/',
                        # required=True,
                        type=str)

    parser.add_argument("--path_to_save_gens",
                        help="Folder you want to store the testing generations througout training",
                        default="/ssd2/AMC_zstack_2_patches/vae_0903/gen",
                        type=str)

    parser.add_argument("--image_size",
                        default=256,
                        type=int)

    parser.add_argument("--num_frames",
                        default=11,
                        type=int)

    args = parser.parse_args()

    return args

def main():
    args = experiment_config_parser()

    ### Load Configs (training config and vae config) ###
    with open(args.training_config, "r") as f:
        training_config = yaml.safe_load(f)["training_args"]

    # with open(args.model_config, "r") as f:
    #     vae_config = yaml.safe_load(f)["vae"]
    #     config = LDMConfig(**vae_config)

    # assert not config.quantize, "This script only supports VAE, use stage1_vqvae_trainer.py for Quantized"

    ### Initialize Accelerator/Tracker ###
    path_to_experiment = os.path.join(args.working_directory, args.experiment_name)
    accelerator = Accelerator(project_dir=path_to_experiment,
                              gradient_accumulation_steps=training_config["gradient_accumulations_steps"],
                              log_with="wandb" if args.log_wandb else None)

    if args.log_wandb:
        accelerator.init_trackers(args.experiment_name, init_kwargs={"wandb": {"name": args.wandb_run_name}})

    ### Load Model ###
    model = AutoencoderKLTemporalDecoder.from_pretrained(
            args.pretrained_model_name_or_path,
        subfolder="vae", revision=None, variant="fp16")
    model.requires_grad_(True)
    for param in model.encoder.parameters():
        param.requires_grad = False
    for param in model.decoder.parameters():
        assert param.requires_grad == True, "Decoder parameter is not trainable"

    trainable_params = [p for p in model.parameters() if p.requires_grad]

    # latent_res = (config.img_size // (len(config.vae_channels_per_block) - 1) ** 2)
    # accelerator.print(f"LATENT SPACE DIMENSIONS: {config.latent_channels, latent_res, latent_res}")

    ### Load LPIPS ###
    use_lpips = False
    if training_config["use_lpips"]:
        use_lpips = True
        if training_config["use_lpips_package"]:
            lpips_loss_fn = lpips.LPIPS(net="vgg").eval()
        else:
            lpips_loss_fn = mylpips()
            lpips_loss_fn.load_checkpoint(training_config["lpips_checkpoint"])

        lpips_loss_fn = lpips_loss_fn.to(accelerator.device)

    ### Load Discriminator ###
    use_disc = False
    if training_config["use_patchgan"]:
        use_disc = True
        discriminator = PatchGAN(input_channels=3,
                                 start_dim=training_config["disc_start_dim"],
                                 depth=training_config["disc_depth"],
                                 # kernel_size=training_config["disc_kernel_size"],
                                 leaky_relu_slope=training_config["disc_leaky_relu"]).apply(init_weights)

        discriminator = discriminator.to(accelerator.device)

        ### If we are training on multiple GPUs, we need to convert BatchNorm to SyncBatchNorm ###
        if accelerator.num_processes > 1:
            discriminator = nn.SyncBatchNorm.convert_sync_batchnorm(discriminator)

    ### Print Out Number of Trainable Parameters ###
    # Count the trainable parameters
    num_trainable_params = sum(p.numel() for p in trainable_params)
    print(f"Number of trainable parameters: {num_trainable_params}")
    accelerator.print(f"NUMBER OF VAE PARAMETERS: {count_num_params(model)}")
    if use_disc:
        accelerator.print(f"NUMBER OF DISC PARAMETERS: {count_num_params(discriminator)}")

    ### Load Optimizers ###
    optimizer = torch.optim.AdamW(trainable_params,
                                  lr=training_config["learning_rate"],
                                  betas=(training_config["optimizer_beta1"], training_config["optimizer_beta2"]),
                                  weight_decay=training_config["optimizer_weight_decay"])

    if use_disc:
        disc_optimizer = torch.optim.AdamW(discriminator.parameters(),
                                           lr=training_config["disc_learning_rate"],
                                           betas=(training_config["optimizer_beta1"], training_config["optimizer_beta2"]),
                                           weight_decay=training_config["optimizer_weight_decay"])

    bce_loss = nn.BCELoss()
    ### Get DataLoader ###
    mini_batchsize = training_config["per_gpu_batch_size"] // training_config["gradient_accumulations_steps"]
    # dataset = get_dataset(dataset=args.dataset,
    #                       path_to_data=args.path_to_dataset,
    #                       num_channels=vae_config["in_channels"],
    #                       img_size=vae_config["img_size"],
    #                       random_resize=training_config["random_resize"],
    #                       interpolation=training_config["interpolation"],
    #                       return_caption=False)

    train_dataset = AMCDataset(data_dir=args.path_to_dataset,
                               split="train",
                               )

    accelerator.print("Number of Training Samples:", len(train_dataset))

    dataloader = DataLoader(train_dataset,
                            batch_size=mini_batchsize,
                            pin_memory=training_config["pin_memory"],
                            num_workers=training_config["num_workers"],
                            shuffle=True)

    effective_epochs = (training_config["per_gpu_batch_size"] * \
                        accelerator.num_processes * \
                        training_config["total_training_iterations"]) / len(train_dataset)

    accelerator.print("Effective Epochs:", round(effective_epochs, 2))

    ### Get Learning Rate Scheduler ###
    lr_scheduler = get_scheduler(
        training_config["lr_scheduler"],
        optimizer=optimizer,
        num_training_steps=training_config["total_training_iterations"],
        num_warmup_steps=training_config["lr_warmup_steps"]
    )

    if use_disc:
        disc_lr_scheduler = get_scheduler(
            training_config["disc_lr_scheduler"],
            optimizer=disc_optimizer,
            num_training_steps=training_config["total_training_iterations"],
            num_warmup_steps=training_config["disc_lr_warmup_steps"],
        )

    ### Prepare Everything ###
    model, optimizer, lr_scheduler, dataloader = accelerator.prepare(
        model, optimizer, lr_scheduler, dataloader)

    if use_disc:
        discriminator, disc_optimizer, disc_lr_scheduler = accelerator.prepare(
            discriminator, disc_optimizer, disc_lr_scheduler
        )

    if use_lpips:
        lpips_loss_fn = accelerator.prepare(lpips_loss_fn)

    ### Initialize Variables to Accumulate ###
    model_log = {"loss": 0,
                 "perceptual_loss": 0,
                 "reconstruction_loss": 0,
                 "lpips_loss": 0,
                 # "kl_loss": 0,
                 "generator_loss": 0,
                 "adp_weight": 0}

    disc_log = {"disc_loss": 0,
                "logits_real": 0,
                "logits_fake": 0}

    ### Quick Helper to Rest Logs ###
    def reset_log(log):
        return {key: 0 for (key, _) in log.items()}

    ### Resume From Checkpoint ###
    if args.resume_from_checkpoint is not None:
        accelerator.print(f"Resuming from Checkpoint: {args.resume_from_checkpoint}")
        path_to_checkpoint = os.path.join(path_to_experiment, args.resume_from_checkpoint)
        accelerator.load_state(path_to_checkpoint)
        global_step = int(args.resume_from_checkpoint.split("_")[-1])
    else:
        global_step = 0

    progress_bar = tqdm(range(training_config["total_training_iterations"]),
                        initial=global_step,
                        disable=not accelerator.is_local_main_process)

    model.train()

    if use_disc:
        discriminator.train()

    for i, batch in enumerate(dataloader):
        # print(i)
        pixel_values = batch["pixel_values"].to(accelerator.device)
        pixel_values = rearrange(pixel_values, "b f c h w -> (b f) c h w")
        model_toggle = (global_step % 2) == 0
        train_disc = (global_step >= training_config["disc_start"])

        ### If we are not using discriminator, then always generator step, and train_disc is false ###
        if not use_disc:
            generator_step = True
            train_disc = False
        else:
            if model_toggle or not train_disc:
                generator_step = True
            else:
                generator_step = False

        if generator_step:
            discriminator.eval()
            model.train()
            optimizer.zero_grad()

            with torch.no_grad():
                posterior = model.module.encode(pixel_values).latent_dist
                z = posterior.mode()
            reconstructions = model.module.decode(z, num_frames=args.num_frames).sample
            # loss_msk = torch.stack(batch["mask"], dim=0)
            # msk = loss_msk.squeeze()
            # # loss_msk = loss_msk.permute(1, 0)
            # loss_msk = loss_msk[:, :, None, None]
            # loss_msk_exp = loss_msk.expand(-1, reconstructions.size(1), reconstructions.size(2),
            #                                reconstructions.size(3))

            # with torch.no_grad():
            with accelerator.accumulate(model):
                ### Reconstruction Loss ###
                if training_config["reconstruction_loss_fn"] == "l1":
                    reconstruction_loss = F.l1_loss(pixel_values, reconstructions, reduction='none').mean()
                elif training_config["reconstruction_loss_fn"] == "l2":
                    reconstruction_loss = F.mse_loss(pixel_values, reconstructions, reduction='none').mean()
                else:
                    raise ValueError(f"{training_config['reconstruction_loss_fn']} is not a Valid Reconstruction Loss")

                # reconstruction_loss = torch.sum(reconstruction_loss[loss_msk_exp]) / torch.sum(
                #     torch.ones_like(loss_msk_exp))
                ### Perceptual Loss ###
                lpips_loss = torch.zeros(size=(), device=pixel_values.device)
                # if use_lpips:
                #     lpips_loss = lpips_loss_fn(reconstructions, pixel_values)[loss_msk].mean()
                if use_lpips:
                    lpips_loss = lpips_loss_fn(reconstructions, pixel_values).mean()

                ### Add Together Losses ###
                perceptual_loss = reconstruction_loss + training_config["lpips_weight"] * lpips_loss
                loss = perceptual_loss

                ### Compute Discriminator Loss (incase we are training the discriminator) ###
                gen_loss = torch.zeros(size=(), device=pixel_values.device)
                adaptive_weight = torch.zeros(size=(), device=pixel_values.device)
                if train_disc:
                    # print(reconstructions.shape)
                    # print(rearrange(reconstructions, "(b f) c h w -> b c f h w", f=args.num_frames).shape)
                    gen_loss = -1 * discriminator(
                        rearrange(reconstructions, "(b f) c h w -> b c f h w", f=args.num_frames)
                    ).mean()
                    # fake = discriminator(rearrange(reconstructions, "(b f) c h w -> b c f h w", b=1))
                    # gen_loss = bce_loss(nn.functional.sigmoid(fake), torch.ones_like(fake))


                    last_layer = accelerator.unwrap_model(model).decoder.conv_out.weight
                    norm_grad_wrt_perceptual_loss = torch.autograd.grad(outputs=loss,
                                                                        inputs=last_layer,
                                                                        retain_graph=True)[0].detach().norm(p=2)
                    norm_grad_wrt_gen_loss = torch.autograd.grad(outputs=gen_loss,
                                                                 inputs=last_layer,
                                                                 retain_graph=True)[0].detach().norm(p=2)

                    adaptive_weight = norm_grad_wrt_perceptual_loss / norm_grad_wrt_gen_loss.clamp(min=1e-8)
                    adaptive_weight = adaptive_weight.clamp(max=1e4)

                    loss = loss + adaptive_weight * gen_loss * training_config["disc_weight"]

                ### Compute KL Loss ###
                # kl_loss = model_outputs["kl_loss"].mean()
                # loss = loss + kl_loss * training_config["kl_weight"]
                # kl_loss = posterior.kl()[msk].mean()
                # kl_loss = posterior.kl().mean()
                # loss = loss + kl_loss * training_config["kl_weight"]

                ### Update Model ###
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                lr_scheduler.step()

                ### Create Log of Everything ###
                log = {"loss": loss,
                       "perceptual_loss": perceptual_loss,
                       "reconstruction_loss": reconstruction_loss,
                       "lpips_loss": lpips_loss,
                       # "kl_loss": kl_loss,
                       "generator_loss": gen_loss,
                       "adp_weight": adaptive_weight}

                ### Accumulate Log ###
                for key, value in log.items():
                    model_log[key] += value.mean() / training_config["gradient_accumulations_steps"]
        else:
            discriminator.train()
            model.eval()
            disc_optimizer.zero_grad()

            with torch.no_grad():
                posterior = model.module.encode(pixel_values).latent_dist
                z = posterior.mode()
            reconstructions = model.module.decode(z, num_frames=args.num_frames).sample

            # loss_msk = torch.stack(batch["mask"], dim=0)
            # msk = loss_msk.squeeze()

            with accelerator.accumulate(discriminator):  # ?
                ### Hinge Loss ###
                # real = discriminator(pixel_values[msk])
                # fake = discriminator(reconstructions[msk])
                # TODO: It will not work when batch size is not 1
                # real = discriminator(rearrange(pixel_values[msk], "(b f) c h w -> b c f h w", b=1))
                real = discriminator(rearrange(pixel_values, "(b f) c h w -> b c f h w", b=1))
                fake = discriminator(rearrange(reconstructions, "(b f) c h w -> b c f h w", b=1))
                loss = (F.relu(1 + fake) + F.relu(1 - real)).mean()
                # loss_real = bce_loss(nn.functional.sigmoid(real), torch.ones_like(real))
                # loss_fake = bce_loss(nn.functional.sigmoid(fake), torch.zeros_like(fake))
                # loss = loss_real + loss_fake

                ### Update Discriminator Model ###
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(discriminator.parameters(), 1.0)

                disc_optimizer.step()
                disc_lr_scheduler.step()

                log = {"disc_loss": loss,
                       "logits_real": real.mean(),
                       "logits_fake": fake.mean()}

                ### Accumulate Log ###
                for key, value in log.items():
                    disc_log[key] += value.mean() / training_config["gradient_accumulations_steps"]

        if accelerator.sync_gradients:
            ### If we updated the VAE ###
            if model_toggle or not train_disc:

                ## Gather Across GPUs ###
                # model_log = {key: accelerator.gather_for_metrics(value).mean().item() for key, value in model_log.items()}
                # model_log = {key: accelerator.gather_for_metrics(value) for key, value in model_log.items()}
                model_log = {
                    key: (
                        value if isinstance(value, (int, float))
                        else accelerator.gather_for_metrics(value).mean().item()
                    )
                    for key, value in model_log.items()
                }
                model_log["lr"] = lr_scheduler.get_last_lr()[0]

                logging_string = "GEN: "
                for k, v in model_log.items():
                    v = v.item() if torch.is_tensor(v) else v
                    if "lr" in k:
                        v = f"{v:.1e}"
                    else:
                        v = round(v, 2)
                    logging_string += f"|{k}: {v}"

                ### Print to Console ###
                accelerator.print(logging_string)

                ### Push to WandB ###
                accelerator.log(model_log, step=global_step)

                ### Reset Log for Next Accumulation ###
                model_log = reset_log(model_log)
                model_log.pop("lr")

            ### If we updated the Discriminator ###
            else:
                ## Gather Across GPUs ###
                # disc_log = {key: accelerator.gather_for_metrics(value).mean().item() for key, value in disc_log.items()}
                # disc_log = {key: accelerator.gather_for_metrics(value) for key, value in disc_log.items()}
                disc_log = {
                    key: (
                        value if isinstance(value, (int, float))
                        else accelerator.gather_for_metrics(value).mean().item()
                    )
                    for key, value in disc_log.items()
                }
                disc_log["disc_lr"] = disc_lr_scheduler.get_last_lr()[0]

                logging_string = "DIS: "
                for k, v in disc_log.items():
                    v = v.item() if torch.is_tensor(v) else v
                    if "lr" in k:
                        v = f"{v:.1e}"
                    else:
                        v = round(v, 2)
                    logging_string += f"|{k}: {v}"

                ### Print to Console ###
                accelerator.print(logging_string)

                ### Push to WandB ###
                accelerator.log(disc_log, step=global_step)

                ### Reset Log for Next Accumulation ###
                disc_log = reset_log(disc_log)
                disc_log.pop("disc_lr")

            global_step += 1
            progress_bar.update(1)

        if global_step % training_config["val_generation_freq"] == 0:
            if accelerator.is_main_process:
                images_to_plot = pixel_values.detach()
                save_orig_and_generated_gifs(original_images=images_to_plot,
                                             generated_image_tensors=reconstructions.detach(),
                                             path_to_save_folder=args.path_to_save_gens,
                                             step=global_step,
                                             accelerator=accelerator)

                model.train()
            accelerator.wait_for_everyone()

        if (global_step % training_config["checkpoint_iterations"] == 0) or (
                global_step == training_config["total_training_iterations"] - 1):
            path_to_checkpoint = os.path.join(path_to_experiment, f"checkpoint_{global_step}")
            accelerator.save_state(output_dir=path_to_checkpoint)

        if global_step >= training_config["total_training_iterations"]:
            path_to_checkpoint = os.path.join(path_to_experiment, f"checkpoint_{global_step}_last")
            accelerator.save_state(output_dir=path_to_checkpoint)
            print("Completed Training")
            break


if __name__ == "__main__":
    main()
