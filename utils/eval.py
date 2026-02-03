import torch
import os
import cv2

import numpy as np

from PIL import Image
from tqdm import tqdm
from typing import Optional

from torchvision import transforms as vt
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from utils.util import get_prompt_template
from utils.viz import draw_overall, draw_overlaid

import datasets.vggsound.eval_utils as vggsound_eval
import datasets.VGGSS.eval_utils as vggss_eval
import datasets.VGGSS.extend_eval_utils as exvggss_eval
import datasets.Flickr.eval_utils as flickr_eval
import datasets.Flickr.extend_eval_utils as exflickr_eval
import datasets.AVSBench.eval_utils as avsbench_eval
from datasets.silence_and_noise.silence_and_noise import get_silence_noise_audios

from typing import List, Optional, Tuple, Dict
from importlib import import_module

import wandb
import sys

@torch.no_grad()
def eval_vggsound_validation(
    model: torch.nn.Module,
    val_dataloader: DataLoader,
    args,
    result_dir: str,
    epoch: Optional[int] = None,
    tensorboard_path: Optional[str] = None,
    rank = 0,
    wandb_run: Optional[wandb.Run] = None
):
    '''
    Evaluate provided model on VGG-Sound validation dataset.

    Args:
        model (torch.nn.Module): Sound localization model to evaluate.
        val_dataloader (DataLoader): DataLoader for the test dataset.
        result_dir (str): Directory to save the evaluation results.
        epoch (int, optional): The current epoch number (default: None).
        tensorboard_path (str, optional): Path to store TensorBoard logs. If None, TensorBoard logs won't be written.

    Returns:
        loss and other things
    '''

    loss_dict = {}
    total_loss_per_epopch = 0.0
    loss_add_count = 0.0

    if tensorboard_path is not None and epoch is not None:
        os.makedirs(tensorboard_path, exist_ok=True)
        writer = SummaryWriter(tensorboard_path)

    # test_split = val_dataloader.dataset.split

    loss_per_epoch_dict = {loss_name: 0.0 for loss_name in args.loss}

    # Get placeholder text
    prompt_template, text_pos_at_prompt, prompt_length = get_prompt_template()

    pbar = tqdm(val_dataloader, desc=f"Validation Epoch [{epoch}/{args.epoch}]", disable=(rank != 0))

    san_dict_base = {'san': False, 'san_real': False, 'neg_audios': None}

    for step, data in enumerate(pbar):
        images, audios, name = data['images'], data['audios'], data['ids']
        noisy_audios = data['noisy_audios']

        # Inference
        placeholder_tokens = model.get_placeholder_token(prompt_template.replace('{}', ''))
        placeholder_tokens = placeholder_tokens.repeat((val_dataloader.batch_size, 1))
        audio_driven_embedding = model.encode_audio(audios.to(model.device), placeholder_tokens, text_pos_at_prompt,
                                                    prompt_length)

        san_dict = san_dict_base
        if 'diff_san_l' in args.loss:
            audio_driven_embedding_noisy = model.encode_audio(noisy_audios.to(model.device), placeholder_tokens,
                                                        text_pos_at_prompt, prompt_length)
            out_dict_noisy = model.forward_for_validation(images.to(model.device), audio_driven_embedding_noisy, 352)
            out_dict_noisy = {f'noisy_{key}': value for key, value in out_dict_noisy.items()}
            san_dict = {'pred_emb_noisy': audio_driven_embedding_noisy, **out_dict_noisy, **san_dict}

        # Localization result
        out_dict = model.forward_for_validation(images.to(model.device), audio_driven_embedding, 224)

        loss_args = {'pred_emb': audio_driven_embedding, **out_dict, **san_dict}

        for j, loss_name in enumerate(args.loss):
            loss_dict[loss_name] = getattr(import_module('utils.loss'), loss_name)(**loss_args) * args.loss_w[j]
            loss_per_epoch_dict[loss_name] += loss_dict[loss_name].item()

        loss = torch.sum(torch.stack(list(loss_dict.values())))

        # Visual results
        for j in range(val_dataloader.batch_size):
            seg = out_dict['heatmap'][j:j+1]
            seg_image = ((1 - seg.squeeze().detach().cpu().numpy()) * 255).astype(np.uint8)

            os.makedirs(f'{result_dir}/heatmap', exist_ok=True)
            cv2.imwrite(f'{result_dir}/heatmap/{name[j]}.jpg', seg_image)

            if step < 2 and wandb_run and rank == 0:
                heatmap_image = cv2.applyColorMap(((seg.squeeze().detach().cpu().numpy()) * 255).astype(np.uint8), cv2.COLORMAP_JET)
                original_image = Image.open(os.path.join(val_dataloader.dataset.image_path, name[j] + '.jpg')).resize((224, 224))
                overlaid_image = cv2.addWeighted(np.array(original_image), 0.5, heatmap_image, 0.5, 0)

                wandb_run.log({f'images/val_overlaid/{name[j]}.jpg': wandb.Image(overlaid_image)})

        total_loss_per_epopch += loss.item()
        loss_add_count += 1.0

        avr_loss = total_loss_per_epopch / loss_add_count

        if rank == 0:
            pbar.set_description(f"Validation Epoch {epoch}, Loss = {round(avr_loss, 5)}")

            if wandb_run:
                wandb_run.log({f'validation_losses/step/{key}': val for key, val in loss_dict.items()})
                wandb_run.log({f'validation_losses/avr/{key}': val / loss_add_count for key, val in loss_per_epoch_dict.items()})
                wandb_run.log({'validation/step/loss' : loss.item()})
                wandb_run.log({'validation/avr/loss' : avr_loss})

    # Save result
    os.makedirs(result_dir, exist_ok=True)
    rst_path = os.path.join(f'{result_dir}/', 'test_rst.txt')
    msg = ''

    print(msg)
    with open(rst_path, 'w') as fp_rst:
        fp_rst.write(msg)

    if tensorboard_path is not None and epoch is not None:
        writer.close()

    return float(total_loss_per_epopch / loss_add_count)


@torch.no_grad()
def eval_vggsound_agg(
    model: torch.nn.Module,
    test_dataloader: DataLoader,
    result_dir: str,
    epoch: Optional[int] = None,
    tensorboard_path: Optional[str] = None
) -> Dict[str, float]:
    '''
    Evaluate provided model on VGGS (VGG-Sound) test dataset.

    Args:
        model (torch.nn.Module): Sound localization model to evaluate.
        test_dataloader (DataLoader): DataLoader for the test dataset.
        result_dir (str): Directory to save the evaluation results.
        epoch (int, optional): The current epoch number (default: None).
        tensorboard_path (str, optional): Path to store TensorBoard logs. If None, TensorBoard logs won't be written.

    Returns:
        result_dict (Dict): Best AUC value (threshold optimized)

    Notes:
        The evaluation includes threshold optimization for VGG-SS.
    '''

    if tensorboard_path is not None and epoch is not None:
        os.makedirs(tensorboard_path, exist_ok=True)
        writer = SummaryWriter(tensorboard_path)

    test_split = test_dataloader.dataset.split

    # Get placeholder text
    prompt_template, text_pos_at_prompt, prompt_length = get_prompt_template()

    # Thresholds for evaluation
    thrs = [0.05, 0.1, 0.15, 0.2, 0.25, 0.30, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.70, 0.75, 0.8, 0.85, 0.9, 0.95]
    evaluators_silence = [vggsound_eval.Evaluator() for i in range(len(thrs))]
    evaluators_noise = [vggsound_eval.Evaluator() for i in range(len(thrs))]

    negative_audios_emb = get_silence_noise_audios(model, test_dataloader.dataset[0]['audios'].shape, san_active=True)
    sil_emb, noise_emb = negative_audios_emb[0, :], negative_audios_emb[1, :]

    for step, data in enumerate(tqdm(test_dataloader, desc=f"Evaluate VGGS({test_split}) dataset...")):
        images, audios = data['images'], data['audios']
        labels, name = data['labels'], data['ids']

        # Localization result
        out_dict = model(images.to(model.device), audios, 224)

        out_dict_silence = model(images.to(model.device), sil_emb, 224)

        out_dict_noise = model(images.to(model.device), noise_emb, 224)

        # Evaluation for all thresholds
        for i, thr in enumerate(thrs):
            evaluators_silence[i].evaluate_batch(out_dict_silence['heatmap'], thr)
            evaluators_noise[i].evaluate_batch(out_dict_noise['heatmap'], thr)

        # Visual results
        for j in range(test_dataloader.batch_size):
            seg = out_dict['heatmap'][j:j+1]
            seg_image = ((1 - seg.squeeze().detach().cpu().numpy()) * 255).astype(np.uint8)

            os.makedirs(f'{result_dir}/heatmap', exist_ok=True)
            cv2.imwrite(f'{result_dir}/heatmap/{name[j]}.jpg', seg_image)

        # Overall figure
        for j in range(test_dataloader.batch_size):
            original_image = Image.open(os.path.join(test_dataloader.dataset.image_path, name[j] + '.jpg')).resize(
                (224, 224))

            seg = out_dict['heatmap'][j:j+1]
            seg_image = ((1 - seg.squeeze().detach().cpu().numpy()) * 255).astype(np.uint8)
            heatmap_image = Image.fromarray(seg_image)

            seg = out_dict_silence['heatmap'][j:j+1]
            seg_image = ((1 - seg.squeeze().detach().cpu().numpy()) * 255).astype(np.uint8)
            heatmap_image_silence = Image.fromarray(seg_image)

            seg = out_dict_noise['heatmap'][j:j+1]
            seg_image = ((1 - seg.squeeze().detach().cpu().numpy()) * 255).astype(np.uint8)
            heatmap_image_noise = Image.fromarray(seg_image)

            draw_overall(result_dir, original_image, heatmap_image, heatmap_image_silence, heatmap_image_noise, labels[j], name[j])
            draw_overlaid(result_dir, original_image, heatmap_image, name[j])

    # Save result
    os.makedirs(result_dir, exist_ok=True)
    rst_path = os.path.join(f'{result_dir}/', 'test_rst.txt')
    msg = ''

    # Final result
    best_AUC_silence = 0.0
    best_AUC_noise = 0.0

    for i, thr in enumerate(thrs):
        audio_loc_key, audio_loc_dict_silence = evaluators_silence[i].finalize()
        audio_loc_key, audio_loc_dict_noise = evaluators_noise[i].finalize()

        msg += f'{model.__class__.__name__} ({test_split} with thr = {thr} evaluated with Silence)\n'
        msg += 'AP50(cIoU)={}, AUC={}\n'.format(audio_loc_dict_silence['pIA'], audio_loc_dict_silence['AUC_N'])
        msg += f'{model.__class__.__name__} ({test_split} with thr = {thr} evaluated with Noise)\n'
        msg += 'AP50(cIoU)={}, AUC={}\n'.format(audio_loc_dict_noise['pIA'], audio_loc_dict_noise['AUC_N'])

        if tensorboard_path is not None and epoch is not None:
            writer.add_scalars(f'test/{test_split}/silence/({thr})', audio_loc_dict_silence, epoch)
            writer.add_scalars(f'test/{test_split}/noise/({thr})', audio_loc_dict_noise, epoch)

        best_AUC_silence = audio_loc_dict_silence['AUC'] if best_AUC_silence < audio_loc_dict_silence['AUC'] else best_AUC_silence
        best_AUC_noise = audio_loc_dict_noise['AUC'] if best_AUC_noise < audio_loc_dict_noise['AUC'] else best_AUC_noise

    print(msg)
    with open(rst_path, 'w') as fp_rst:
        fp_rst.write(msg)

    if tensorboard_path is not None and epoch is not None:
        writer.close()

    result_dict = {'epoch': epoch, 'best_AUC_silence': best_AUC_silence, 'best_AUC_noise': best_AUC_noise}

    return result_dict


@torch.no_grad()
def eval_vggss_agg(
    model: torch.nn.Module,
    test_dataloader: DataLoader,
    result_dir: str,
    epoch: Optional[int] = None,
    tensorboard_path: Optional[str] = None
) -> Dict[str, float]:
    '''
    Evaluate provided model on VGG-SS (VGG Sound Source) test dataset.

    Args:
        model (torch.nn.Module): Sound localization model to evaluate.
        test_dataloader (DataLoader): DataLoader for the test dataset.
        result_dir (str): Directory to save the evaluation results.
        epoch (int, optional): The current epoch number (default: None).
        tensorboard_path (str, optional): Path to store TensorBoard logs. If None, TensorBoard logs won't be written.

    Returns:
        result_dict (Dict): Best AUC value (threshold optimized)

    Notes:
        The evaluation includes threshold optimization for VGG-SS.
    '''

    if tensorboard_path is not None and epoch is not None:
        os.makedirs(tensorboard_path, exist_ok=True)
        writer = SummaryWriter(tensorboard_path)

    test_split = test_dataloader.dataset.split

    # Get placeholder text
    prompt_template, text_pos_at_prompt, prompt_length = get_prompt_template()

    # Thresholds for evaluation
    thrs = [0.05, 0.1, 0.15, 0.2, 0.25, 0.30, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.70, 0.75, 0.8, 0.85, 0.9, 0.95]
    evaluators = [vggss_eval.Evaluator() for i in range(len(thrs))]

    for step, data in enumerate(tqdm(test_dataloader, desc=f"Evaluate VGG-SS({test_split}) dataset...")):
        images, audios, bboxes = data['images'], data['audios'], data['bboxes']
        labels, name = data['labels'], data['ids']

        # Inference
        placeholder_tokens = model.get_placeholder_token(prompt_template.replace('{}', ''))
        placeholder_tokens = placeholder_tokens.repeat((test_dataloader.batch_size, 1))
        audio_driven_embedding = model.encode_audio(audios.to(model.device), placeholder_tokens, text_pos_at_prompt,
                                                    prompt_length)

        # Localization result
        out_dict = model(images.to(model.device), audio_driven_embedding, 224)

        # Evaluation for all thresholds
        for i, thr in enumerate(thrs):
            evaluators[i].evaluate_batch(out_dict['heatmap'], bboxes, thr)

        # Visual results
        for j in range(test_dataloader.batch_size):
            seg = out_dict['heatmap'][j:j+1]
            seg_image = ((1 - seg.squeeze().detach().cpu().numpy()) * 255).astype(np.uint8)

            os.makedirs(f'{result_dir}/heatmap', exist_ok=True)
            cv2.imwrite(f'{result_dir}/heatmap/{name[j]}.jpg', seg_image)

        # Overall figure
        for j in range(test_dataloader.batch_size):
            original_image = Image.open(os.path.join(test_dataloader.dataset.image_path, name[j] + '.jpg')).resize(
                (224, 224))
            gt_image = vt.ToPILImage()(bboxes[j]).resize((224, 224)).point(lambda p: 255 - p)
            heatmap_image = Image.open(f'{result_dir}/heatmap/{name[j]}.jpg').resize((224, 224))
            seg_image = Image.open(f'{result_dir}/heatmap/{name[j]}.jpg').resize((224, 224)).point(
                lambda p: 0 if p / 255 < 0.5 else 255)

            draw_overall(result_dir, original_image, gt_image, heatmap_image, seg_image, labels[j], name[j])
            draw_overlaid(result_dir, original_image, heatmap_image, name[j])

    # Save result
    os.makedirs(result_dir, exist_ok=True)
    rst_path = os.path.join(f'{result_dir}/', 'test_rst.txt')
    msg = ''

    # Final result
    best_AUC = 0.0

    for i, thr in enumerate(thrs):
        audio_loc_key, audio_loc_dict = evaluators[i].finalize()

        msg += f'{model.__class__.__name__} ({test_split} with thr = {thr})\n'
        msg += 'AP50(cIoU)={}, AUC={}\n'.format(audio_loc_dict['cIoU'], audio_loc_dict['AUC'])

        if tensorboard_path is not None and epoch is not None:
            writer.add_scalars(f'test/{test_split}({thr})', audio_loc_dict, epoch)

        best_AUC = audio_loc_dict['AUC'] if best_AUC < audio_loc_dict['AUC'] else best_AUC

    print(msg)
    with open(rst_path, 'w') as fp_rst:
        fp_rst.write(msg)

    if tensorboard_path is not None and epoch is not None:
        writer.close()

    result_dict = {'epoch': epoch, 'best_AUC': best_AUC}

    return result_dict


@torch.no_grad()
def eval_avsbench_agg(
    model: torch.nn.Module,
    test_dataloader: DataLoader,
    result_dir: str,
    epoch: Optional[int] = None,
    tensorboard_path: Optional[str] = None
) -> None:
    '''
    Evaluate provided  model on AVSBench (S4, MS3) test dataset.

    Args:
        model (torch.nn.Module): Sound localization model to evaluate.
        test_dataloader (DataLoader): DataLoader for the test dataset.
        result_dir (str): Directory to save the evaluation results.
        epoch (int, optional): The current epoch number (default: None).
        tensorboard_path (str, optional): Path to store TensorBoard logs. If None, TensorBoard logs won't be written.

    Returns:
        None

    Notes:
        The evaluation includes threshold optimization for AVSBench.
    '''
    if tensorboard_path is not None and epoch is not None:
        os.makedirs(tensorboard_path, exist_ok=True)
        writer = SummaryWriter(tensorboard_path)

    test_split = test_dataloader.dataset.setting

    # Get placeholder text
    prompt_template, text_pos_at_prompt, prompt_length = get_prompt_template()

    # Thresholds for evaluation
    thrs = [0.05, 0.1, 0.15, 0.2, 0.25, 0.30, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.70, 0.75, 0.8, 0.85, 0.9, 0.95]
    evaluators = [avsbench_eval.Evaluator() for i in range(len(thrs))]

    for step, data in enumerate(tqdm(test_dataloader, desc=f"Evaluate AVSBench dataset({test_split})...")):
        images, audios, gts, labels, name = data['images'], data['audios'], data['gts'], data['labels'], data['ids']

        # Inference
        placeholder_tokens = model.get_placeholder_token(prompt_template.replace('{}', ''))
        placeholder_tokens = placeholder_tokens.repeat((test_dataloader.batch_size, 1))
        audio_driven_embedding = model.encode_audio(audios.to(model.device), placeholder_tokens, text_pos_at_prompt,
                                                    prompt_length)

        # Localization result
        out_dict = model(images.to(model.device), audio_driven_embedding, 224)

        # Evaluation for all thresholds
        for i, thr in enumerate(thrs):
            evaluators[i].evaluate_batch(out_dict['heatmap'], gts.to(model.device), thr)

        # Visual results
        for j in range(test_dataloader.batch_size):
            seg = out_dict['heatmap'][j:j+1]
            seg_image = ((1 - seg.squeeze().detach().cpu().numpy()) * 255).astype(np.uint8)

            os.makedirs(f'{result_dir}/heatmap', exist_ok=True)
            cv2.imwrite(f'{result_dir}/heatmap/{name[j]}.jpg', seg_image)

        # Overall figure
        for j in range(test_dataloader.batch_size):
            original_image = Image.open(os.path.join(test_dataloader.dataset.image_path, name[j] + '.png')).resize(
                (224, 224))
            gt_image = Image.open(os.path.join(test_dataloader.dataset.gt_path, name[j] + '.png')).resize(
                (224, 224)).point(
                lambda p: 255 - p)
            heatmap_image = Image.open(f'{result_dir}/heatmap/{name[j]}.jpg').resize((224, 224))
            seg_image = Image.open(f'{result_dir}/heatmap/{name[j]}.jpg').resize((224, 224)).point(
                lambda p: 0 if p / 255 < 0.5 else 255)

            draw_overall(result_dir, original_image, gt_image, heatmap_image, seg_image, labels[j], name[j])
            draw_overlaid(result_dir, original_image, heatmap_image, name[j])

    # Save result
    os.makedirs(result_dir, exist_ok=True)
    rst_path = os.path.join(f'{result_dir}/', 'test_rst.txt')
    msg = ''

    # Final result
    for i, thr in enumerate(thrs):
        audio_loc_key, audio_loc_dict = evaluators[i].finalize()

        msg += f'{model.__class__.__name__} ({test_split} with thr = {thr})\n'
        msg += 'mIoU={}, F={}\n'.format(audio_loc_dict['mIoU'], audio_loc_dict['Fmeasure'])

        if tensorboard_path is not None and epoch is not None:
            writer.add_scalars(f'test/avs({test_split})({thr})', audio_loc_dict, epoch)

    print(msg)
    with open(rst_path, 'w') as fp_rst:
        fp_rst.write(msg)

    if tensorboard_path is not None and epoch is not None:
        writer.close()


@torch.no_grad()
def eval_flickr_agg(
    model: torch.nn.Module,
    test_dataloader: DataLoader,
    result_dir: str,
    epoch: Optional[int] = None,
    tensorboard_path: Optional[str] = None
) -> None:
    '''
    Evaluate provided  model on AVSBench (S4, MS3) test dataset.

    Args:
        model (torch.nn.Module): Sound localization model to evaluate.
        test_dataloader (DataLoader): DataLoader for the test dataset.
        result_dir (str): Directory to save the evaluation results.
        epoch (int, optional): The current epoch number (default: None).
        tensorboard_path (str, optional): Path to store TensorBoard logs. If None, TensorBoard logs won't be written.

    Returns:
        None

    Notes:
        The evaluation includes threshold optimization for AVSBench.
    '''
    if tensorboard_path is not None and epoch is not None:
        os.makedirs(tensorboard_path, exist_ok=True)
        writer = SummaryWriter(tensorboard_path)

    test_split = test_dataloader.dataset.split

    # Get placeholder text
    prompt_template, text_pos_at_prompt, prompt_length = get_prompt_template()

    # Thresholds for evaluation
    thrs = [0.05, 0.1, 0.15, 0.2, 0.25, 0.30, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.70, 0.75, 0.8, 0.85, 0.9, 0.95]
    evaluators = [flickr_eval.Evaluator() for i in range(len(thrs))]

    for step, data in enumerate(tqdm(test_dataloader, desc="Evaluate Flickr dataset...")):
        images, audios, bboxes = data['images'], data['audios'], data['bboxes']
        labels, name = data['labels'], data['ids']

        # Inference
        placeholder_tokens = model.get_placeholder_token(prompt_template.replace('{}', ''))
        placeholder_tokens = placeholder_tokens.repeat((test_dataloader.batch_size, 1))
        audio_driven_embedding = model.encode_audio(audios.to(model.device), placeholder_tokens, text_pos_at_prompt,
                                                    prompt_length)

        # Localization result
        out_dict = model(images.to(model.device), audio_driven_embedding, 224)

        # Evaluation for all thresholds
        for i, thr in enumerate(thrs):
            evaluators[i].evaluate_batch(out_dict['heatmap'], bboxes, thr)

        # Visual results
        for j in range(test_dataloader.batch_size):
            seg = (out_dict['heatmap'][j:j+1])
            seg_image = ((1 - seg.squeeze().detach().cpu().numpy()) * 255).astype(np.uint8)

            os.makedirs(f'{result_dir}/heatmap', exist_ok=True)
            cv2.imwrite(f'{result_dir}/heatmap/{name[j]}.jpg', seg_image)

        # Overall figure
        for j in range(test_dataloader.batch_size):
            original_image = Image.open(os.path.join(test_dataloader.dataset.image_path, name[j] + '.jpg')).resize(
                (224, 224))
            gt_image = vt.ToPILImage()(bboxes[j]).resize((224, 224)).point(lambda p: 255 - p)
            heatmap_image = Image.open(f'{result_dir}/heatmap/{name[j]}.jpg').resize((224, 224))
            seg_image = Image.open(f'{result_dir}/heatmap/{name[j]}.jpg').resize((224, 224)).point(
                lambda p: 0 if p / 255 < 0.5 else 255)

            draw_overall(result_dir, original_image, gt_image, heatmap_image, seg_image, labels[j], name[j])
            draw_overlaid(result_dir, original_image, heatmap_image, name[j])

    # Save result
    os.makedirs(result_dir, exist_ok=True)
    rst_path = os.path.join(f'{result_dir}/', 'test_rst.txt')
    msg = ''

    # Final result (aggressive)
    for i, thr in enumerate(thrs):
        audio_loc_key, audio_loc_dict = evaluators[i].finalize()

        msg += f'{model.__class__.__name__} ({test_split} with thr = {thr})\n'
        msg += 'AP50(cIoU)={}, AUC={}\n'.format(audio_loc_dict['cIoU'], audio_loc_dict['AUC'])

        if tensorboard_path is not None and epoch is not None:
            writer.add_scalars(f'test/flickr({thr})', audio_loc_dict, epoch)

    print(msg)
    with open(rst_path, 'w') as fp_rst:
        fp_rst.write(msg)

    if tensorboard_path is not None and epoch is not None:
        writer.close()


@torch.no_grad()
def eval_exvggss_agg(
    model: torch.nn.Module,
    test_dataloader: DataLoader,
    result_dir: str,
    epoch: Optional[int] = None,
    tensorboard_path: Optional[str] = None
) -> None:
    '''
    Evaluate provided  model on AVSBench (S4, MS3) test dataset.

    Args:
        model (torch.nn.Module): Sound localization model to evaluate.
        test_dataloader (DataLoader): DataLoader for the test dataset.
        result_dir (str): Directory to save the evaluation results.
        epoch (int, optional): The current epoch number (default: None).
        tensorboard_path (str, optional): Path to store TensorBoard logs. If None, TensorBoard logs won't be written.

    Returns:
        None

    Notes:
        The evaluation includes threshold optimization for AVSBench.
    '''
    if tensorboard_path is not None and epoch is not None:
        os.makedirs(tensorboard_path, exist_ok=True)
        writer = SummaryWriter(tensorboard_path)

    test_split = test_dataloader.dataset.split

    # Get placeholder text
    prompt_template, text_pos_at_prompt, prompt_length = get_prompt_template()

    # Thresholds for evaluation
    thrs = [0.05, 0.1, 0.15, 0.2, 0.25, 0.30, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.70, 0.75, 0.8, 0.85, 0.9, 0.95]
    evaluators = [exvggss_eval.Evaluator() for i in range(len(thrs))]

    for step, data in enumerate(tqdm(test_dataloader, desc="Evaluate Extend VGG-SS dataset...")):
        images, audios, bboxes,  = data['images'], data['audios'], data['bboxes']
        labels, name = data['labels'], data['ids']

        # Inference
        placeholder_tokens = model.get_placeholder_token(prompt_template.replace('{}', ''))
        placeholder_tokens = placeholder_tokens.repeat((test_dataloader.batch_size, 1))
        audio_driven_embedding = model.encode_audio(audios.to(model.device), placeholder_tokens, text_pos_at_prompt,
                                                    prompt_length)

        # Localization result
        out_dict = model(images.to(model.device), audio_driven_embedding, 224)

        # Calculate confidence value for extended dataset
        v_f = model.encode_masked_vision(images.to(model.device), audio_driven_embedding)[0]
        ind = torch.arange(test_dataloader.batch_size).to(images.device)
        confs = torch.cosine_similarity(v_f[ind, ind, :], audio_driven_embedding)

        # Evaluation for all thresholds
        for i, thr in enumerate(thrs):
            evaluators[i].evaluate_batch(out_dict['heatmap'], bboxes, labels, confs, name, thr)

    # Save result
    os.makedirs(result_dir, exist_ok=True)
    rst_path = os.path.join(f'{result_dir}/', 'test_rst.txt')
    msg = ''

    # Final result
    for i, thr in enumerate(thrs):
        audio_loc_key, audio_loc_dict = evaluators[i].finalize()

        msg += f'{model.__class__.__name__} ({test_split} with thr = {thr})\n'
        msg += 'AP={}, Max-F1={}\n'.format(audio_loc_dict['AP'], audio_loc_dict['Max-F1'])

        if tensorboard_path is not None and epoch is not None:
            writer.add_scalars(f'test/exvggss({thr})', audio_loc_dict, epoch)

    print(msg)
    with open(rst_path, 'w') as fp_rst:
        fp_rst.write(msg)

    if tensorboard_path is not None and epoch is not None:
        writer.close()


@torch.no_grad()
def eval_exflickr_agg(
    model: torch.nn.Module,
    test_dataloader: DataLoader,
    result_dir: str,
    epoch: Optional[int] = None,
    tensorboard_path: Optional[str] = None
) -> None:
    '''
    Evaluate provided  model on AVSBench (S4, MS3) test dataset.

    Args:
        model (torch.nn.Module): Sound localization model to evaluate.
        test_dataloader (DataLoader): DataLoader for the test dataset.
        result_dir (str): Directory to save the evaluation results.
        epoch (int, optional): The current epoch number (default: None).
        tensorboard_path (str, optional): Path to store TensorBoard logs. If None, TensorBoard logs won't be written.

    Returns:
        None

    Notes:
        The evaluation includes threshold optimization for AVSBench.
    '''
    if tensorboard_path is not None and epoch is not None:
        os.makedirs(tensorboard_path, exist_ok=True)
        writer = SummaryWriter(tensorboard_path)

    test_split = test_dataloader.dataset.split

    # Get placeholder text
    prompt_template, text_pos_at_prompt, prompt_length = get_prompt_template()

    # Thresholds for evaluation
    thrs = [0.05, 0.1, 0.15, 0.2, 0.25, 0.30, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.70, 0.75, 0.8, 0.85, 0.9, 0.95]
    evaluators = [exflickr_eval.Evaluator() for i in range(len(thrs))]

    for step, data in enumerate(tqdm(test_dataloader, desc="Evaluate Extend Flickr dataset...")):
        images, audios, bboxes,  = data['images'], data['audios'], data['bboxes']
        labels, name = data['labels'], data['ids']

        # Inference
        placeholder_tokens = model.get_placeholder_token(prompt_template.replace('{}', ''))
        placeholder_tokens = placeholder_tokens.repeat((test_dataloader.batch_size, 1))
        audio_driven_embedding = model.encode_audio(audios.to(model.device), placeholder_tokens, text_pos_at_prompt,
                                                    prompt_length)

        # Localization result
        out_dict = model(images.to(model.device), audio_driven_embedding, 224)

        # Calculate confidence value for extended dataset
        v_f = model.encode_masked_vision(images.to(model.device), audio_driven_embedding)[0]
        ind = torch.arange(test_dataloader.batch_size).to(images.device)
        confs = torch.cosine_similarity(v_f[ind, ind, :], audio_driven_embedding)

        # Evaluation for all thresholds
        for i, thr in enumerate(thrs):
            evaluators[i].evaluate_batch(out_dict['heatmap'], bboxes, labels, confs, name, thr)

    # Save result
    os.makedirs(result_dir, exist_ok=True)
    rst_path = os.path.join(f'{result_dir}/', 'test_rst.txt')
    msg = ''

    # Final result
    for i, thr in enumerate(thrs):
        audio_loc_key, audio_loc_dict = evaluators[i].finalize()

        msg += f'{model.__class__.__name__} ({test_split} with thr = {thr})\n'
        msg += 'AP={}, Max-F1={}\n'.format(audio_loc_dict['AP'], audio_loc_dict['Max-F1'])

        if tensorboard_path is not None and epoch is not None:
            writer.add_scalars(f'test/exflickr({thr})', audio_loc_dict, epoch)

    print(msg)
    with open(rst_path, 'w') as fp_rst:
        fp_rst.write(msg)

    if tensorboard_path is not None and epoch is not None:
        writer.close()
