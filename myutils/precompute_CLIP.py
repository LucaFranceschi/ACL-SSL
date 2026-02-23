import torch
import os
import yaml
import argparse
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor # For Parallel IO
from modules.CLIPSeg.clipseg_for_audio import CLIPSeg

from utils.util import get_prompt_template, fix_seed, seed_worker
from datasets.vggsound.VGGSound_Dataset import VGGSoundDataset

# Helper function for background saving
def save_worker(tensor, path, pp):
    """Saves a single tensor to disk. Runs in a background thread."""
    if pp == 1:
        os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(tensor, path)

@torch.no_grad()
def main(model_name, model_path, train_config_name, data_path_dict, save_path):
    device = torch.device("cuda" if USE_CUDA else "cpu")
    print(f'Device: {device} is used\n')
    print(f'Precomputing image embeddings v_D {train_config_name} and storing results in {save_path}')

    ''' Get train configure '''
    train_conf_file = f'./config/train/{train_config_name}.yaml'
    with open(train_conf_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        args = argparse.Namespace(**config['common'])
        args.optim = config['optim_conf'][config['optimizer']]

    ''' Fix random seed'''
    fix_seed(args.seed)

    # Get Test Dataloader (VGGSound)
    subset=''
    train_dataset = VGGSoundDataset(data_path_dict['vggsound'], f'vggsound_train{subset}', is_train=True,
        input_resolution=args.input_resolution, noise_transform_train=False, set_length=3)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=256,
        num_workers=args.num_workers, pin_memory=False, drop_last=True, worker_init_fn=seed_worker, shuffle=False)

    av_grounder = CLIPSeg.from_pretrained(model_path + "/clipseg-rd64-refined-local")
    av_grounder.to(device=device)
    av_grounder.requires_grad_(False)

    # Use a ThreadPoolExecutor for concurrent disk writes
    with ThreadPoolExecutor(max_workers=16) as executor:
        for pretrain_pass in range(1, 7):
            pbar = tqdm(train_dataloader, desc=f"Pass [{pretrain_pass}/{7}]")

            for step, data in enumerate(pbar):
                images, ids = data['images'].to(device), data['ids']

                # 1. GPU Computation
                vision_outputs = av_grounder.get_pixels(images)

                # 2. Preparation for IO
                # Move to CPU and unbind to separate the batch
                # .cpu().clone() ensures we don't hold onto GPU memory in the background thread
                tensors_to_save = [t.cpu().clone() for t in torch.unbind(vision_outputs, dim=0)]

                # 3. Offload to Background Threads
                for tensor, id in zip(tensors_to_save, ids):
                    file_name = f"{pretrain_pass}.pt"
                    full_path = os.path.join(save_path, id, file_name)

                    # This returns immediately; saving happens in the background
                    executor.submit(save_worker, tensor, full_path, pretrain_pass)

    print("All tasks submitted. Waiting for final writes to finish...")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='', help='Use model config file name')
    parser.add_argument('--model_path', type=str, default='', help='Use model save path')
    parser.add_argument('--model_weights', type=str, default='', help='Path for model weights')
    parser.add_argument('--train_config', type=str, default='', help='Use train config file name')
    parser.add_argument('--save_path', type=str, default='', help='Save path for results')
    parser.add_argument('--vggsound_path', type=str, default='', help='VGGSound dataset directory')

    args = parser.parse_args()

    data_path = {'vggsound': args.vggsound_path}

    USE_CUDA = torch.cuda.is_available()

    # Check the number of GPUs for training
    NUM_GPUS = len(os.environ.get('CUDA_VISIBLE_DEVICES', '').split(','))

    if NUM_GPUS == 1:
        main(args.model_name, args.model_path, args.train_config, data_path, args.save_path)

    exit(1)
