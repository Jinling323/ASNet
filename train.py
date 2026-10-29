"""Two-stage ASNet training for SHA/<variant>/{train,val} jpg + point-npy pairs."""

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.crowd_count import CrowdCount
from src.network import load_net, save_net, weights_normal_init
from src.sha_dataset import SHADataset, attention_labels, estimate_thresholds


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data-root', type=Path, default=Path('SHA/mix_level'))
    parser.add_argument('--output', type=Path, help='Default: runs/asnet_mix_level, or resume directory')
    parser.add_argument('--stage', choices=['all', 'prior', 'asnet'], default='all')
    parser.add_argument('--prior-epochs', type=int, default=50)
    parser.add_argument('--epochs', type=int, default=300, help='ASNet epochs, including completed epochs on resume')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--prior-batch-size', type=int, default=1)
    parser.add_argument('--crop-size', type=int, default=128, help='ASNet crop; multiple of 64 for the original APLoss')
    parser.add_argument('--prior-crop-size', type=int, default=256)
    parser.add_argument('--patches-per-image', type=int, default=9)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--prior-lr', type=float, default=1e-5)
    parser.add_argument('--weight-decay', type=float, default=0)
    parser.add_argument('--workers', type=int, default=0,
                        help='Data loader processes; 0 also works in restricted environments')
    parser.add_argument('--device', default='auto', help='auto, cpu, cuda, or cuda:N')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cpu-threads', type=int, default=4)
    parser.add_argument('--kernel-size', type=int, default=15)
    parser.add_argument('--sigma', type=float, default=4.0)
    parser.add_argument('--color-jitter', type=float, default=0.2)
    parser.add_argument('--threshold-samples', type=int, default=64)
    parser.add_argument('--pretrained', choices=['imagenet', 'none'], default='imagenet')
    parser.add_argument('--vgg-weights', type=Path, help='Local torchvision VGG16 state_dict')
    parser.add_argument('--prior-checkpoint', type=Path, help='prior_best.pt for --stage asnet')
    parser.add_argument('--init-checkpoint', type=Path, help='Full ASNet .h5 or this trainer\'s .pt for fine-tuning')
    parser.add_argument('--resume', type=Path, help='Resume this trainer\'s last.pt, including optimizer and RNG')
    parser.add_argument('--max-val-size', type=int, default=0, help='Optional longest-side limit; 0 keeps original resolution')
    parser.add_argument('--max-train-batches', type=int, default=0, help='Debug only; 0 processes all batches')
    parser.add_argument('--max-val-batches', type=int, default=0, help='Debug only; 0 evaluates all images')
    parser.add_argument('--log-every', type=int, default=50)
    args = parser.parse_args()
    positive = ['prior_epochs', 'epochs', 'batch_size', 'prior_batch_size', 'patches_per_image',
                'lr', 'prior_lr', 'sigma', 'threshold_samples', 'cpu_threads', 'log_every']
    for key in positive:
        if getattr(args, key) <= 0:
            parser.error('--{} must be positive'.format(key.replace('_', '-')))
    if args.crop_size < 64 or args.crop_size % 64:
        parser.error('--crop-size must be a positive multiple of 64')
    if args.prior_crop_size < 16 or args.prior_crop_size % 16:
        parser.error('--prior-crop-size must be a positive multiple of 16')
    if args.kernel_size < 1 or args.kernel_size % 2 != 1:
        parser.error('--kernel-size must be a positive odd number')
    if not 0 <= args.color_jitter <= 1:
        parser.error('--color-jitter must be between 0 and 1')
    if any(getattr(args, key) < 0 for key in ['workers', 'max_val_size', 'max_train_batches',
                                            'max_val_batches', 'weight_decay']):
        parser.error('Worker counts, size/batch limits and weight decay cannot be negative')
    if sum(bool(x) for x in [args.resume, args.prior_checkpoint, args.init_checkpoint]) > 1:
        parser.error('Choose only one of --resume, --prior-checkpoint and --init-checkpoint')
    if (args.prior_checkpoint or args.init_checkpoint) and args.stage != 'asnet':
        parser.error('--prior-checkpoint/--init-checkpoint require --stage asnet')
    if args.stage == 'asnet' and not (args.resume or args.prior_checkpoint or args.init_checkpoint):
        parser.error('--stage asnet requires a trained prior or full model checkpoint')
    return args


def seed_worker(_):
    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)


def load_checkpoint(path):
    # Checkpoints contain only tensors and primitive containers, including RNG states.
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(checkpoint, dict) or checkpoint.get('format_version') != 1:
        raise ValueError('{} is not a train.py checkpoint'.format(path))
    return checkpoint


def initialize_vgg(net, args):
    weights_normal_init(net, dev=0.01)
    if args.pretrained == 'none' and not args.vgg_weights:
        print('Random initialization (--pretrained none); use ImageNet weights for real training.', flush=True)
        return
    cached = Path(torch.hub.get_dir()) / 'checkpoints/vgg16-397923af.pth'
    path = args.vgg_weights or (cached if cached.is_file() else None)
    if path:
        state = torch.load(path, map_location='cpu', weights_only=True)
    else:
        print('Downloading ImageNet VGG16 weights into the output directory...', flush=True)
        state = torch.hub.load_state_dict_from_url(
            'https://download.pytorch.org/models/vgg16-397923af.pth',
            model_dir=str(args.output / 'pretrained'), map_location='cpu', check_hash=True)
    indices = [0, 2, 5, 7, 10, 12, 14, 17, 19, 21, 24, 26, 28]
    with torch.no_grad():
        for branch in [net.features.prior, net.features.vgg16]:
            layers = [m for m in branch.modules() if isinstance(m, torch.nn.Conv2d)][:13]
            for layer, index in zip(layers, indices):
                layer.weight.copy_(state['features.{}.weight'.format(index)])
                layer.bias.copy_(state['features.{}.bias'.format(index)])
    print('Initialized the first 13 convolutions of both branches from VGG16.', flush=True)


def make_loader(dataset, args, training=False, prior=False):
    return DataLoader(dataset, batch_size=(args.prior_batch_size if prior else args.batch_size) if training else 1,
                      shuffle=training, num_workers=args.workers, pin_memory=args.device.type == 'cuda',
                      worker_init_fn=seed_worker, timeout=60 if args.workers else 0)


def run_epoch(net, loader, args, stage, thresholds, optimizer=None):
    training = optimizer is not None
    net.train(training)
    if stage == 'asnet':
        net.features.prior.eval()
    total_loss = total_correct = total_pixels = absolute = squared = 0.0
    foreground = batches = samples = 0
    limit = args.max_train_batches if training else args.max_val_batches
    with torch.set_grad_enabled(training):
        for index, blob in enumerate(loader):
            image, density, roi = [blob[key].to(args.device, non_blocking=True)
                                   for key in ['image', 'density', 'roi']]
            if training:
                optimizer.zero_grad(set_to_none=True)
            if stage == 'prior':
                logits = net.features.prior(image)
                labels = attention_labels(density, thresholds['attention'])
                valid = F.interpolate(roi, size=labels.shape[-2:], mode='nearest').squeeze(1)
                loss_map = F.cross_entropy(logits, labels, reduction='none')
                loss = (loss_map * valid).sum() / valid.sum().clamp_min(1)
                total_correct += ((logits.argmax(1) == labels) * valid).sum().item()
                total_pixels += valid.sum().item()
            else:
                estimate, _, visuals = net(image, roi, density if training else None)
                foreground += int((visuals['masks'].sum(1) > 0).sum().item())
                if training:
                    loss = net.loss
                else:
                    # Count the entire image, including its final partial 8x8 cell.
                    error = estimate.sum().item() - blob['count'].item()
                    absolute += abs(error)
                    squared += error ** 2
                    loss = None
            if loss is not None:
                if not torch.isfinite(loss):
                    raise RuntimeError('Non-finite {} loss at batch {}'.format(stage, index + 1))
                if training:
                    loss.backward()
                    optimizer.step()
                total_loss += loss.item() * image.shape[0]
            samples += image.shape[0]
            batches += 1
            if (index + 1) % args.log_every == 0:
                metric = 'MAE' if stage == 'asnet' and not training else 'loss'
                value = absolute / samples if metric == 'MAE' else total_loss / samples
                print('  {} {} batch {}/{} {}={:.6g}'.format(
                    stage, 'train' if training else 'val', index + 1, len(loader),
                    metric, value), flush=True)
            if limit and batches >= limit:
                break
    if not samples:
        raise RuntimeError('Empty data loader')
    if stage == 'asnet' and training and not foreground:
        raise RuntimeError('The prior predicted background for every training pixel; train the prior longer.')
    result = {'samples': samples}
    if training or stage == 'prior':
        result['loss'] = total_loss / samples
    if stage == 'prior':
        result['accuracy'] = total_correct / max(total_pixels, 1)
    elif not training:
        result.update(mae=absolute / samples, rmse=math.sqrt(squared / samples))
    return result


def save_checkpoint(path, net, optimizer, stage, epoch, best, thresholds, args):
    state = np.random.get_state()
    checkpoint = {
        'format_version': 1, 'stage': stage, 'epoch': epoch, 'best': best,
        'model': net.state_dict(), 'optimizer': optimizer.state_dict(), 'thresholds': thresholds,
        'args': {key: str(value) if isinstance(value, (Path, torch.device)) else value
                 for key, value in vars(args).items()},
        'torch_rng': torch.get_rng_state(), 'python_rng': random.getstate(),
        'numpy_rng': (state[0], state[1].tolist(), state[2], state[3], state[4]),
        'cuda_rng': torch.cuda.get_rng_state_all() if args.device.type == 'cuda' else None,
    }
    temporary = path.with_suffix('.tmp')
    torch.save(checkpoint, temporary)
    os.replace(temporary, path)


def restore_rng(checkpoint, device):
    torch.set_rng_state(checkpoint['torch_rng'])
    random.setstate(checkpoint['python_rng'])
    state = checkpoint['numpy_rng']
    np.random.set_state((state[0], np.array(state[1], dtype=np.uint32), *state[2:]))
    if device.type == 'cuda' and checkpoint['cuda_rng'] is not None:
        torch.cuda.set_rng_state_all(checkpoint['cuda_rng'])


def main():
    args = parse_args()
    args.output = args.output or (args.resume.parent if args.resume else Path('runs/asnet_mix_level'))
    if args.device == 'auto':
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.device = torch.device(args.device)
    if args.device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable in {}'.format(sys.executable))
    torch.set_num_threads(args.cpu_threads)
    print('Python: {}\nDevice: {}\nDataset: {}'.format(sys.executable, args.device, args.data_root.resolve()), flush=True)
    if args.device.type == 'cpu':
        print('CPU execution enabled; full training will be slow.', flush=True)
    checkpoint_path = args.resume or args.prior_checkpoint or args.init_checkpoint
    checkpoint = load_checkpoint(checkpoint_path) if checkpoint_path and checkpoint_path.suffix.lower() != '.h5' else None
    if args.resume and checkpoint is None:
        raise ValueError('--resume requires a .pt training checkpoint')
    if args.resume and args.stage != 'all' and checkpoint['stage'] != args.stage:
        raise ValueError('--stage does not match the resume checkpoint')
    if args.prior_checkpoint and (checkpoint is None or checkpoint['stage'] != 'prior'):
        raise ValueError('--prior-checkpoint must contain a trained prior stage')
    if args.init_checkpoint and checkpoint and checkpoint['stage'] != 'asnet':
        raise ValueError('Use --prior-checkpoint for a prior-stage checkpoint')
    if checkpoint:
        # Keep density construction, crop geometry and augmentation consistent with the checkpoint.
        for key in ['kernel_size', 'sigma', 'crop_size', 'prior_crop_size', 'patches_per_image',
                    'color_jitter', 'max_val_size', 'seed']:
            setattr(args, key, checkpoint['args'][key])
        print('Restored density generation and crop settings from {}'.format(checkpoint_path), flush=True)
        if args.resume:
            for key in ['lr', 'prior_lr', 'weight_decay']:
                setattr(args, key, checkpoint['args'][key])
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device.type == 'cuda':
        torch.cuda.manual_seed_all(args.seed)
    if not args.resume and any((args.output / name).exists() for name in ['last.pt', 'prior_best.pt', 'best.pt']):
        raise FileExistsError('Output already contains checkpoints; use --resume or a different --output')
    args.output.mkdir(parents=True, exist_ok=True)
    dataset_options = dict(kernel_size=args.kernel_size, sigma=args.sigma, color_jitter=args.color_jitter)
    train_data = SHADataset(args.data_root / 'train', crop_size=args.crop_size,
                            patches_per_image=args.patches_per_image, **dataset_options)
    prior_data = SHADataset(args.data_root / 'train', crop_size=args.prior_crop_size,
                            patches_per_image=args.patches_per_image, **dataset_options)
    val_data = SHADataset(args.data_root / 'val', max_size=args.max_val_size, **dataset_options)
    print('Train: {} images, {} patches/epoch; validation: {} images'.format(
        len(train_data.images), len(train_data), len(val_data)), flush=True)
    if args.max_train_batches or args.max_val_batches:
        print('DEBUG batch limits active; validation metrics cover only the selected subset.', flush=True)
    net = CrowdCount()
    if checkpoint:
        net.load_state_dict(checkpoint['model'])
    elif args.init_checkpoint:
        load_net(str(args.init_checkpoint), net)
    else:
        initialize_vgg(net, args)
    if checkpoint:
        thresholds = checkpoint['thresholds']
    else:
        print('Estimating density thresholds from training images...', flush=True)
        thresholds = estimate_thresholds(train_data, args.threshold_samples, args.seed)
    print('Density thresholds: {}'.format(thresholds), flush=True)
    net.this_dataset_density_level = (thresholds['pyramid'], 0.0)
    net.to(args.device)
    with (args.output / 'config.json').open('w') as file:
        json.dump({**{key: str(value) if isinstance(value, (Path, torch.device)) else value
                      for key, value in vars(args).items()}, 'thresholds': thresholds}, file, indent=2)
    stages = ['prior', 'asnet'] if args.stage == 'all' else [args.stage]
    if args.resume and checkpoint['stage'] == 'asnet':
        stages = ['asnet']
    for stage in stages:
        if stage == 'asnet' and args.stage == 'all' and not (args.resume and checkpoint['stage'] == 'asnet'):
            prior_path = args.output / 'prior_best.pt'
            if not prior_path.is_file() and args.resume:
                prior_path = args.resume.parent / 'prior_best.pt'
            best_prior = load_checkpoint(prior_path)
            net.load_state_dict(best_prior['model'])
            del best_prior
        for name, parameter in net.named_parameters():
            parameter.requires_grad_(name.startswith('features.prior.') == (stage == 'prior'))
        optimizer = torch.optim.Adam([p for p in net.parameters() if p.requires_grad],
                                     lr=args.prior_lr if stage == 'prior' else args.lr,
                                     weight_decay=args.weight_decay)
        start, best = 0, float('inf')
        if args.resume and checkpoint['stage'] == stage:
            optimizer.load_state_dict(checkpoint['optimizer'])
            start, best = checkpoint['epoch'], checkpoint['best']
            restore_rng(checkpoint, args.device)
        training = make_loader(prior_data if stage == 'prior' else train_data, args, training=True, prior=stage == 'prior')
        validation = make_loader(val_data, args)
        epochs = args.prior_epochs if stage == 'prior' else args.epochs
        for epoch in range(start, epochs):
            begin = time.monotonic()
            train_result = run_epoch(net, training, args, stage, thresholds, optimizer)
            val_result = run_epoch(net, validation, args, stage, thresholds)
            score = val_result['loss'] if stage == 'prior' else val_result['mae']
            improved = score < best
            best = min(best, score)
            record = dict(stage=stage, epoch=epoch + 1, train=train_result, val=val_result,
                          best=best, seconds=time.monotonic() - begin)
            print(json.dumps(record), flush=True)
            with (args.output / 'metrics.jsonl').open('a') as file:
                file.write(json.dumps(record) + '\n')
            save_checkpoint(args.output / 'last.pt', net, optimizer, stage, epoch + 1, best, thresholds, args)
            if improved:
                best_name = 'prior_best.pt' if stage == 'prior' else 'best.pt'
                save_checkpoint(args.output / best_name, net, optimizer, stage, epoch + 1, best, thresholds, args)
                if stage == 'asnet':
                    save_net(str(args.output / 'best.h5'), net)
        print('{} finished; best validation {}={:.6g}'.format(
            stage, 'cross entropy' if stage == 'prior' else 'MAE', best), flush=True)
    print('Checkpoints and metrics: {}'.format(args.output.resolve()), flush=True)


if __name__ == '__main__':
    main()
