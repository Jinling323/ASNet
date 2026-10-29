import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

from src.crowd_count import CrowdCount
from src.network import load_net, save_net
from src.sha_dataset import SHADataset, attention_labels, estimate_thresholds, point_density
from train import load_checkpoint, restore_rng, save_checkpoint


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_density_preserves_duplicate_and_boundary_points(self):
        points = np.array([[0, 0], [0, 0], [10, 8], [11.2, -0.2]], dtype=np.float32)
        density = point_density(points, height=9, width=11)
        self.assertTrue(np.isfinite(density).all())
        self.assertAlmostEqual(float(density.sum()), len(points), places=5)
        self.assertEqual(float(point_density(np.empty((0, 2)), 9, 11).sum()), 0)

    def test_sha_adapter_padding_resizing_and_both_annotation_formats(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for columns in [2, 3]:
                Image.new('RGB', (81, 73)).save(root / '{}.jpg'.format(columns))
                points = np.array([[0, 0, 4], [81.2, 72.8, 8]], dtype=np.float32)
                np.save(root / '{}.npy'.format(columns), points[:, :columns])
            for max_size in [0, 48]:
                dataset = SHADataset(root, max_size=max_size)
                for sample in dataset:
                    _, h, w = sample['image'].shape
                    self.assertEqual(h % 16, 0)
                    self.assertEqual(w % 16, 0)
                    self.assertEqual(sample['density'].shape, (1, h // 8, w // 8))
                    self.assertAlmostEqual(sample['density'].sum().item(), sample['count'], places=5)
                    self.assertEqual(sample['roi'][0, -1, -1], 0)
            crops = SHADataset(root, crop_size=128, patches_per_image=2)
            self.assertEqual(len(crops), 4)
            self.assertEqual(crops[0]['image'].shape, (3, 128, 128))
            self.assertTrue(all(value > 0 for value in estimate_thresholds(dataset, 2).values()))

    def test_attention_labels_have_three_density_classes(self):
        density = torch.zeros(1, 1, 32, 32)
        density[0, 0, 2, 2] = 0.64
        density[0, 0, 24, 24] = 64
        labels = attention_labels(density, threshold=0.1)
        self.assertEqual(set(labels.unique().tolist()), {0, 1, 2})

    def test_asnet_backward_keeps_prior_frozen_and_h5_roundtrip(self):
        net = CrowdCount()
        # A deterministic foreground prior isolates the counting branch's gradient contract.
        final = net.features.prior[-1].conv
        with torch.no_grad():
            final.weight.zero_()
            final.bias.copy_(torch.tensor([0., 1., 0.]))
        net.train()
        prediction, _, _ = net(torch.rand(2, 3, 128, 128), torch.ones(2, 1, 128, 128),
                               torch.ones(2, 1, 16, 16))
        self.assertEqual(prediction.shape, (2, 1, 16, 16))
        self.assertTrue(torch.isfinite(net.loss))
        net.loss.backward()
        self.assertTrue(all(p.grad is None for p in net.features.prior.parameters()))
        self.assertTrue(any(p.grad is not None for p in net.features.map.parameters()))
        # The legacy evaluator can read the trainer's exported HDF5 format.
        with tempfile.TemporaryDirectory() as folder:
            small = torch.nn.Linear(3, 1)
            path = str(Path(folder) / 'model.h5')
            save_net(path, small)
            restored = torch.nn.Linear(3, 1)
            load_net(path, restored)
            torch.testing.assert_close(restored.weight, small.weight)

    def test_checkpoint_restores_optimizer_and_all_cpu_rngs(self):
        with tempfile.TemporaryDirectory() as folder:
            net = torch.nn.Linear(2, 1)
            optimizer = torch.optim.Adam(net.parameters())
            net(torch.ones(1, 2)).sum().backward()
            optimizer.step()
            args = SimpleNamespace(device=torch.device('cpu'), output=Path(folder))
            path = Path(folder) / 'last.pt'
            save_checkpoint(path, net, optimizer, 'asnet', 3, 1.2, {'pyramid': 2.0}, args)
            expected = (random.random(), np.random.random(), torch.rand(2))
            checkpoint = load_checkpoint(path)
            restore_rng(checkpoint, torch.device('cpu'))
            self.assertEqual(random.random(), expected[0])
            self.assertEqual(np.random.random(), expected[1])
            torch.testing.assert_close(torch.rand(2), expected[2])
            restored_optimizer = torch.optim.Adam(net.parameters())
            restored_optimizer.load_state_dict(checkpoint['optimizer'])
            self.assertEqual(len(restored_optimizer.state), len(optimizer.state))
            self.assertEqual(checkpoint['epoch'], 3)


if __name__ == '__main__':
    unittest.main()
