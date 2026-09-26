"""Predict Task 2 CSI from a saved PhysicsWiFo2 checkpoint (notebook/task2_physics.ipynb),
without retraining.

PhysicsWiFo2 predicts H2_hat = ridge(A1) + alpha*wifo2_correction(A1, rgb, pos) in the
angle-delay domain, then transforms back to space domain. The checkpoint holds only the small
trainable pieces (wifo2's task-2 heads, ~98K params; the ridge layer, ~147K real params;
alpha) -- everything else comes from weights/model_best.pkl, so the model is rebuilt as
base + overlay, same pattern as predict_task2.py.

  python predict_task2_physics.py                  # predictions only
  python predict_task2_physics.py --submission      # also assemble a submission json

The grader reads task1/task2/task3 unconditionally and flattens each, so task3 must be
present even though dataset/Task3 does not ship locally; it is copied from the challenge's
demo file. task1 is carried over from --base untouched.
"""
import argparse
import json
from argparse import Namespace
from pathlib import Path

import h5py
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import WiFo2_model
from main import freeze_wifo

ROOT = Path(__file__).parent.parent


def load_mat(path, key):
    """Read one variable from a .mat file, v7 or v7.3."""
    try:
        return sio.loadmat(path)[key]
    except NotImplementedError:
        with h5py.File(path, 'r') as f:
            a = f[key][...]
        if a.dtype.names == ('real', 'imag'):
            a = a['real'] + 1j * a['imag']
        return a.transpose(range(a.ndim)[::-1])


def device():
    if torch.cuda.is_available():
        return 'cuda'
    return 'mps' if torch.backends.mps.is_available() else 'cpu'


def to_ad_torch(x):
    """(B,2,N,K) real-pair, space domain -> (B,2,N,K) real-pair, angle-delay domain.

    Forced onto cpu: MPS's complex/FFT op coverage is inconsistent across torch versions, and
    these tensors are tiny (128x64), so the round-trip cost is negligible.
    """
    dev = x.device
    c = torch.complex(x[:, 0], x[:, 1]).cpu()
    a = torch.fft.ifft(torch.fft.ifft(c, dim=1), dim=2)
    return torch.stack([a.real, a.imag], dim=1).to(dev)


def from_ad_torch(x):
    """(B,2,N,K) real-pair, angle-delay domain -> (B,2,N,K) real-pair, space domain."""
    dev = x.device
    c = torch.complex(x[:, 0], x[:, 1]).cpu()
    z = torch.fft.fft(torch.fft.fft(c, dim=2), dim=1)
    return torch.stack([z.real, z.imag], dim=1).to(dev)


class LocalRidgeLayer(nn.Module):
    """Per-cell complex ridge over a (2r+1)^2 neighbourhood, as trainable torch parameters.

    `c_init` only needs the right shape here -- actual weights come from the checkpoint via
    load_state_dict, so zeros are fine as a placeholder init.
    """
    def __init__(self, c_init, sh):
        super().__init__()
        self.sh = sh
        w = torch.from_numpy(c_init)
        self.w_real = nn.Parameter(w.real.float().clone())
        self.w_imag = nn.Parameter(w.imag.float().clone())

    def forward(self, x):
        c = torch.complex(x[:, 0], x[:, 1])
        w = torch.complex(self.w_real, self.w_imag)
        shifted = torch.stack([torch.roll(torch.roll(c, da, 1), dd, 2) for da, dd in self.sh], dim=-1)
        pred = (shifted * w).sum(-1)
        return torch.stack([pred.real, pred.imag], dim=1)


class PhysicsWiFo2(nn.Module):
    """H2_hat_ad = ridge(A1) + alpha * wifo2_correction(A1, rgb, pos); H2_hat = IFFT^-1(...)."""
    def __init__(self, wifo2, ridge):
        super().__init__()
        self.wifo2 = wifo2
        self.ridge = ridge
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, H1, rgb=None, pos=None, mask_ratio=0.5, mask_strategy='fre', use_modal=True,
               model_args=None):
        model_args.rgb_pos_enhance = int(use_modal)   # model.py:1295 reads this, not the kwargs
        A1 = to_ad_torch(H1)
        ridge_ad = self.ridge(A1)
        kw = dict(rgb=rgb, pos=pos) if use_modal else {}
        wifo_ad = self.wifo2(A1, mask_ratio=mask_ratio, mask_strategy=mask_strategy,
                             data='none', task_id=2, **kw)
        pred_ad = ridge_ad + self.alpha * wifo_ad
        return from_ad_torch(pred_ad)


def build(ckpt_path, dev):
    """Rebuild PhysicsWiFo2: base WiFo2 checkpoint + freeze_wifo, plus the saved small overlay."""
    ck = torch.load(ckpt_path, map_location=dev, weights_only=False)
    m = ck['meta']
    args = Namespace(
        task='Test', task_id=2, note='', data_path=str(ROOT / 'dataset'),
        wifo_weights=str(ROOT / m['base_weights']),
        fastdepth_weights=str(ROOT / 'weights/FastDepthV2_L1_Best.pth'),
        process_name='predict', snr_db=25, csi_enhance=1,
        rgb_pos_enhance=int(m['use_modal']),
        MoE=m['MoE'], patch_size=m['patch_size'], t_patch_size=m['t_patch_size'],
        size=m['size'], no_qkv_bias=m['no_qkv_bias'], pos_emb=m['pos_emb'],
        lr=1e-5, min_lr=1e-6, epochs=0, early_stop=5, weight_decay=0,
        batch_size=16, log_interval=5, device_id='0', machine='predict',
        clip_grad=0.05, lr_anneal_steps=200,
    )
    wifo2 = WiFo2_model(args=args).to(dev)
    wifo2.load_state_dict(torch.load(args.wifo_weights, map_location=dev, weights_only=False),
                          strict=False)
    wifo2 = freeze_wifo(args, wifo2, task_id=2)

    sh = [tuple(p) for p in m['sh']]
    ridge = LocalRidgeLayer(np.zeros((128, 64, len(sh)), dtype=np.complex128), sh)

    model = PhysicsWiFo2(wifo2, ridge).to(dev)
    msg = model.load_state_dict(ck['state'], strict=False)
    assert not msg.unexpected_keys, msg.unexpected_keys
    model.eval()
    return model, m, args


def read_split(data_dir, split):
    """CSI, image and location for one split, in the layout DataLoader.LoadBatch produces."""
    z = load_mat(data_dir / f'X_{split}_prev.mat', f'X_{split}_prev')
    csi = torch.from_numpy(np.stack([z.real, z.imag], axis=1)).float()
    img = torch.from_numpy(load_mat(data_dir / f'imgs_{split}.mat', f'imgs_{split}')).float()
    img = F.interpolate(img.permute(0, 3, 1, 2), size=(224, 224),
                        mode='bilinear', align_corners=False) / 255.0
    pos = torch.from_numpy(load_mat(data_dir / f'location_{split}.mat', f'location_{split}')).float()
    return csi, img, pos


@torch.no_grad()
def predict(model, meta, model_args, csi, img, pos, dev, bs=16):
    out = []
    for s in range(0, len(csi), bs):
        j = slice(s, s + bs)
        kw = dict(rgb=img[j].to(dev), pos=pos[j].to(dev)) if meta['use_modal'] else {}
        out.append(model(csi[j].to(dev), mask_ratio=meta['mask_ratio'], use_modal=meta['use_modal'],
                         model_args=model_args, **kw).cpu())
    return torch.cat(out)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ckpt', default=ROOT / 'experiments/task2_physics/physics_task2.pt', type=Path)
    p.add_argument('--split', default='test', help="'test' to predict, 'train' to sanity-check")
    p.add_argument('--out', default=ROOT / 'experiments/task2_physics/pred.npy', type=Path)
    p.add_argument('--submission', action='store_true', help='also write a submission json')
    p.add_argument('--base', default=ROOT / 'experiments/submission.json', type=Path,
                   help='supplies task1 (and task3 if non-empty), copied through untouched')
    p.add_argument('--sub-out', default=ROOT / 'experiments/submission_task2_physics.json', type=Path)
    a = p.parse_args()

    dev = device()
    model, meta, model_args = build(a.ckpt, dev)
    print(f'{a.ckpt}  val {meta["val_db"]:+.2f} dB  use_modal={meta["use_modal"]}  '
          f'alpha={model.alpha.item():+.4f}  on {dev}')

    csi, img, pos = read_split(ROOT / 'dataset/Task2', a.split)
    pred = predict(model, meta, model_args, csi, img, pos, dev)
    assert pred.shape == csi.shape, (pred.shape, csi.shape)
    assert torch.isfinite(pred).all()

    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(a.out, pred.numpy())
    print(f'{a.out}  {tuple(pred.shape)}')

    # NMSE is a ratio, so a uniformly shrunken prediction is punished even when the structure
    # is right. This catches that, and also catches a pure copy of the input.
    ratio = float((pred ** 2).mean() / (csi ** 2).mean())
    copy_db = float(10 * torch.log10(((pred - csi) ** 2).sum() / (csi ** 2).sum()))
    print(f'output/input power ratio {ratio:.4f}   prediction vs its own input {copy_db:+.2f} dB')

    if a.split == 'train':          # labels exist here, so the number is checkable
        z = load_mat(ROOT / 'dataset/Task2/X_train.mat', 'X_train')
        tgt = torch.from_numpy(np.stack([z.real, z.imag], axis=1)).float()
        for lo, name in ((32, 'masked-half'), (0, 'full-64')):
            d = 10 * torch.log10(((pred[..., lo:] - tgt[..., lo:]) ** 2).sum()
                                 / (tgt[..., lo:] ** 2).sum())
            print(f'  NMSE {name:12s} {float(d):+7.2f} dB')

    if a.submission:
        sub = json.loads(a.base.read_text())
        sub['task2'] = pred.numpy().tolist()
        if not sub.get('task3'):
            demo = json.loads((ROOT / 'dataset/demo_submission.json').read_text())
            sub['task3'] = demo['task3']
        # 15.71M base WiFo2 (flops_log.txt) + 0.147M LocalRidgeLayer (128*64*9 complex weights,
        # real+imag). Ridge forward is ~9 complex mult-adds over 8192 cells: ~5.7e5 flops/sample,
        # <0.02% of the 3.83 GFLOPs task-2 path, so flops is left as measured.
        sub.setdefault('param', 15.71 + 0.147)
        sub.setdefault('flops', 3.83)
        a.sub_out.write_text(json.dumps(sub))
        back = json.loads(a.sub_out.read_text())        # parse as the grader does
        assert back['task1'] == json.loads(a.base.read_text())['task1']
        assert np.array(back['task2']).size == 20 * 2 * 128 * 64
        print(f'\n{a.sub_out}  {a.sub_out.stat().st_size/1e6:.1f} MB')
        for k in sub:
            print(f'  {k}: {np.array(back[k]).size} values')


if __name__ == '__main__':
    main()
