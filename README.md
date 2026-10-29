# ASNet

This is the repo for [Attention scaling for crowd counting](http://openaccess.thecvf.com/content_CVPR_2020/html/Jiang_Attention_Scaling_for_Crowd_Counting_CVPR_2020_paper.html) in CVPR 2020, which proposed a simple but effective architecture for crowd counting.

## Requirements

In an Anaconda environment, we need the following packages.

Python: 3.7.5

PyTorch: 1.3.1

## 使用 SHA/mix_level 訓練

新增的 `train.py` 使用現代 PyTorch（2.1 以上）。請在你選定的 Python 環境中安裝
`requirements-train.txt`；GPU 版 PyTorch 請使用與你驅動相容的版本。
訓練入口只需要 PyTorch、NumPy、Pillow、h5py，不需要原測試程式的 OpenCV、pandas 或 Excel 套件。
資料載入預設 `--workers 0`，也能在限制多程序通訊的環境執行；一般終端可用
`--workers 4` 增加載入程序。

資料目錄：

```text
SHA/mix_level/
  train/
    clean_IMG_1.jpg
    clean_IMG_1.npy
    hazy_IMG_1_light.jpg
    hazy_IMG_1_light.npy
    ...
  val/
    clean_IMG_1.jpg
    clean_IMG_1.npy
    ...
```

每個 `.npy` 必須是 `N×2` 或 `N×3` 的人頭座標陣列。只讀前兩欄 `(x, y)`，
第三欄不參與訓練；不接受已生成的密度圖。影像採 RGB、像素值除以 255，
沿用原本 preload loader 的輸入方式。每個人頭生成固定 `15×15, sigma=4` Gaussian，
邊界處重新正規化；略微越界的點會裁回影像邊界，重複座標分別計數。
先生成整張密度圖，再同步裁切、水平翻轉；降採樣使用區塊加總以保留人數。

在專案根目錄執行：

```bash
python train.py --data-root SHA/mix_level --output runs/asnet_mix_level --device cuda
```

預設先訓練 prior 50 epochs，再載入驗證 cross entropy 最低的 prior，固定它並訓練
ASNet 300 epochs。兩階段都使用 Adam、learning rate `1e-5`，每張影像每 epoch
隨機取 9 個 patches，並做水平翻轉和亮度、對比、飽和度增強。
prior 使用 `256×256`、batch size 1；ASNet 使用 `128×128`、batch size 8。
這些是本訓練器的起始設定；prior 的固定尺寸裁切等細節與論文流程有差異，
不保證重現原論文數值。可透過 `--prior-epochs`、`--epochs`、`--prior-lr`、`--lr` 調整。

預設從 ImageNet VGG16 初始化兩分支的前 13 個卷積層。優先使用本機 Torch cache；
沒有快取時會下載至輸出目錄的 `pretrained/`。離線時可用
`--vgg-weights /path/to/vgg16-397923af.pth` 指定 torchvision 格式的權重。
`--pretrained none` 僅適合測試或明確要研究隨機初始化的實驗。

注意力分類門檻與 APLoss 門檻預設由訓練集隨機取 64 張完整影像估算，
不使用驗證集；可用 `--threshold-samples 1200` 計算整個 mix_level 訓練集。
驗證預設逐張使用完整解析度，padding 到 16 的倍數，以完整標註人數計算 MAE、RMSE。
`--max-val-size 1024` 可限制驗證影像最長邊，但會改變評估解析度。

輸出檔案：

- `last.pt`：最近完成的 epoch，包含階段、模型、optimizer、密度門檻和亂數狀態。
- `prior_best.pt`：驗證 cross entropy 最低的 prior checkpoint。
- `best.pt`、`best.h5`：ASNet 驗證 MAE 最低的 checkpoint 與模型權重。
- `metrics.jsonl`、`config.json`：每個 epoch 的指標和執行設定。

`.h5` 沿用原專案的權重格式。原 `test.py` 的資料路徑與 CSV loader 仍需要另行設定，
不能直接讀取 SHA 的座標 `.npy`；訓練期間的驗證已由新入口處理。

續訓（`--epochs` 表示目標總 epochs，不是額外訓練輪數）：

```bash
python train.py --data-root SHA/mix_level --resume runs/asnet_mix_level/last.pt --device cuda
```

續訓會還原 optimizer 和資料生成設定；batch size、workers、device 可重新指定。
中斷時從最近完成的 epoch 接續，尚未完成的 epoch 會重跑。

分開跑兩階段，或從完整模型微調：

```bash
python train.py --stage prior --data-root SHA/mix_level --output runs/sha_prior --device cuda
python train.py --stage asnet --data-root SHA/mix_level --prior-checkpoint runs/sha_prior/prior_best.pt --output runs/sha_asnet --device cuda
python train.py --stage asnet --data-root SHA/mix_level --init-checkpoint /path/to/shtechA.h5 --output runs/sha_finetune --device cuda
```

先測試少量批次（結果只用來確認流程，不代表模型訓練成效）：

```bash
python train.py --data-root SHA/mix_level --output runs/sha_smoke --device cuda --workers 0 --prior-epochs 1 --epochs 1 --batch-size 1 --patches-per-image 1 --threshold-samples 2 --max-val-size 256 --max-train-batches 2 --max-val-batches 2
```

若 prior 尚未學會前景，計數階段會停止並提示先延長 prior 訓練。顯存不足時可降低
`--batch-size` 或 `--prior-crop-size`；ASNet `--crop-size` 須為 64 的倍數以符合原 APLoss。
用 `--device cpu` 可在 CPU 測試。完整參數見 `python train.py --help`。
檢查資料生成、計數 loss 梯度與 checkpoint：`python -m unittest discover -s tests -v`。

## Results

For ShanghaiTech Part A dataset

Model: [Link](https://pan.baidu.com/s/1jQgBsDy90UfzlLafXgTcXQ), Password: 585s

## References

If you find the ASNet useful, please cite our paper. Thank you!

```
@inproceedings{jiang2020attention,
  title={Attention scaling for crowd counting},
  author={Jiang, Xiaoheng and Zhang, Li and Xu, Mingliang and Zhang, Tianzhu and Lv, Pei and Zhou, Bing and Yang, Xin and Pang, Yanwei},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={4706--4715},
  month={June},
  year={2020}
}
```
