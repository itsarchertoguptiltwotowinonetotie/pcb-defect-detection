# Reference-Guided Contextual Defect Detection and Localization for Microscopic Printed Circuit Board Images

Detects and localizes six PCB defect types (open, short, mousebite, spur, spurious copper, pin-hole) by comparing each test image with its defect-free template. Blurred images are first restored and binarized by a deblur-and-binarize U-Net.

## Folders

| Folder | Contents |
|---|---|
| `detector/` | Reference-guided Faster R-CNN (ResNet-50-FPN, template comparison, NNVR) trained on DeepPCB |
| `deblur_binarize/` | Deblur-and-binarize module trained with the detector, and with the blur-trained detector |
| `pku_market_pcb/` | Fine-tuning and testing on PKU-Market-PCB photographs |
| `baselines/` | Ablation and comparison runs |
| `evaluation/` | Confidence threshold tuning for false alarms |
| `demo/` | Side-by-side inspection demo |
| `webapp/` | Streamlit web application |

Each experiment is available as a Python script and as a notebook with the same name.

## Datasets

- DeepPCB: https://github.com/tangsanli5201/DeepPCB
- PKU-Market-PCB: https://github.com/Ironbrotherstyle/PCB-DATASET

The notebooks download both datasets automatically.

## Running

The notebooks were run on Google Colab with a T4 GPU. Open a notebook in Colab, select a GPU runtime and run all cells. Checkpoints and results are saved to `MyDrive/PCB_Defect_Detection`.

Run order: `detector`, `deblur_binarize/deblur_binarize_training`, `deblur_binarize/deblur_binarize_with_blur_trained_detector`, `pku_market_pcb`, then `baselines`, `evaluation` and `demo`.

Web application:

```
cd webapp
pip install -r requirements.txt
streamlit run streamlit_app.py
```
