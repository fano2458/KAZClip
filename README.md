# KAZClip

## Overview
KAZClip is an image-text matching model inspired by OpenAI's CLIP, tailored for Kazakh-language data. It uses transformer-based encoders for both text and images, then aligns these representations through a shared latent space. This model has been trained using [this dataset](http://images.cocodataset.org/zips/train2017.zip) which has been translated to Kazakh Language.

## Examples
| Caption                   | Caption                    |
| ------------------------- | -------------------------- |
| Kөшеде келе жатқан адам.  | Kітапті оқитын бала.       |
| <img src="examples/000000021839.jpg" width="300" height="300"/> | <img src="examples/000000563281.jpg" width="300" height="300"/> |

## Requirements
- Python 3.8+
- PyTorch
- Transformers (HuggingFace)
- Pillow
- Requests
- TQDM
- wandb (optional, for experiment tracking)

## Installation
1. Clone this repository.  
2. Install the dependencies.

## How to Train
1. Prepare the necessary data in the "data" directory (train2017 and val2017 folders plus corresponding JSON files).  
2. Run the train.py script:
   ```
   python train.py
   ```
3. The best model weights will be saved as best_model.pt in the current directory.

## How to Use
1. Compute image embeddings for your dataset:
   ```
   python compute_image_embeddings.py
   ```
2. Run predict.py to query your images:
   ```
   python predict.py
   ```
   Provide text queries (captions) and the script returns top matches.
