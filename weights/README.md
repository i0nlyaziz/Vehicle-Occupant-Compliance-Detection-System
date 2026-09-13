# Model Weights

The custom-trained weights aren't committed to this repo directly — they're excluded in `.gitignore` since `.pt` files are large.

## Download

| Model | File | Download |
|-------|------|----------|
| Vehicle | `vehicle.pt` | [Download from Releases](https://github.com/i0nlyaziz/Vehicle-Occupant-Compliance-Detection-System/releases/download/v1.0/vehicle.pt) |
| Seatbelt | `seatbelt.pt` | [Download from Releases](https://github.com/i0nlyaziz/Vehicle-Occupant-Compliance-Detection-System/releases/download/v1.0/seatbelt.pt) |
| Mobile phone | `mobile.pt` | [Download from Releases](https://github.com/i0nlyaziz/Vehicle-Occupant-Compliance-Detection-System/releases/download/v1.0/mobile.pt) |
| License plate | `plate.pt` | [Download from Releases](https://github.com/i0nlyaziz/Vehicle-Occupant-Compliance-Detection-System/releases/download/v1.0/plate.pt) |

`vehicle.pt` is the stock YOLO11n COCO model, saved locally under this name since `main.py` loads it by that filename — it's included here for convenience so the exact file used doesn't need to be renamed manually after downloading.

## Using the weights

1. Download all four files above and place them in this `weights/` folder.
2. `main.py` loads each `.pt` file directly — no export step is required.
3. Update the model paths in `main.py` if you place the files somewhere other than `weights/`.