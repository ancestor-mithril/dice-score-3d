# dice-score-3d
Utility for calculating the Dice Similarity Coefficient (DSC) for 3D segmentation masks. Writes the results in a csv or json file and can be used both from the terminal or from a Python script.

## Installation

```
pip install --upgrade dice-score-3d
```

## Usage

Simple usage (Python):
```py
from dice_score_3d import dice_metrics


dice_metrics(gt_dir, pred_dir, output_path='results.csv',  indices={'lung': 1, 'heart': 2}, suffix='.nii.gz', num_workers=8)
```

Simple usage (terminal):
```
dice_score_3d GT.nii.gz PRED.nii.gz -output results.json -indices "{'lung': 1, 'heart': 2}" --console
```

Complete documentation:
```
usage: dice_score_3d [-h] -output OUTPUT -indices INDICES [--reorient] [-dtype {uint8,uint16}] [-prefix PREFIX] [-suffix SUFFIX] [-num_workers NUM_WORKERS] [--console]
                     ground_truths predictions

DICE Score 3D

positional arguments:
  ground_truths         Path to Ground Truth. Can be a single file or a folder with all the GT volumes. The number of GT files must match the number of predictions. When passing a     
                        folder of GT files, the name of the GT files must match the name of the predictions. This is not applicable when passing a single file. Supported file
                        formats: .nii, .nii.gz, .nrrd, .mha, .gipl.
  predictions           Path to Ground Truth. Can be a single file or a folder with all the predicted volumes. The number of prediction files must match the number of GT files. When   
                        passing a folder of prediction files, the name of the prediction files must match the name of the GT files. This is not applicable when passing a single file.  
                        Supported file formats: .nii, .nii.gz, .nrrd, .mha, .gipl.

options:
  -h, --help            show this help message and exit
  -output OUTPUT        The output path to write the computed metrics. Can be a csv or json file, depending on extension. Example: "results.csv", "results.json".
  -indices INDICES      Path to the json file describing the indices used for calculating the Dice Similarity Coefficient. Can also be a json string. Only the indices present in the   
                        json are considered when evaluating the Dice Score. Example: "{"lung_left": 1, "lung_right": 2}".
  --reorient            Reorients both the GT and the prediction to the default "LPS" orientation before calculating the Dice Score.
  -dtype {uint8,uint16}
                        Must be either "uint8" when having less than 255 classes, or "uint16" otherwise. Default: uint8.
  -prefix PREFIX        This parameter is used when the ground truth path is a folder. It filters all the files in the folder and selects only the files with this prefix.
  -suffix SUFFIX        This parameter is used when the ground truth path is a folder. It filters all the files in the folder and selects only the files with this suffix. Default: .nii.gz.
  -num_workers NUM_WORKERS
                        Number of parallel processes to be used to calculate the Dice Score in parallel. Default: 0.
  --console             Also prints the Dice metrics to console.
```
