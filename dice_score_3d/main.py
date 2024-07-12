import argparse
import json
import os.path

from dice_score_3d import dice_metrics


def main():
    parser = argparse.ArgumentParser(description='DICE Score 3D')
    parser.add_argument('ground_truths', type=str,
                        help='Path to Ground Truth. Can be a single file or a folder with all the GT volumes.'
                             'The number of GT files must match the number of predictions.'
                             'When passing a folder of GT files, the name of the GT files must match the name of the '
                             'predictions. This is not applicable when passing a single file. '
                             'Either SimpleITK or Nibabel must be able to read the files.')
    parser.add_argument('predictions', type=str,
                        help='Path to Ground Truth. Can be a single file or a folder with all the predicted volumes.'
                             'The number of prediction files must match the number of GT files.'
                             'When passing a folder of prediction files, the name of the prediction files must match '
                             'the name of the GT files. This is not applicable when passing a single file.'
                             'Either SimpleITK or Nibabel must be able to read the files.')
    parser.add_argument('-output', type=str, required=True,
                        help='The output path to write the csv or json file, depending on extension. '
                             'Example: results.csv, results.json')
    parser.add_argument('--reorient', action='store_true', default=False,
                        help='Reorients both the GT and the prediction to the default DICOM orientation before '
                             'calculating the Dice score.')
    parser.add_argument('-dtype', type=str, required=False, default='uint8', choices=['uint8', 'uint16'],
                        help='Dtype must be either uint8 when having less than 255 classes, and uint16 otherwise.')
    parser.add_argument('-prefix', type=str, required=False, default='',
                        help='This parameter is used when the ground truth path is a folder. '
                             'It filters all the files in the folders and selects only the files with this prefix.')
    parser.add_argument('-suffix', type=str, required=False, default='.nii.gz',
                        help='This parameter is used when the ground truth path is a folder. '
                             'It filters all the files in the folders and selects only the files with this suffix.')
    parser.add_argument('-indices', type=str, required=True,
                        help='Path to the json file describing the indices used for calculating the Dice Similarity '
                             'Coefficient. Can also be a json string. Only the indices present in the json as text are '
                             'considered when evaluating the Dice Score.'
                             'Example: "{\"lung_left\": 1, \"lung_right\": 2}".')
    parser.add_argument('-num_workers', type=int, required=False, default=0,
                        help='Number of parallel processes to be used to calculate the Dice score in parallel.')
    parser.add_argument('--console', action='store_true', default=False,
                        help='Also prints the Dice metrics to console.')
    args = parser.parse_args()
    if os.path.isfile(args.indices):
        with open(args.indices, 'r') as f:
            args.indices = json.load(f)
    else:
        args.indices = args.indices.replace('\'', '"')
        # TODO: Check json parse error
        args.indices = json.loads(args.indices)

    dice_metrics(args.ground_truths, args.predictions, args.output, args.reorient, args.dtype, args.prefix,
                 args.suffix, args.indices, args.num_workers, args.console)


if __name__ == '__main__':
    main()