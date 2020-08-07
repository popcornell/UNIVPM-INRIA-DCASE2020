#!/bin/bash
# python TestModel.py -m "stored_data/multi_2412_with_synthetic/model/baseline_best" -g ../dataset/metadata/validation/validation.tsv  \
# -ga ../dataset/audio/validation -s stored_data/baseline/validation_predictions.tsv


# python TestModel.py -m "../baseline/weights/baseline_best_2422.p" -g ../dataset/metadata/validation/validation.tsv  \
# -ga ../dataset/audio/validation -s stored_data/baseline/validation_predictions.tsv

# model_path='stored_data/multi_2412_with_synthetic/model/baseline_best'
# metadata_path='../dataset/metadata/validation/validation.tsv'
# validation_path='../dataset/audio/validation'

# model_path='stored_data/multi_2412_with_synthetic/model/baseline_best'
# metadata_path='/srv/storage/talc@talc-data.nancy/multispeech/corpus/environmental_audio/DCASE2019/dataset/metadata/eval/vimeo.csv'
# validation_path='/srv/storage/talc@talc-data.nancy/multispeech/corpus/environmental_audio/DCASE2019/dataset/audio/eval/vimeo'


el_path='stored_data/multi_2412_with_synthetic/model/baseline_best'
metadata_path='/srv/storage/talc@talc-data.nancy/multispeech/corpus/environmental_audio/DCASE2019/dataset/metadata/eval/public.csv'
validation_path='/srv/storage/talc@talc-data.nancy/multispeech/corpus/environmental_audio/DCASE2019/dataset/audio/eval/public'

python TestModel.py -m ${model_path} -g ${metadata_path} -ga ${validation_path} -s stored_data/baseline/validation_predictions.tsv

