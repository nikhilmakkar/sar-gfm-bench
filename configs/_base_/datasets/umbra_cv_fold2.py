_base_ = ['./umbra_cv_fold0.py']
import os

fold = 2
data_root = os.path.abspath(os.environ.get(
    'UMBRA_CV_ROOT', 'datasets/Umbra/umbra_cv'))
fold_root = os.path.join(data_root, f'fold_{fold}')

train_dataloader = dict(dataset=dict(
    data_prefix=dict(img_path=os.path.join(fold_root, 'train/images')),
    ann_file=os.path.join(fold_root, 'train/annfiles') + os.sep))
val_dataloader = dict(dataset=dict(
    data_prefix=dict(img_path=os.path.join(fold_root, 'val/images')),
    ann_file=os.path.join(fold_root, 'val/annfiles') + os.sep))
test_dataloader = val_dataloader
