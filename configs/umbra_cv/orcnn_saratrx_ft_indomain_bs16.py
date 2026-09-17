# Finetuned SARATR-X in-domain at bs16 (bs32 from the shared in-domain
# dataset block OOMs a 41.8GB finetuned run; bs16 matches its CV folds).
_base_ = ['./orcnn_saratrx_ft_indomain.py']

train_dataloader = dict(batch_size=16)
