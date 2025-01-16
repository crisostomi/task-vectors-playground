from tvp.data.datasets.glue_data_loader import GLUEDataLoader
from transformers import AutoTokenizer, BioGptTokenizer
from open_clip.tokenizer import SimpleTokenizer as ClipTokenizer

from torch.utils.data import DataLoader

from transformers.data.data_collator import DataCollatorWithPadding
from tvp.data.datasets.glue_data_loader import clip_collate_fn


DATASET_NAME = "qqp"

class QQP:
    def __init__(
        self, 
        tokenizer: AutoTokenizer,
        train_split_ratio_for_val: float = 0.1,
        max_seq_length: int = 128,
        batch_size: int = 128,
        num_workers: int = 16
    ):
        glue_data_loader = GLUEDataLoader(tokenizer=tokenizer)

        if isinstance(tokenizer, ClipTokenizer):
            collate_fn = clip_collate_fn
        else:
            collate_fn = DataCollatorWithPadding(
                tokenizer=tokenizer,
                padding="longest",
                max_length=max_seq_length,
                return_tensors="pt",
                collate_fn=collate_fn
        )

        train_dataset, val_dataset, test_dataset, num_labels = glue_data_loader.load_dataset(
            dataset_name=DATASET_NAME,
            train_split_ratio_for_val=train_split_ratio_for_val,
            max_seq_length=max_seq_length
        )

        self.train_dataset = train_dataset
        
        self.train_loader = DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=collate_fn
        )
        
        self.val_dataset = val_dataset
        
        self.val_loader = DataLoader(
            self.val_dataset,
            shuffle=True,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=collate_fn
        )


        self.test_dataset = test_dataset

        self.test_loader = DataLoader(
            self.test_dataset, 
            shuffle=False,
            batch_size=batch_size, 
            num_workers=num_workers,
            collate_fn=collate_fn
        )