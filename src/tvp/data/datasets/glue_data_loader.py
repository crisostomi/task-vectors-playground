import os
import numpy as np
from torch.utils.data import Subset, Dataset
from datasets import load_dataset
import transformers
from open_clip.tokenizer import SimpleTokenizer
import numpy as np
import torch

glue_data_keys_map = {
    "cola": ("sentence", None),
    "sst2": ("sentence", None),
    "mrpc": ("sentence1", "sentence2"),
    "stsb": ("sentence1", "sentence2"),
    "qqp": ("question1", "question2"),
    "mnli": ("premise", "hypothesis"),
    "qnli": ("question", "sentence"),
    "rte": ("sentence1", "sentence2")
}

glue_data_metrics_map = {
    "cola": "matthews_correlation",
    "sst2": "accuracy",
    "mrpc": "averaged_scores",   # average of accuracy and f1
    "stsb": "averaged_scores",   # average of pearson and spearmanr
    "qqp": "averaged_scores",    # average of accuracy and f1
    "mnli": "accuracy",
    "qnli": "accuracy",
    "rte": "accuracy"
}

glue_data_num_labels_map = {
    "cola": 2,
    "sst2": 2,
    "mrpc": 2,
    "stsb": 1,
    "qqp": 2,
    "mnli": 3,
    "qnli": 2,
    "rte": 2
}

glue_data_id_map = {
    "cola": 0,
    "sst2": 1,
    "mrpc": 2,
    "stsb": 3,
    "qqp": 4,
    "mnli": 5,
    "qnli": 6,
    "rte": 7
}

rev_glue_data_id_map = {value: key for key, value in glue_data_id_map.items()}


class GLUEDataLoader:
    def __init__(self, tokenizer: transformers.AutoTokenizer):
        """
        Dataloader for GLUE datasets.
        :param tokenizer: AutoTokenizer, tokenizer
        :return:
        """
        self.tokenizer = tokenizer

    def load_dataset(
        self, 
        dataset_name: str, 
        train_split_ratio_for_val: float = 0.1, 
        max_seq_length: int = 128,
        cache_dir: str = "cache"
    ):
        """
        load GLUE dataset based on dataset_name
        :param dataset_name: str, name of the dataset to load
        :param train_split_ratio_for_val: float, split ratio of train data for validation,
        since the test data of GLUE is unavailable, we need to use a part of the original train data for validation (select the best model),
        and we use the original validation data for testing
        :param max_seq_length: int, maximal input length of examples in the dataset
        :param cache_dir: str, cache directory to store the dataset
        :return:
        """
        dataset = load_dataset(path="glue", name=dataset_name, cache_dir=cache_dir)

        # get the key of datasets
        sentence1_key, sentence2_key = glue_data_keys_map[dataset_name]

        if isinstance(self.tokenizer, SimpleTokenizer):
            dataset = dataset.map(
                lambda examples: openclip_preprocess(
                    examples=examples, tokenizer=self.tokenizer, sentence1_key=sentence1_key, sentence2_key=sentence2_key,
                    max_seq_length=77
                ),
                batched=True
            )
        else:
            dataset = dataset.map(
                lambda examples: self.tokenizer(
                    text=examples[sentence1_key],
                    text_pair=examples[sentence2_key] if sentence2_key else None,
                    max_length=max_seq_length, 
                    truncation=True
                ),
                batched=True
            )
        
        # add the "dataset_ids" column for each example
        dataset = dataset.map(lambda x: {"dataset_ids": glue_data_id_map[dataset_name]})

        dataset = dataset.select_columns(['input_ids', 'label', 'attention_mask'])
        dataset.set_format(type='torch', columns=['input_ids', 'label', 'attention_mask'])

        permuted_indices = np.random.RandomState(seed=0).permutation(len(dataset["train"])).tolist()
        num_train_data = int((1 - train_split_ratio_for_val) * len(dataset["train"]))
        train_dataset = Subset(dataset=dataset["train"], indices=permuted_indices[:num_train_data])
        # use a part of the original train data for validation
        val_dataset = Subset(dataset=dataset["train"], indices=permuted_indices[num_train_data:])
        test_dataset = dataset["validation_matched"] if dataset_name == "mnli" else dataset["validation"]
        num_labels = glue_data_num_labels_map[dataset_name]

        return train_dataset, val_dataset, test_dataset, num_labels

    def load_multitask_datasets(self, dataset_names: list, train_split_ratio_for_val: float = 0.1, max_seq_length: int = 128):
        """
        load GLUE datasets based on "dataset_names"
        :param dataset_names: list, name of the datasets to load
        :param train_split_ratio_for_val: float, split ratio of train data for validation,
        since the test data of GLUE is unavailable, we need to use a part of the original train data for validation (select the best model),
        and we use the original validation data for testing
        :param max_seq_length: int, maximal input length of examples in the dataset
        :return:
        """
        assert isinstance(dataset_names, list) and len(dataset_names) > 1, f"wrong setting on datasets {dataset_names}!"
        multiple_datasets = [self.load_dataset(dataset_name=dataset_name, train_split_ratio_for_val=train_split_ratio_for_val,
                                               max_seq_length=max_seq_length)
                             for dataset_name in dataset_names]
        train_datasets, val_datasets, test_datasets, datasets_num_labels = [dataset[0] for dataset in multiple_datasets], \
            [dataset[1] for dataset in multiple_datasets], [dataset[2] for dataset in multiple_datasets], [dataset[3] for dataset in multiple_datasets]

        multi_train_datasets = MultiDatasets(datasets=train_datasets)
        multi_val_datasets = MultiDatasets(datasets=val_datasets)
        multi_test_datasets = MultiDatasets(datasets=test_datasets)

        return multi_train_datasets, multi_val_datasets, multi_test_datasets, datasets_num_labels


class MultiDatasets(Dataset):
    def __init__(self, datasets: list):
        """
        MultiDatasets.
        :param datasets: list, list of datasets
        """
        super(MultiDatasets, self).__init__()
        self.datasets = datasets

    def __getitem__(self, index: int):
        """
        get item based on index
        :param index: int, data index
        :return:
        """
        # first find the corresponding dataset index, then get data based on the actual index
        dataset_index = 0
        assert 0 <= index < len(self), f"index {index} out of the length of data {len(self)}!"
        while index >= len(self.datasets[dataset_index]):
            index -= len(self.datasets[dataset_index])
            dataset_index += 1
        return self.datasets[dataset_index][index]

    def __len__(self):
        return sum([len(dataset) for dataset in self.datasets])


def openclip_preprocess(examples, tokenizer, sentence1_key, sentence2_key, max_seq_length=128):
    """
    examples: dict of lists (batch from HF 'map')
    tokenizer: instance of SimpleTokenizer()
    sentence1_key: str for the first text field (e.g. "sentence1")
    sentence2_key: str for the second text field if relevant (e.g. "sentence2")
    max_seq_length: int
    """
    # These will store the tokenized results for the entire batch
    all_input_ids = []
    all_attention_masks = []

    # If the dataset doesn’t have a second sentence, sentence2_key will be None
    texts1 = examples[sentence1_key]
    texts2 = examples[sentence2_key] if sentence2_key else [None]*len(texts1)

    for text1, text2 in zip(texts1, texts2):
        # Construct the actual text we want to tokenize
        # If text2 is not None, combine them in some way.
        # Here, we'll simply do something naive like "text1 [SEP] text2"
        # because SimpleTokenizer doesn't have a built-in `text_pair` concept
        if text2 is not None:
            combined_text = text1 + " [SEP] " + text2
        else:
            combined_text = text1

        # Encode (a list of token IDs)
        token_ids = tokenizer.encode(combined_text)

        # Insert special tokens if needed (start_of_text, end_of_text)
        # The open_clip.tokenizer.tokenize(...) function does this automatically,
        # but if you're calling tokenizer.encode(...) directly, you might do:
        #   token_ids = [sot_token] + token_ids + [eot_token]
        # For now, let's skip that or rely on the existing logic.

        # Truncate if longer than max_seq_length
        if len(token_ids) > max_seq_length:
            token_ids = token_ids[:max_seq_length]

        # Build attention mask
        attention_mask = [1]*len(token_ids)

        # Pad if shorter than max_seq_length
        pad_length = max_seq_length - len(token_ids)
        if pad_length > 0:
            token_ids = token_ids + [0]*pad_length
            attention_mask = attention_mask + [0]*pad_length

        all_input_ids.append(token_ids)
        all_attention_masks.append(attention_mask)

    return {
        "input_ids": all_input_ids,
        "attention_mask": all_attention_masks,
        # The original 'label' field is automatically preserved if the dataset has it
        # because we didn't drop it from 'examples'. If you want to pass it explicitly,
        # you could do:
        # "label": examples["label"]
    }

def clip_collate_fn(batch):
    # batch is a list of individual dataset items, each of which is already a dict
    # with 'input_ids', 'attention_mask', 'label', etc.
    # We just want to stack them into tensors.

    # Example if everything is in the same shape already:
    input_ids = torch.stack([example["input_ids"] for example in batch])
    attention_mask = torch.stack([example["attention_mask"] for example in batch])
    labels = torch.stack([example["label"] for example in batch])
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }