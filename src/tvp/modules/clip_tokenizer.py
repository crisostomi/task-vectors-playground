from transformers import PreTrainedTokenizer
from open_clip.tokenizer import SimpleTokenizer

class CLIPLikeTokenizer(PreTrainedTokenizer):
    def __init__(self, max_len=128):
        self._simple_tokenizer = SimpleTokenizer()
        super().__init__()
        self.model_max_length = max_len
        # define special tokens, etc. if needed

    def __call__(self, text, text_pair=None, max_length=None, truncation=True, **kwargs):
        """
        This method is called by HF pipelines & DataCollatorWithPadding.
        You must return a dict with at least:
            {
                "input_ids": [...],
                "attention_mask": [...],
            }
        Optionally "token_type_ids", etc.
        """
        # Handle single or batch input
        if isinstance(text, str):
            text = [text]
        if text_pair is not None and isinstance(text_pair, str):
            text_pair = [text_pair]

        all_input_ids = []
        all_attention_masks = []

        for i, t in enumerate(text):
            # If we have a text_pair, combine it
            t2 = text_pair[i] if text_pair else None
            combined_text = t + " [SEP] " + t2 if t2 else t

            token_ids = self._simple_tokenizer.encode(combined_text)
            # truncate if too long
            if len(token_ids) > self.model_max_length:
                token_ids = token_ids[: self.model_max_length]
            attention_mask = [1]*len(token_ids)

            # pad if needed
            if len(token_ids) < self.model_max_length:
                pad_len = self.model_max_length - len(token_ids)
                token_ids += [0]*pad_len
                attention_mask += [0]*pad_len

            all_input_ids.append(token_ids)
            all_attention_masks.append(attention_mask)

        return {
            "input_ids": all_input_ids, 
            "attention_mask": all_attention_masks
        }

    def get_vocab(self):
        """
        Must return a dict: token_string -> token_id
        for the current vocabulary. 
        """
        # `self._simple_tokenizer.encoder` is typically dict[str, int]
        return dict(self._simple_tokenizer.encoder)  # copy or return directly

    @property
    def vocab_size(self):
        """Return the size of the vocabulary."""
        return len(self._simple_tokenizer.encoder)
