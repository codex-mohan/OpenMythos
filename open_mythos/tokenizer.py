from __future__ import annotations

from typing import Optional

from transformers import AutoTokenizer, PreTrainedTokenizerFast

DEFAULT_MODEL_ID: str = "openai/gpt-oss-20b"


def get_vocab_size(model_id: str) -> int:
    """
    Resolve vocabulary size for a HuggingFace model ID.

    Loads the tokenizer (downloads only small json/txt files — no model weights,
    typically < 5 MB) and returns its actual vocabulary size.  Results are
    cached by HuggingFace so repeated calls are instant.

    Args:
        model_id: HuggingFace model identifier or local tokenizer path.

    Returns:
        Vocabulary size as an integer.
    """
    return AutoTokenizer.from_pretrained(model_id).vocab_size


# ---------------------------------------------------------------------------
# Custom BPE training (for training a tokenizer from scratch)
# ---------------------------------------------------------------------------


def train_bpe_tokenizer(
    text_iterator: iter,
    vocab_size: int = 50000,
    output_dir: str = "mythos_tokenizer",
    model_id: str = "openai/gpt-oss-20b",
) -> PreTrainedTokenizerFast:
    """
    Train a BPE tokenizer from a text corpus and save it to disk.

    Wraps HuggingFace ``tokenizers`` to build a Byte-Pair-Encoding tokenizer
    matching the pre-tokenization and normalisation of an existing tokenizer
    (``model_id``), but with a custom vocabulary size trained on your own data.
    The result is saved as a HuggingFace-compatible tokenizer that works
    transparently with ``MythosTokenizer``.

    Args:
        text_iterator: An iterable yielding raw text strings (e.g. a generator
                       streaming lines from a file or dataset).
        vocab_size:    Target vocabulary size (default 50 000).
        output_dir:    Directory where tokenizer files will be written.
        model_id:      Base tokenizer whose pre-tokenizer and normaliser are
                       cloned.  Defaults to ``openai/gpt-oss-20b`` which uses
                       GPT-2 style byte-level BPE with no unicode normalisation.

    Returns:
        A ``PreTrainedTokenizerFast`` ready to be wrapped in ``MythosTokenizer``.
        The tokenizer and its files are also saved to ``output_dir`` so that
        subsequent runs can load them with ``MythosTokenizer(output_dir)``.

    Example:
        >>> from datasets import load_dataset
        >>> ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT",
        ...                   split="train", streaming=True)
        >>> texts = (x["text"] for x in ds.take(100_000))
        >>> train_bpe_tokenizer(texts, vocab_size=50000)
    """
    from tokenizers import (
        Tokenizer,
        models,
        normalizers,
        pre_tokenizers,
        decoders,
        trainers,
    )
    from transformers import PreTrainedTokenizerFast
    import os

    base_tok = AutoTokenizer.from_pretrained(model_id)
    if hasattr(base_tok, "backend_tokenizer"):
        backend = base_tok.backend_tokenizer
    else:
        backend = base_tok._tokenizer if hasattr(base_tok, "_tokenizer") else None

    tokenizer = Tokenizer(models.BPE(unk_token=base_tok.unk_token or "<|endoftext|>"))

    if backend is not None:
        tokenizer.normalizer = backend.normalizer
        tokenizer.pre_tokenizer = backend.pre_tokenizer
        tokenizer.decoder = backend.decoder
    else:
        tokenizer.normalizer = normalizers.NFKC()
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=(
            base_tok.all_special_tokens
            if base_tok.all_special_tokens
            else ["<|endoftext|>", "<|pad|>"]
        ),
        show_progress=True,
    )
    tokenizer.train_from_iterator(text_iterator, trainer)

    os.makedirs(output_dir, exist_ok=True)
    tokenizer.save(os.path.join(output_dir, "tokenizer.json"))

    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token=base_tok.unk_token,
        pad_token=base_tok.pad_token,
        bos_token=base_tok.bos_token,
        eos_token=base_tok.eos_token,
    )
    fast.save_pretrained(output_dir)
    return fast


# ---------------------------------------------------------------------------
# MythosTokenizer — thin wrapper with zero-copy encode/decode
# ---------------------------------------------------------------------------


class MythosTokenizer:
    """
    Tokenizer wrapper for OpenMythos.

    Supports any HuggingFace tokenizer (pre-trained or custom-trained via
    ``train_bpe_tokenizer``).  Use ``get_vocab_size(model_id)`` to discover
    the vocabulary size of a known model without downloading the tokenizer,
    or instantiate ``MythosTokenizer(model_id)`` to load it.

    Args:
        model_id: A HuggingFace model ID, a local directory path, or a
                  ``PreTrainedTokenizerFast`` instance.  Defaults to
                  ``openai/gpt-oss-20b`` (vocab_size=200064).
    """

    def __init__(self, model_id: str | PreTrainedTokenizerFast = DEFAULT_MODEL_ID):
        if isinstance(model_id, PreTrainedTokenizerFast):
            self.tokenizer = model_id
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(str(model_id))

    @property
    def vocab_size(self) -> int:
        return int(self.tokenizer.vocab_size)

    @property
    def model_id(self) -> str:
        return getattr(self.tokenizer, "name_or_path", "custom")

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)  # type: ignore[return-value]

    def decode(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)  # type: ignore[return-value]

    def __repr__(self) -> str:
        return f"MythosTokenizer(model_id={self.model_id!r}, vocab_size={self.vocab_size:,})"


# ---------------------------------------------------------------------------
# Convenience loader (kept for backward compatibility + ergonomics)
# ---------------------------------------------------------------------------


def load_tokenizer(model_id: str | None = None) -> MythosTokenizer:
    return MythosTokenizer(model_id=model_id or DEFAULT_MODEL_ID)
