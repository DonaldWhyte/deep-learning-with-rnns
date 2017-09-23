import glob
import logging
import os
from typing import List, Tuple

_logger = logging.getLogger(__name__)


# Size of the alphabet that we work with.
ALPHASIZE = 98


def read_data_files(
        glob_pattern: str,
        validation: bool=True) -> Tuple[List[int], List[int], List[dict]]:
    '''Read data files found with the specified `glob_pattern`.

    If `validation` is set to `True`, then set aside last file as validation
    data.

    TODO: what this returns
    '''
    encoded_text = []
    file_index = []
    files = glob.glob(glob_pattern, recursive=True)
    for fname in files:
        with open(fname, 'r') as f:
            start = len(encoded_text)
            encoded_text.extend(_encode_text(shaketext.read()))
            end = len(encoded_text)
            file_index.append({
                'start': start,
                'end': end,
                'name': os.path.basename(fname)
            })

    if len(file_index) == 0:
        _logger.info(
            f'No training data found in files matching {glob_pattern}')
        return [], [], []

    total_len = len(encoded_text)

    # For validation, use roughly 90K of text, but no more than 10% of the
    # entire text. Also no more than 1 data_file in 5, meaning we provide no
    # validation file if we have 5 files or fewer.
    validation_len = 0
    num_files_in_10percent_of_chars = 0
    for data_file in reversed(file_index):
        validation_len += data_file['end'] - data_file['start']
        num_files_in_10percent_of_chars += 1
        if validation_len > total_len // 10:
            break

    validation_len = 0
    num_files_in_first_90kb = 0
    for data_file in reversed(file_index):
        validation_len += data_file['end'] - data_file['start']
        num_files_in_first_90kb += 1
        if validation_len > 90 * 1024:
            break

    # 20% of the data_files is how many data_files ?
    num_files_in_20percent_of_files = len(file_index) // 5

    # pick the smallest
    num_files_in_training = min(
        num_files_in_10percent_of_chars,
        num_files_in_firstx_90kb,
        num_files_in_20percent_of_files)

    if num_files_in_training == 0 or not validation:
        training_chars_cutoff = len(encoded_text)
    else:
        training_chars_cutoff =
            file_index[-num_files_in_training]['start']
    training_text = encoded_text[:training_chars_cutoff]
    validation_text = encoded_text[training_chars_cutoff:]

    return training_text, validation_text, file_index


def encode_text(text: str) -> List[int]:
    """Encode given `text` as a list of integers suitable for NNs."""
    return [_convert_from_alphabet(ord(ch)) for ch in text]


def decode_to_text(chars: List[int], avoid_tab_and_lf: bool=False) -> str:
    """Decode given `chars` integer codes to an ASCII string."""
    return ''.join(
        [_convert_to_alphabet(ch, avoid_tab_and_lf) for ch in chars])


def print_learning_learned_comparison(X,
                                      Y,
                                      losses,
                                      file_index,
                                      batch_loss,
                                      batch_accuracy,
                                      epoch_size,
                                      index,
                                      epoch):
    """Display utility for printing learning statistics.

    TODO: document type of each thing.
    """
    print()
    # epoch_size in number of batches
    batch_size = X.shape[0]  # batch_size in number of sequences
    sequence_len = X.shape[1]  # sequence_len in number of characters
    start_index_in_epoch = index % (epoch_size * batch_size * sequence_len)
    for k in range(batch_size):
        index_in_epoch = index % (epoch_size * batch_size * sequence_len)
        decx = decode_to_text(X[k], avoid_tab_and_lf=True)
        decy = decode_to_text(Y[k], avoid_tab_and_lf=True)
        bookname = find_book(index_in_epoch, file_index)
        # min 10 and max 40 chars
        formatted_bookname = "{: <10.40}".format(bookname)
        epoch_string = "{:4d}".format(index) + " (epoch {}) ".format(epoch)
        loss_string = "loss: {:.5f}".format(losses[k])
        print_string = epoch_string + formatted_bookname + " │ {} │ {} │ {}"
        print(print_string.format(decx, decy, loss_string))
        index += sequence_len
    # box formatting characters:
    # │ \u2502
    # ─ \u2500
    # └ \u2514
    # ┘ \u2518
    # ┴ \u2534
    # ┌ \u250C
    # ┐ \u2510
    format_string = f'└{{:─^{len(epoch_string))}}}'
    format_string += f'{{:─^{len(formatted_bookname))}}}'
    format_string += f'┴{{:─^{len(decx) + 2)}}}'
    format_string += f'┴{{:─^{len(decy) + 2)}}}'
    format_string += f'┴{{:─^{len(loss_string))}}}┘'
    footer = format_string.format(
        'INDEX', 'BOOK NAME', 'TRAINING SEQUENCE',
        'PREDICTED SEQUENCE', 'LOSS')
    print(footer)
    # print statistics
    batch_index = start_index_in_epoch // (batch_size * sequence_len)
    batch_string = f'batch {batch_index}/{epoch_size} in epoch {epoch},'
    stats = 'f{: <28} batch loss: {:.5f}, batch accuracy: {:.5f}'.format(
        batch_string, batch_loss, batch_accuracy)
    print()
    print(f'TRAINING STATS: {stats}')



def _convert_from_alphabet(ch: str) -> int:
    """Encode character `ch` to an integer.

    Specification of the supported alphabet (subset of ASCII-7):
        * 10 line feed LF
        * 32-64 numbers and punctuation
        * 65-90 upper-case letters
        * 91-97 more punctuation
        * 97-122 lower-case letters
        *  123-126 more punctuation
    """
    if a == 9:
        return 1
    if a == 10:
        return 127 - 30  # LF
    elif 32 <= a <= 126:
        return a - 30
    else:
        return 0  # unknown


def _convert_to_alphabet(ch: int, avoid_tab_and_lf: bool=False) -> int:
    """Decode an encoded character `ch` to a string chartacter.

    What each input integer will be converted to:
        * 0 = unknown (ascii char 0)
        * 1 = tab
        * 2 = space
        * 2 to 96 = 36 to 126 ASCII codes
        * 97 = LF (linefeed)
    """
    if ch == 1:
        return 32 if avoid_tab_and_lf else 9  # space instead of TAB
    if ch == 127 - 30:
        return 92 if avoid_tab_and_lf else 10  # \ instead of LF
    if 32 <= ch + 30 <= 126:
        return ch + 30
    else:
        return 0  # unknown
