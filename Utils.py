import numpy as np
import collections
from constants import *


def printD(x):
    if DEBUG:
        print(x)


def check_path(path):
    if not os.path.exists(path):
        print(f"Err: {path} doesn't exist!")
    else:
        print(f"{path} exists!")


def get_vocab(dataset):
    '''
    :param dataset: the text from load_data

    :return: a _ordered_ dictionary from words to counts
    '''
    vocab = {}

    # create a counter for each word
    for example in dataset:
        example_as_list = example.split()
        for word in example_as_list:
            vocab[word] = 0

    for example in dataset:
        example_as_list = example.split()
        for word in example_as_list:
            vocab[word] += 1

    # sort from greatest to least by count
    return collections.OrderedDict(sorted(vocab.items(), key=lambda x: x[1], reverse=True))


def text_to_rank(dataset, _vocab, desired_vocab_size=1000):
    '''
    :param dataset: the text from load_data
    :vocab: a _ordered_ dictionary of vocab words and counts from get_vocab
    :param desired_vocab_size: the desired vocabulary size
    words no longer in vocab become UUUNNNKKK
    :return: the text corpus with words mapped to their vocab rank,
    with all sufficiently infrequent words mapped to UUUNNNKKK; UUUNNNKKK has rank desired_vocab_size
    (the infrequent word cutoff is determined by desired_vocab size)
    '''
    _dataset = dataset[:]  # aliasing safeguard
    vocab_ordered = list(_vocab)
    count_cutoff = _vocab[vocab_ordered[desired_vocab_size - 2]]  # get word by its rank and map to its count

    word_to_rank = {}
    for i in range(len(vocab_ordered)):
        # we add one to make room for any future padding symbol with value 0
        word_to_rank[vocab_ordered[i]] = i

    for i in range(len(_dataset)):
        example = _dataset[i]
        example_as_list = example.split()
        for j in range(len(example_as_list)):
            try:
                if _vocab[example_as_list[j]] >= count_cutoff and word_to_rank[example_as_list[j]] < desired_vocab_size:
                    # we need to ensure that other words below the word on the edge of our desired_vocab size
                    # are not also on the count cutoff
                    example_as_list[j] = word_to_rank[example_as_list[j]]
                else:
                    example_as_list[j] = desired_vocab_size - 1  # UUUNNNKKK
            except:
                example_as_list[j] = desired_vocab_size - 1  # UUUNNNKKK
        _dataset[i] = example_as_list

    return _dataset


def text_to_matrix(dataset, _vocab, desired_vocab_size=1000):
    sequences = text_to_rank(dataset, _vocab, desired_vocab_size)

    mat = np.zeros((len(sequences), desired_vocab_size), dtype=int)

    for i, seq in enumerate(sequences):
        for token in seq:
            mat[i][token] = 1

    return mat


def partition_data_in_two(dataset, dataset_labels, in_sample_labels, oos_labels):
    '''
    :param dataset: the text from text_to_rank
    :param dataset_labels: dataset labels
    :param in_sample_labels: a list of newsgroups which the network will/did train on
    :param oos_labels: the complement of in_sample_labels; these newsgroups the network has never seen
    :return: the dataset partitioned into in_sample_examples, in_sample_labels,
    oos_examples, and oos_labels in that order
    '''
    _dataset = dataset[:]  # aliasing safeguard
    _dataset_labels = dataset_labels

    in_sample_idxs = np.zeros(np.shape(_dataset_labels), dtype=bool)
    ones_vec = np.ones(np.shape(_dataset_labels), dtype=int)
    for label in in_sample_labels:
        in_sample_idxs = np.logical_or(in_sample_idxs, _dataset_labels == label * ones_vec)

    return _dataset[in_sample_idxs], _dataset_labels[in_sample_idxs], \
           _dataset[np.logical_not(in_sample_idxs)], _dataset_labels[np.logical_not(in_sample_idxs)]


def relabel_in_sample_labels(labels):
    labels_as_list = labels.tolist()

    set_of_labels = []
    for label in labels_as_list:
        set_of_labels.append(label)
    labels_ordered = sorted(list(set(set_of_labels)))

    relabeled = np.zeros(labels.shape, dtype=int)
    for i in range(len(labels_as_list)):
        relabeled[i] = labels_ordered.index(labels_as_list[i])

    return relabeled


def init_folders():
    if not os.path.exists(OUTPUT_FOLDER_NAME):
        os.makedirs(OUTPUT_FOLDER_NAME)

    if not os.path.exists(FEATURES_DATA_FOLDER):
        os.makedirs(FEATURES_DATA_FOLDER)

    if not os.path.exists(OOD_FEATURES_DATA_FOLDER):
        os.makedirs(OOD_FEATURES_DATA_FOLDER)

    if not os.path.exists(DISTS_DATA_FOLDER):
        os.makedirs(DISTS_DATA_FOLDER)

    if not os.path.exists(OOD_DISTS_DATA_FOLDER):
        os.makedirs(OOD_DISTS_DATA_FOLDER)

    if not os.path.exists(CLOSEST_CLASS_DATA_FOLDER):
        os.makedirs(CLOSEST_CLASS_DATA_FOLDER)

    if not os.path.exists(OOD_CLOSEST_CLASS_DATA_FOLDER):
        os.makedirs(OOD_CLOSEST_CLASS_DATA_FOLDER)

    if not os.path.exists(LABELS_DATA_FOLDER):
        os.makedirs(LABELS_DATA_FOLDER)

    if not os.path.exists(MEANS_DATA_FOLDER):
        os.makedirs(MEANS_DATA_FOLDER)

    if not os.path.exists(RADIUS_DATA_FOLDER):
        os.makedirs(RADIUS_DATA_FOLDER)

    printD(f"Initialized folders at {OUTPUT_FOLDER_NAME}")


def check_all_paths():
    check_path(OUTPUT_FOLDER_NAME)
    check_path(FEATURES_DATA_FOLDER)
    check_path(OOD_FEATURES_DATA_FOLDER)
    check_path(DISTS_DATA_FOLDER)
    check_path(OOD_DISTS_DATA_FOLDER)
    check_path(CLOSEST_CLASS_DATA_FOLDER)
    check_path(OOD_CLOSEST_CLASS_DATA_FOLDER)
    check_path(LABELS_DATA_FOLDER)
    check_path(MEANS_DATA_FOLDER)
    check_path(RADIUS_DATA_FOLDER)


def get_class_info(in_sample_labels):
    classes = list(set(in_sample_labels))
    NUM_CLASSES = len(classes)
    printD(f"classes = {classes}")
    printD(f"NUM_CLASSES = {NUM_CLASSES}")
    return classes, NUM_CLASSES


def get_per_class_info(in_sample_examples, in_sample_labels, classes):
    total = 0
    per_class_examples = []
    for classID in classes:
        in_samples_class = in_sample_examples[in_sample_labels == classID]
        ss = len(in_samples_class)
        total += ss
        per_class_examples.append(ss)
    printD(f"per_class_examples = {per_class_examples}")
    printD(f"total == len(in_sample_examples)  == ({total == len(in_sample_examples)})")
    return per_class_examples


# taken from keras
def pad_sequences(sequences, maxlen=None, dtype='int32',
                  padding='pre', truncating='pre', value=0.):
    '''Pads each sequence to the same length:
    the length of the longest sequence.
    If maxlen is provided, any sequence longer
    than maxlen is truncated to maxlen.
    Truncation happens off either the beginning (default) or
    the end of the sequence.
    Supports post-padding and pre-padding (default).
    # Arguments
        sequences: list of lists where each element is a sequence
        maxlen: int, maximum length
        dtype: type to cast the resulting sequence.
        padding: 'pre' or 'post', pad either before or after each sequence.
        truncating: 'pre' or 'post', remove values from sequences larger than
            maxlen either in the beginning or in the end of the sequence
        value: float, value to pad the sequences to the desired value.
    # Returns
        x: numpy array with dimensions (number_of_sequences, maxlen)
    '''
    lengths = [len(s) for s in sequences]

    nb_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)

    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break

    x = (np.ones((nb_samples, maxlen) + sample_shape) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if len(s) == 0:
            continue  # empty list was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" not understood' % truncating)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x
