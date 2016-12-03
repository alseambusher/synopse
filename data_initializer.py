import os

EOS_VAL_IN_VOCAB = 2

def initialize_data(review_path, summary_path, buckets, vocabulary_path = None):
    reversed_vocab = []
    vocab_list = []
    vocab = dict()
    if vocabulary_path is not None:
        if os.path.exists(vocabulary_path):
            with open(vocabulary_path, mode="r") as f:
                reversed_vocab.extend(f.readlines())
                reversed_vocab = [line.strip() for line in reversed_vocab]
                for (y, x) in enumerate(reversed_vocab):
                    vocab_list.append((x,y))
                vocab = dict(vocab_list)
        else:
            raise ValueError("Something went wrong with the vocab path", vocabulary_path)
    data_set = [[] for _ in buckets]
    with open(review_path, mode="r") as s_file:
        with open(summary_path, mode="r") as t_file:
            review = s_file.readline()
            summary = t_file.readline()
            while review and summary:
                review_ids = [int(x) for x in review.split()]
                summary_ids = [int(x) for x in summary.split()]
                summary_ids.append(EOS_VAL_IN_VOCAB)
                for bucket_id, (review_size, summary_size) in enumerate(buckets):
                    if len(review_ids) < review_size and len(summary_ids) < summary_size:
                        data_set[bucket_id].append([review_ids, summary_ids])
                        break
                review, summary = s_file.readline(), t_file.readline()
    return data_set, vocab, reversed_vocab
