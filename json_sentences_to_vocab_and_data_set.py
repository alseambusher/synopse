import sys
import json
import os

if not len(sys.argv) == 2:
    sys.exit('Invalid command line arguments! Enter expected JSON file directory')
if not os.path.isdir(sys.argv[1]) or not os.listdir(sys.argv[1]):
    sys.exit('Invalid directory!')

dump_vocab = open('vocab.txt', 'w+')

vocab = dict()

for file in os.listdir(sys.argv[1]):
    with open(os.path.join(sys.argv[1],file), 'r') as json_file:
            for line in json_file:
                try:
                    json_obj = json.loads(line)
                except:
                    print(file + "line\n" + line)
                review_text = json_obj['reviewText']
                summary = json_obj['summary']
                for word in review_text.split():
                    if word not in vocab:
                        vocab[word] = 1
                    else:
                        vocab[word] += 1
                for word in summary.split():
                    if word not in vocab:
                        vocab[word] = 1
                    else:
                        vocab[word] += 1


sorted_vocab_list = ["_PAD","_GO","_EOS","_UNK"] + sorted(vocab, key=vocab.get, reverse=True)

vocab_desired = dict()
for i in range(len(sorted_vocab_list)):
    vocab_desired[sorted_vocab_list[i]] = i
    dump_vocab.write(sorted_vocab_list[i]+"\n")

print("vocab size = " + str(len(vocab_desired)))

for file in os.listdir(sys.argv[1]):
    basename = os.path.basename((os.path.join(sys.argv[1], file)))
    dump_x = open(os.path.splitext(basename)[0] + '_x.txt', 'w+')
    dump_y = open(os.path.splitext(basename)[0] + '_y.txt', 'w+')
    if file.endswith('.json'):
        with open(os.path.join(sys.argv[1],file), 'r') as json_file:
            for line in json_file:
                json_obj = json.loads(line)
                review_text = json_obj['reviewText']
                summary = json_obj['summary']
                review_text_line = []
                summary_line = []
                for word in review_text.split():
                    if word in vocab_desired:
                        review_text_line.append(vocab_desired[word])
                for word in summary.split():
                    if word in vocab_desired:
                        summary_line.append(vocab_desired[word])
                dump_x.write(" ".join([str(id) for id in review_text_line])+"\n")
                dump_y.write(" ".join([str(id) for id in summary_line])+"\n")
