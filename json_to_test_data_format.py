import sys
import json

if not len(sys.argv) == 2:
    sys.exit('Invalid command line arguments! Enter expected JSON file')

dump_re = open('/media/prabhanjan/New/NLP_proj/review_train.txt', 'w+')
dump_s = open('/media/prabhanjan/New/NLP_proj/summary_train.txt', 'w+')

cnt = 0
with open(sys.argv[1], 'r') as json_file:
            for line in json_file:
                if cnt<1000:
                    json_obj = json.loads(line)
                    review_text = json_obj['reviewText']
                    summary = json_obj['summary']
                    dump_re.write(review_text+"\n")
                    dump_s.write(summary+"\n")
                    cnt+=1
                else:
                    break



