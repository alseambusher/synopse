import sys
import os
import json

if not len(sys.argv) == 2:
    sys.exit('Invalid command line arguments!')
if not os.path.isdir(sys.argv[1]) or not os.listdir(sys.argv[1]):
    sys.exit('Invalid directory!')

MIN_SUMMARY_LEN = 32
TOTAL_MAX_CNT_PER_CATEGORY = 170000
TRAIN_MAX_CNT_PER_CATEGORY = 140000
VALIDATION_MAX_CNT_PER_CATEGORY = 20000
TEST_MAX_CNT_PER_CATEGORY = TOTAL_MAX_CNT_PER_CATEGORY - TRAIN_MAX_CNT_PER_CATEGORY - VALIDATION_MAX_CNT_PER_CATEGORY
OUTPUT_DIR = '/media/prabhanjan/New/Downloads/'

invalid_chars = r"<>&#\\"
chars_to_remove = set([':',';','\'','\"','(','[',']',')','{','}','|','*','/','^','%','@','!','~','.','-',',','+','$','=','_','?','`'])


train_file = open(OUTPUT_DIR + 'train.json', 'w+')
validation_file = open(OUTPUT_DIR + 'validation.json', 'w+')
test_file = open(OUTPUT_DIR + 'test.json', 'w+')

for file in os.listdir(sys.argv[1]):
    with open(os.path.join(sys.argv[1],file), 'r') as json_file:
        jsons_to_select = TOTAL_MAX_CNT_PER_CATEGORY
        for line in json_file:
            if jsons_to_select == 0:
                break
            json_obj = json.loads(line)
            review_text = json_obj['reviewText']
            summary = json_obj['summary']
            flag = any(elem in review_text for elem in invalid_chars) or any(elem in summary for elem in invalid_chars) or len(summary) <= MIN_SUMMARY_LEN or len(review_text) < len(summary) or len(review_text) == 0
            if not flag:
                jsons_to_select -= 1
                temp = ""
                for ch in review_text:
                    if ch not in chars_to_remove and not ch.isdigit():
                        temp += ch
                    elif ch == "'" or ch == '\"':
                        continue
                    else:
                        temp += ' '
                review_text = temp.lower()
                temp = ""
                for ch in summary:
                    if ch not in chars_to_remove and not ch.isdigit():
                        temp += ch
                    elif ch == "'" or ch == '\"':
                        continue
                    else:
                        temp += ' '
                summary = temp.lower()
                if jsons_to_select < TEST_MAX_CNT_PER_CATEGORY:
                    test_file.write(
                        "{" + "\"reviewText\": \"" + review_text + "\", " + "\"summary\": \"" + summary + "\"}\n")
                elif jsons_to_select < TEST_MAX_CNT_PER_CATEGORY + VALIDATION_MAX_CNT_PER_CATEGORY:
                    validation_file.write(
                        "{" + "\"reviewText\": \"" + review_text + "\", " + "\"summary\": \"" + summary + "\"}\n")
                else:
                    train_file.write("{" + "\"reviewText\": \"" + review_text + "\", " + "\"summary\": \"" + summary + "\"}\n")
