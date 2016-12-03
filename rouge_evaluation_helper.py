import sys

# script to convert prediction file to rouge tool expected format.
# if you run rnn_summarize.py --decode <vtest_file.txt > prediction_file.txt, then the second arg to this simple script
#  would be prediction_file.txt
if not len(sys.argv) == 3:
    sys.exit('Invalid command line arguments! Enter summary prediction file and output file name')
dump_file = open(sys.argv[2], 'w+')
with open(sys.argv[1],'r') as file:
    for i,line in enumerate(file.readlines()):
        if i==0:
            continue
        else:
            dump_file.write(line[2:])