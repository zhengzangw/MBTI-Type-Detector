.PHONY: cleanlog, downcsv, downglove, cnn, cnns, rnn, rnns
cleanlog:
	rm -rf logs
downcsv:
	scp ml-winter-camp@106.15.92.255:~/data/MBTIv0_tokenized.csv ./
downglove:
	wget http://nlp.stanford.edu/data/wordvecs/glove.6B.zip
	unzip glove.6B.zip
cnn:
	python end2end.py -mcnn
rnn:
	python end2end.py -mrnn
cnns:
	python end2end.py -mcnn -s
rnns:
	python end2end.py -mrnn -s
