.PHONY: cleanlog, downcsv, downglove, cnn, cnns, rnn, rnns
cleanlog:
	rm -rf logs
downcsv:
	scp ml-winter-camp@106.15.92.255:~/data/MBTIv0_tokenized.csv ./
downglove:
	wget http://nlp.stanford.edu/data/wordvecs/glove.6B.zip
	unzip glove.6B.zip
cnn:
	python end2end.py -m zzw_cnn
rnn:
	python end2end.py -m zzw_lstm
cnns:
	python end2end.py -m zzw_cnn -s
rnns:
	python end2end.py -m zzw_lstm -s
tune:
	git pull
	python end2end.py -m zzw_cnn -e
