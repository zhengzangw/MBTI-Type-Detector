.PHONY: demo, cleanlog, downcsv, downglove

demo:
	python interface.py

cleanlog:
	rm -rf logs

downcsv:
	scp ml-winter-camp@106.15.92.255:~/data/MBTIv0_tokenized.csv ./

downglove:
	wget http://nlp.stanford.edu/data/wordvecs/glove.6B.zip
	unzip glove.6B.zip

