.PHONY: cnn, lstm, downcsv, cleanlog, downglove, generatecsv, oversample
cleanlog:
	rm -rf logs
downcsv:
	scp ml-winter-camp@106.15.92.255:~/data/MBTIv0_tokenized.csv ./
downglove:
	wget http://nlp.stanford.edu/data/wordvecs/glove.6B.zip
	unzip glove.6B.zip
generatecsv:
	mv MBTIv0_tokenized.csv MBTIv0.csv
	python cleandata.csv
demo:
	python -m cnn -s -c 16
	python demo
oversample:
	python oversampling.py
	mv MBTIv0.csv MBTIv0_tokens.csv
	mv MBTIv0_oversample.csv MBTIv0.csv
	rm MBTIv1.csv
	mv MBTIv0.csv MBTIv1.csv
	
