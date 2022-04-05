if [ ! -f glove.840B.300d.txt ]; then
  wget http://nlp.stanford.edu/data/glove.840B.300d.zip -O /tmp2/b08902116/glove.840B.300d.zip
  unzip /tmp2/b08902116/glove.840B.300d.zip
fi
python preprocess_intent.py --glove_path ./glove.840B.300d.txt
python preprocess_slot.py --glove_path ./glove.840B.300d.txt
