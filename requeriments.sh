# Install PyWSD
pip install -U pywsd
echo "Downloading resources..."
python <<< "import nltk
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('punkt')
"
