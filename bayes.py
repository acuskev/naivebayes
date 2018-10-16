import math, os, pickle, re
from typing import Tuple, List, Dict

class BayesClassifier:
  """BayesClassifier implementation

  attributes:
    pos_freqs - dictionary of frequencies of positive words
    neg_freqs - dictionary of frequencies of negative words
    pos_filename - name of positive dictionary cache file
    neg_filename - name of positive dictionary cache file
    training_data_directory - relative path to training directory
    neg_file_prefix - prefix of negative reviews
    pos_file_prefix - prefix of positive reviews
  """

  def __init__(self):
    """constructor initializes and trains the Naive Bayes Sentiment classifier.
    if cache of a trained classifier already exists in current folder, it'll be 
    loaded, otherwise system will train
    """
    # initialize attributes
    self.pos_freqs: Dict[str, int] = {}
    self.neg_freqs: Dict[str, int] = {}
    self.pos_filename: str = "pos.dat"
    self.neg_filename: str = "neg.dat"
    self.training_data_directory: str = "movie_reviews/" #IMPORTANT -- assumes movie_reviews lies in the same folder as this script
    self.neg_file_prefix: str = "movies-1" #indicates neg reviews
    self.pos_file_prefix: str = "movies-5" #indicates pos reviews 

    # check if both cached classifiers exist within the current directory
    if os.path.isfile(self.pos_filename) and os.path.isfile(self.neg_filename):
      print("Data files found - loading to use cached values...")
      self.pos_freqs = self.load_dict(self.pos_filename)
      self.neg_freqs = self.load_dict(self.neg_filename)
    else:
      print("Data files not found - running training...")
      self.train()

  def train(self) -> None:
    """trains classifier, generating pos_freq/neg_freq dictionaries using frequencies
    of words found in corresponding positive/negative reviews
    """
    # get the list of file names from the training data directory
    # os.walk returns a generator
    _, __, files = next(os.walk(self.training_data_directory), (None, None, []))
    if not files:
      raise RuntimeError(f"Couldn't find path {self.training_data_directory}")

    # files now holds a list of the filenames
    # self.training_data_directory holds the folder name where these files are stored

    # text = literal text of the file, what you would see if you
    # opened the file in a text editor
    # text = self.load_file(os.path.join(self.training_data_directory, fName))

    # want to fill pos_freqs and neg_freqs with the correct counts of words
    # their respective reviews

    # for each file, which we have already labelled by prefix:
    #  if negative file, update frequencies in negative dictionary.
    #  if positive file, update frequencies in positive dictionary.
    #  if neither postive or negative, ignore and move to next file 
    
    for index, f in enumerate(files): # type: ignore
      print(f"Training on file {index} of {len(files)}")
      filename_list = self.tokenize(files[index])
      if f.startswith(self.neg_file_prefix):
        #negative review
        text = self.load_file(os.path.join(self.training_data_directory, files[index]))
        text_list = self.tokenize(text)
        self.update_dict(text_list, self.neg_freqs)
      if f.startswith(self.pos_file_prefix):
        #positive review
        text = self.load_file(os.path.join(self.training_data_directory, files[index]))
        text_list = self.tokenize(text)
        self.update_dict(text_list, self.pos_freqs)
    
    self.save_dict(self.pos_freqs, self.pos_filename)
    self.save_dict(self.neg_freqs, self.neg_filename)
    # to update the frequencies for each file, need to get text of each file, tokenize, 
    # then update the appropriate dictionary for those tokens 
    # to make life easier, helper function `update_dict` takes our list of tokens from the 
    # file and the appropriate dictionary
    
    # self.save_dict saves the frequency dictionaries to avoid extra
    # work in the future: self.pos_freqs and self.neg_freqs and their filepaths, 
    # self.pos_filename and self.neg_filename

  def classify(self, text: str) -> str: #our string classifier which we can use after training
    """classifies given text as positive, negative or neutral from calculating
    the most likely document class to which the target string belongs

    Args:
      text - text to classify

    Returns:
      classification, either positive, negative or neutral
    """
    textlist = self.tokenize(text)
    pos_prob = 0.0
    neg_prob = 0.0
    pos_dict = self.load_dict(self.pos_filename)
    neg_dict = self.load_dict(self.neg_filename)
    pos_denom = sum(pos_dict.values())
    neg_denom = sum(neg_dict.values()) 
    for i, f in enumerate(textlist): #type: ignore
      if f in pos_dict:
        pos_prob += math.log((pos_dict[textlist[i]] + 1)/pos_denom)
      else:
        pos_prob += 1/pos_denom
      if f in neg_dict:
        neg_prob += math.log((neg_dict[textlist[i]]+ 1)/neg_denom)
      else:
        neg_prob += 1/neg_denom
    #print(pos_prob)
    #print(neg_prob)
    print("Classification for entered string: " + text)
    if pos_prob > neg_prob:
      return "positive :)"
    elif pos_prob < neg_prob:
      return "negative :("
    else:
      return "even"
    #tokenize text 
    #for each word:
      #update the positive probability 
      #pos_prob += math.log((number of times this word appears in positive documents + 1) / pos_denom)
      #neg_prob += math.log((number of times this word occurred in negative documents +1 / neg_denom))
      #however, some words can appear zero times in a given document - to fix this we want to add 1 
      #to the numerator and 'smooth' out the data; if a word doesn't exist in the dictionary, its numerator is 1
      #because all of these are fractions, though, the numbers will become quite small 
      #instead of multiplying the probabilities, we want to just add the logs of the probabilities, hence +=
      #math.log(fraction)
      #we'll need to wrap this all in an 'if' to check if such words appear in the dictionary 
      #just compare the two words with a greater than or less than 

    # return a string of "positive" or "negative"

  def load_file(self, filepath: str) -> str:
    """loads text of given file

    Args:
      filepath - relative path to file to load

    Returns:
      text of the given file
    """
    with open(filepath, encoding = 'utf-8') as f:
      return f.read()

  def save_dict(self, dict: Dict, filepath: str) -> None:
    """pickles given dictionary to a file with the given name

    Args:
      dict - a dictionary to pickle
      filepath - relative path to file to save
    """
    print(f'Dictionary saved to file: {filepath}')
    with open(filepath, 'wb') as f:
      pickle.Pickler(f).dump(dict)

  def load_dict(self, filepath: str) -> Dict:
    """loads pickled dictionary stored in given file

    Args:
      filepath - relative path to file to load

    Returns:
      dictionary stored in given file
    """
    #print(f'Loading dictionary from file: {filepath}')
    with open(filepath, "rb") as f:
      return pickle.Unpickler(f).load()

  def tokenize(self, text: str) -> List[str]: #tokenize input sentences to compare word by word
    """splits given text into a list of the individual tokens in order

    Args:
      text - text to tokenize

    Returns:
      tokens of given text in order
    """
    tokens = []
    token = ""
    for c in text:
      if re.match("[a-zA-Z0-9]", str(c)) != None or c == "\'" or c == "_" or c == '-':
        token += c
      else:
        if token != "":
          tokens.append(token)
          token = ""
        if c.strip() != "":
          tokens.append(str(c.strip()))

    if token != "": tokens.append(token)
    return tokens

  def update_dict(self, words: List[str], freqs: Dict[str, int]) -> None:
    """updates given (word -> frequency) dictionary with given words list

    Args:
      words - list of tokens to update frequencies of
      freqs - dictionary of frequencies to update
    """
    for word in words:
      word = word.strip().lower()
      if word in freqs:
        freqs[word] += 1
      else:
        freqs[word] = 1


bc = BayesClassifier()
print(bc.classify("I love python!")) # should print positive
print(bc.classify("I hate python!")) # print negative
print(bc.classify("happy")) # and so forth
print(bc.classify("please follow @blessed_animals on instagram for quality cute animal pictures ugh why is this classified as negative :((((("))
print(bc.classify("let's try again: @blessed_animals on instagram is incredible and aesthetically pleasing"))
print(bc.classify("thank you for pulling thru naive bayes"))