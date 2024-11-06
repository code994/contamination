import gzip
import json
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import subprocess

# Function to read and decompress JSON data from a gzip file as a generator
def read(file_path):
    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
        for line in f:
            yield json.loads(line)  # Yield each line as a dictionary

def extract_values(d):
    return ' '.join(d['metadata']['wikipedia']['passage'])+' '+d['text']

subprocess.run(["pip", "install", "wikitextparser"])
import wikitextparser
import string
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

def preprocess_text(text):
    text = wikitextparser.parse(text).plain_text()
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Tokenize and remove stop words
    tokens = text.split()
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

def jaccard_similarity_texts(text1, text2):
    set1 = set(text1.split())
    set2 = set(text2.split())
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def cosine_similarity_tfidf(text1, text2):
    # Combine the texts into a list for vectorization
    texts = [text1, text2]
    
    # Initialize the TF-IDF Vectorizer
    vectorizer = TfidfVectorizer()
    
    # Transform the texts into TF-IDF vectors
    tfidf_matrix = vectorizer.fit_transform(texts)
    
    # Calculate cosine similarity between the two TF-IDF vectors
    cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    
    return cosine_sim

subprocess.run(["pip", "install", "Levenshtein"], check=True)

import Levenshtein

def levenshtein_ratio(text1, text2):
    # Calculate Levenshtein distance
    dist = Levenshtein.distance(text1, text2)
    # Normalize by the length of the longer text
    max_len = max(len(text1), len(text2))
    # Return similarity ratio
    return 1 - (dist / max_len)

# from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity

# model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# def semantic_similarity(text1, text2):
#     """
#     Calculate the semantic similarity between two texts using Sentence Transformers.

#     Args:
#         text1 (str): The first text string.
#         text2 (str): The second text string.

#     Returns:
#         float: Cosine similarity score between 0 and 1, where 1 means identical.
#     """
#     # Load the Sentence Transformer model
#     # model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')  # You can choose a different model if needed

#     # Generate embeddings for both texts
#     embeddings = model.encode([text1, text2])

#     # Calculate cosine similarity
#     similarity = cosine_similarity([embeddings[0]], [embeddings[1]])

#     return similarity[0][0]

def is_content_included(short_text, long_text):
    """
    Check if the short text is included in the long text.

    Args:
        short_text (str): The short text string.
        long_text (str): The long text string.

    Returns:
        int: 1 if short text is in long text, 0 otherwise.
    """
    return 1 if short_text in long_text else 0

def are_words_included(short_text, long_text):
    """
    Check if all words in the short text are present in the long text.

    Args:
        short_text (str): The short text string.
        long_text (str): The long text string.

    Returns:
        int: 1 if all words in the short text are in the long text, 0 otherwise.
    """
    short_words = set(short_text.split())
    long_words = set(long_text.split())

    return 1 if short_words.issubset(long_words) else 0

from collections import Counter

def ngram_overlap(text1, text2, n):
    # Create n-grams for each text
    def ngrams(words, n):
        return [' '.join(words[i:i+n]) for i in range(len(words) - n + 1)]

    # Split texts into words
    words1 = text1.split()
    words2 = text2.split()

    # Generate n-grams
    ngrams1 = ngrams(words1, n)
    ngrams2 = ngrams(words2, n)

    # Count n-grams
    count1 = Counter(ngrams1)
    count2 = Counter(ngrams2)

    # Calculate the overlap
    overlap = sum((count1 & count2).values())
    
    # Normalize by the length of the longer text's n-grams
    max_length = max(len(ngrams1), len(ngrams2))
    similarity = overlap / max_length if max_length > 0 else 0
    
    return similarity

from rapidfuzz import fuzz

def fuzz_partial(short_text, long_text):
    similarity_score = fuzz.partial_ratio(short_text, long_text)
    return similarity_score

subprocess.run(['pip', 'install', 'evaluate'])

import evaluate

# Load the METEOR metric
meteor = evaluate.load("meteor")

def compute_windowed_meteor_score(query: str, matched_text: str, window_factor: int = 2) -> float:
    """
    Computes the highest METEOR score between a query and a sliding window over the matched text.

    Args:
        query (str): The processed query text (space-separated words).
        matched_text (str): The processed matched text (space-separated words).
        window_factor (int): Multiplier for the window size based on the query length.

    Returns:
        float: The highest METEOR score within the window constraint.
    """
    # Split both texts into tokens
    query_tokens = query.split()
    matched_tokens = matched_text.split()
    
    # Calculate the length of the window based on the query length
    query_length = len(query_tokens)
    window_size = min(window_factor * query_length, len(matched_tokens))

    # Initialize the highest score
    highest_score = 0.0

    # Slide the window across the matched tokens
    for i in range(len(matched_tokens) - window_size + 1):
        # Extract the current window segment
        window_segment = " ".join(matched_tokens[i:i + window_size])
        
        # Compute METEOR score between the query and current window segment
        result = meteor.compute(predictions=[query], references=[window_segment])
        score = result["meteor"]

        # Update the highest score if this one is greater
        highest_score = max(highest_score, score)

    return highest_score

# Function to list all files in a specified folder
def list_files_in_folder(folder_path):
    file_paths = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_paths.append(os.path.join(root, file))
    return file_paths

# List all files in the specified directory
all_files = list_files_in_folder('../dolma_data')

all_files_megawika = [f for f in all_files if 'megawika-' in f]
print(len(all_files_megawika))

def contamination_detection(file_path, l):
  # data_generator = read(file_path)

  file_data = [preprocess_text(extract_values(d)) for d in read(file_path)]

  print(f"{file_path} done reading...")

  l_pro = [preprocess_text(item) for item in l]

  jc, cos, lev, ng, fuz = [], [], [], [], []

  for it in l_pro:
    jct, cost, levt, ngt, fuzt = [], [], [], [], []
    for processed_text in file_data:
      jct.append(jaccard_similarity_texts(it, processed_text))
      cost.append(cosine_similarity_tfidf(it, processed_text))
      levt.append(levenshtein_ratio(it, processed_text))
      # semant.append(semantic_similarity(preprocess_text(it), preprocess_text(extract_values(d))))
      ngt.append(ngram_overlap(it, processed_text, 3))
      fuzt.append(fuzz_partial(it, processed_text))
      # mett.append(compute_windowed_meteor_score(preprocess_text(it), preprocess_text(extract_values(d))))
    jc.append(np.mean(jct))
    cos.append(np.mean(cost))
    lev.append(np.mean(levt))
    # seman.append(np.mean(semant))
    ng.append(np.mean(ngt))
    fuz.append(np.mean(fuzt))
    # met.append(np.mean(mett))
  return jc, cos, lev, ng, fuz

def pre_summarise_results(result):
  return [list(item) for item in zip(*result)]

def summarise_results(results):
  FINAL = []
  for i in range(5):
    FINAL.append([np.mean(item) for item in pre_summarise_results([result[i] for result in results])])
  return FINAL

# main function below

boolq_passages = list(set(pd.read_csv('../play_BoolQ/boolq-data.csv')['original context'].values.tolist()))
boolq_np_passages = list(set(pd.read_csv('../play_BoolQ/boolq-data.csv')['perturbed context'].values.tolist()))
boolq_questions = list(set(pd.read_csv('../play_BoolQ/boolq-data.csv')['question'].values.tolist()))

def contamination_wrapper_passage(file):
  print(f"Processing {file}...")  # Log file being processed
  return contamination_detection(file, boolq_passages)
def contamination_wrapper_np_passage(file):
  print(f"Processing {file}...")  # Log file being processed
  return contamination_detection(file, boolq_np_passages)
def contamination_wrapper_question(file):
  print(f"Processing {file}...")  # Log file being processed
  return contamination_detection(file, boolq_questions)

def process_files(all_files_megawika, suffix, data_list):
  wrapper_name = f"contamination_wrapper_{suffix}"
  contamination_wrapper = globals().get(wrapper_name)
  with ProcessPoolExecutor(max_workers=8) as executor:
    print(f"Total files to process: {len(all_files_megawika)}")
    results = list(tqdm(executor.map(contamination_wrapper, all_files_megawika), 
                        total=len(all_files_megawika), 
                        desc="Processing Files"))
  summaries = summarise_results(results)
  df = {
        f"{suffix}": data_list,
        'jc': summaries[0],
        'cos': summaries[1],
        'lev': summaries[2],
        # 'seman': summarise_results(results)[3],
        'ng': summaries[3],
        'fuz': summaries[4],
        # 'met': summarise_results(results)[5],
    }
  pd.DataFrame(df).to_csv(f'./{suffix}-boolq.csv', index=False)
  return 0

match_types = {
    'passage':boolq_passages,
    'np_passage':boolq_np_passages,
    'question': boolq_questions,
}

for suffix, data_list in match_types.items():
    process_files(all_files_megawika, suffix, data_list)
