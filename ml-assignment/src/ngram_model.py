from collections import defaultdict, Counter
import random
import re
from preprocessing import TextPreprocessor

class TrigramModel:
    def __init__(self):
        """
        Initializes the TrigramModel.
        """
        # TODO: Initialize any data structures you need to store the n-gram counts.
        # Store trigram counts: (w1, w2) -> {w3: count}
        self.trigrams = defaultdict(Counter)
        # Store bigram counts for probability calculation
        self.bigrams = defaultdict(int)
        # onstain all the unique possible words
        self.vocab = set()
        pass

    def tokenize_sentences(self, text):
        """
        Trains the trigram model on the given text.

        Args:
            text (str): The text to train the model on.
        """
        # TODO: Implement the training logic.
        # This will involve:
        # 1. Cleaning the text (e.g., converting to lowercase, removing punctuation).
        # 2. Tokenizing the text into words.
        # 3. Padding the text with start and end tokens.
        # 4. Counting the trigrams.

        # There is separate file created for the preprocessing
        # We can change the paramter depending upon our use case
        preprocessor = TextPreprocessor(
        lowercase=True,
        remove_punctuation=False,
        keep_sentence_boundaries=True,
        handle_contractions=True,
        remove_numbers=True,
        min_word_length=1,
        min_word_frequency=2
        )
        tokens = preprocessor.preprocess_for_training(text)
        sentences = preprocessor.get_sentences(tokens)
        return sentences
    
    def fit(self, text):
        """Train the model on input text"""
        sentences = self.tokenize_sentences(text)
        
        # Add start tokens
        for sentence in sentences:
            tokens = ['<START>', '<START>'] + sentence + ['<END>']
            
        # Build trigram and bigram counts
            for i in range(len(tokens) - 2):
                w1, w2, w3 = tokens[i], tokens[i+1], tokens[i+2]
                self.trigrams[(w1, w2)][w3] += 1
                self.bigrams[(w1, w2)] += 1
                self.vocab.update([w1, w2, w3])

    def replace_unknown_words(self, words):
        """Replace unknown words with <UNK> token"""
        return [word if word in self.vocab else '<UNK>' for word in words]

    def generate(self, max_length=50, seed=None, seed_words=None):
        """
        Generates new text using the trained trigram model.

        Args:
            max_length (int): The maximum length of the generated text.

        Returns:
            str: The generated text.
        """
        # TODO: Implement the generation logic.
        # This will involve:
        # 1. Starting with the start tokens.
        # 2. Probabilistically choosing the next word based on the current context.
        # 3. Repeating until the end token is generated or the maximum length is reached.
        pass
        """Generate text using the trained model"""
        if seed:
            random.seed(seed)
        
        # Initialize with seed words or START tokens
        if seed_words:
            # Replace unknown seed words with <UNK>
            seed_words = self.replace_unknown_words(seed_words)
            
            if len(seed_words) == 1:
                w1, w2 = '<START>', seed_words[0]
            else:
                w1, w2 = seed_words[-2], seed_words[-1]
        else:
            # Start with START tokens
            w1, w2 = '<START>', '<START>'
        generated = []
        
        for _ in range(max_length):
            # Get possible next words and their counts
            possible_words = self.trigrams[(w1, w2)]
            
            if not possible_words:
                break
            
            # Sample next word based on probabilities
            words = list(possible_words.keys())
            weights = list(possible_words.values())
            w3 = random.choices(words, weights=weights)[0]
            
            if w3 == '<END>':
                break
            
            generated.append(w3)
            w1, w2 = w2, w3
        
        return ' '.join(generated)