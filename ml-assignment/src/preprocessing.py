import re
from collections import Counter

class TextPreprocessor:
    def __init__(self, 
                 lowercase=True,
                 remove_punctuation=False,
                 keep_sentence_boundaries=True,
                 handle_contractions=True,
                 min_word_length=1,
                 remove_numbers=True,
                 min_word_frequency=2):
        """
        Initialize preprocessor with various options
        
        Args:
            lowercase: Convert all text to lowercase
            remove_punctuation: Remove all punctuation marks
            keep_sentence_boundaries: Keep sentence-ending punctuation (. ! ?)
            handle_contractions: Expand contractions (don't -> do not)
            min_word_length: Minimum word length to keep
            remove_numbers: Remove numeric tokens
            remove_metadata: Remove Project Gutenberg headers/footers
            remove_chapter_markers: Remove chapter headings
        """
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.keep_sentence_boundaries = keep_sentence_boundaries
        self.handle_contractions = handle_contractions
        self.min_word_length = min_word_length
        self.remove_numbers = remove_numbers
        self.min_word_frequency = min_word_frequency

        self.word_counts = Counter()
        self.vocab = set()
        
        # Common contractions mapping
        self.contractions = {
            "don't": "do not", "won't": "will not", "can't": "cannot",
            "n't": " not", "'re": " are", "'ve": " have", "'ll": " will",
            "'d": " would", "'m": " am", "i'm": "i am", "he's": "he is",
            "she's": "she is", "it's": "it is", "that's": "that is",
            "what's": "what is", "where's": "where is", "who's": "who is",
            "how's": "how is", "let's": "let us", "there's": "there is",
            "here's": "here is"
        }
    
    
    def clean_text(self, text):
        """Remove unwanted characters and normalize whitespace"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        # This removes things like ™, ©, weird Unicode chars
        text = re.sub(r'[^\w\s.,!?;:\'\"-]', '', text)
        
        return text.strip()
    
    def expand_contractions(self, text):
        """Expand contractions like don't -> do not"""
        if not self.handle_contractions:
            return text
        
        # Sort by length (longest first) to handle overlapping patterns
        for contraction, expansion in sorted(self.contractions.items(), 
                                            key=lambda x: len(x[0]), 
                                            reverse=True):
            text = re.sub(r'\b' + re.escape(contraction) + r'\b', 
                         expansion, text, flags=re.IGNORECASE)
        return text
    
    def handle_punctuation(self, text):
        """Handle punctuation based on settings"""
        if self.keep_sentence_boundaries:
            # Keep sentence-ending punctuation as separate tokens
            text = re.sub(r'([.!?])', r' \1 ', text)
            # Remove other punctuation
            if self.remove_punctuation:
                text = re.sub(r'[^\w\s.!?]', '', text)
        elif self.remove_punctuation:
            # Remove all punctuation
            text = re.sub(r'[^\w\s]', '', text)
        else:
            # Keep all punctuation but separate it
            text = re.sub(r'([.,!?;:\'\"-])', r' \1 ', text)
        
        return text
    
    def tokenize(self, text):
        """
        Tokenize text into words/tokens
        Returns list of tokens
        """
        # Step 1: Clean text
        text = self.clean_text(text)
        
        # Step 2: Lowercase if needed
        if self.lowercase:
            text = text.lower()
        
        # Step 3: Expand contractions
        text = self.expand_contractions(text)
        
        # Step 4: Handle punctuation
        text = self.handle_punctuation(text)
        
        # Step 5: Split into tokens
        tokens = text.split()
        
        # Step 6: Filter tokens
        filtered_tokens = []
        for token in tokens:
            # Remove numbers if specified
            if self.remove_numbers and token.isdigit():
                continue
            
            # Filter by minimum length (but keep punctuation)
            if len(token) >= self.min_word_length or token in '.!?,;:':
                filtered_tokens.append(token)
        
        return filtered_tokens
    
    def build_vocabulary(self, tokens):
        """
    Count word frequencies to identify rare words
    Only counts actual words, not punctuation
        """
        words_only = [t for t in tokens if t not in '.!?,;:\'"']
        self.word_counts = Counter(words_only)
        
        # Build vocabulary of words that meet frequency threshold
        self.vocab = {
            word for word, count in self.word_counts.items() 
            if count >= self.min_word_frequency
        }
        
        # Always include special tokens
        self.vocab.update(['<START>', '<END>', '<UNK>'])
        
        # Always keep punctuation in vocab
        self.vocab.update(['.', '!', '?', ',', ';', ':', '"', "'"])

    def replace_rare_words(self, tokens):
        """
        Replace rare words (below min_word_frequency) with <UNK>
        Keeps punctuation unchanged
        """
        replaced_tokens = []
        
        for token in tokens:
            # Keep punctuation as-is
            if token in '.!?,;:\'"':
                replaced_tokens.append(token)
            # Replace rare words with <UNK>
            elif token not in self.vocab:
                replaced_tokens.append('<UNK>')
            else:
                replaced_tokens.append(token)
        
        return replaced_tokens

    
    def preprocess_for_training(self, text):
        """
        Two-pass preprocessing for training:
        1. First pass: tokenize and build vocabulary
        2. Second pass: replace rare words with <UNK>
        """
        # First pass: tokenize
        tokens = self.tokenize(text)
        
        # Build vocabulary based on frequency
        self.build_vocabulary(tokens)
        
        # Second pass: replace rare words
        # tokens = self.replace_rare_words(tokens)
        
        return tokens
    
    def get_sentences(self, tokens):
        """
        Split tokens into sentences based on sentence-ending punctuation
        Returns list of lists (each inner list is a sentence)
        """
        sentences = []
        current_sentence = []
        
        for token in tokens:
            current_sentence.append(token)
            if token in ['.', '!', '?']:
                if current_sentence:
                    sentences.append(current_sentence)
                    current_sentence = []
        
        # Add remaining tokens as a sentence if any
        if current_sentence:
            sentences.append(current_sentence)
        
        return sentences
    
    def analyze_text(self, text):
        """
        Analyze text and return statistics
        """
        tokens = self.tokenize(text)
        
        # Count unique words (excluding punctuation)
        words = [t for t in tokens if t not in '.!?,;:']
        word_counts = Counter(words)
        
        sentences = self.get_sentences(tokens)
        
        stats = {
            'total_tokens': len(tokens),
            'total_words': len(words),
            'unique_words': len(word_counts),
            'vocabulary_size': len(word_counts),
            'num_sentences': len(sentences),
            'avg_sentence_length': len(words) / len(sentences) if sentences else 0,
            'most_common_words': word_counts.most_common(20),
            'sample_tokens': tokens[:50]
        }
        
        return stats


# Example usage and comparison
if __name__ == "__main__":
    print("Reading Pride and Prejudice...")
    with open(r"C:\Users\rishi\Desktop\ml-intern-assessment-main\pg1342.txt", "r", encoding="utf-8") as f:
        sample_text = f.read()
    
    # Configuration 2: Moderate preprocessing
    print("\nPREPROCESSING (expand contractions, keep punctuation)")
    print("-" * 70)
    preprocessor2 = TextPreprocessor(
        lowercase=True,
        remove_punctuation=False,
        keep_sentence_boundaries=True,
        handle_contractions=True
    )
    tokens2 = (preprocessor2.tokenize(sample_text))
    # token2 = preprocessor2.remove_chapter_headings(tokens2)
    print(tokens2[:100])
    tokens2 = preprocessor2.preprocess_for_training(sample_text)
    print('The Final Token is:', tokens2[:20])
    sentences = preprocessor2.get_sentences(tokens2)
    print(f'The final sentences are: ', sentences[:10])
    print('-'*70)
    print(f'The Stats are: {preprocessor2.analyze_text(sample_text)}')