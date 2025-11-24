from ngram_model import TrigramModel

def main():
    # Create a new TrigramModel
    model = TrigramModel()

    # Train the model on the example corpus
    import os

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_PATH = os.path.join(BASE_DIR, "data", "example_corpus.txt")

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        book_text = f.read()

    model.fit(book_text)

    # Generate new text
    
    generated_text = model.generate()
    print(f"Generated Text")
    print(generated_text)

if __name__ == "__main__":
    main()
