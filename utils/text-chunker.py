# E. Culurciello, May 2023
# text chunker - cust pieces of text to feed to language model LM, LLM


def text_to_chunks(texts, word_length=150, start_page=1):
    chunks = []
    words = texts.split()
    num_words = len(words)

    for i in range(0, num_words, word_length):
        chunk = words[i:i+word_length]
        chunk = ' '.join(chunk).strip()
        chunk = f'[{i+1}]' + ' ' + '"' + chunk + '"'
        chunks.append(chunk)

    return chunks


if __name__ == "__main__":
    text_filename = "../data/text-stories/cthulhu-tiny.txt"

    with open(text_filename) as file:
        text = file.read()

    chunks = text_to_chunks(text, word_length=100)
    print("number of chunks:", len(chunks))
    print(chunks)