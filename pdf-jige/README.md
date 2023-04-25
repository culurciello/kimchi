# PDF Processor

A set of tools to load pdf documents and be able to semantically search them. This is not a simple text-based search, but a semantic search, where if you search for "last-nname", you can get hits for "surname", "family name", "lastname", etc.


# create dataset

loads all PDF from folder `test_pdfs/` and extracts text and pictures in dir `results/`:

`python pdf_extractor.py`

# sentence pre-processor

add embeddings for every sentence:

`python pdf_embed.py`


# search in dataset
Search for a string in the dataset / PDFs

`python pdf_search.py "multi-headed attention"`