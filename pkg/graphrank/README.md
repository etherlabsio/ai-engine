# GraphRank

Custom graph functions used for building, extending and processing word cooccurrences.
A `TextRank`-based approach (variant of `PageRank`) is used for scoring nodes and extracting keyphrases.

## Installing
- **To install latest (master)**
`pip install git+https://github.com/etherlabsio/GraphRank`

- **To install a specific branch or release**
`pip install -e git+https://github.com/etherlabsio/GraphRank.git@0.5.5#egg=graphrank
`

## Instruction to run keyphrase service

Use `from graphrank import GraphRank` to work with this package

Refer `example.ipynb` to see what all functions can be performed w.r.t keyphrase extraction and graph building