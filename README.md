# TopK vs MRR PoC

This Proof of Concept (PoC) project demonstrates the implementation and comparison of two different methods for selecting relevant strings from a dataset based on a given query: Top K and Maximal Marginal Relevance (MMR). The project is designed to help users understand the practical differences between these methods when applied to text data.

## Introduction

### What is Top K?

The Top K method selects the top K entries from a dataset based solely on their relevance to the query. Relevance is typically determined by a similarity score, such as cosine similarity, which measures how close two text entries are in a vector space. This method is straightforward and ensures that the most relevant items are chosen, but it may result in a lack of diversity.

### What is MMR?

Maximal Marginal Relevance (MMR) aims to balance relevance to the query with diversity among the results. It not only considers how similar each entry is to the query but also how different it is from the items already selected. This approach helps in reducing redundancy and improving the breadth of information in the results.

## Project Overview

This project includes a simple setup where a predefined set of strings related to AI, machine learning, and various computer science topics is queried to extract relevant information. It showcases how both the Top K and MMR algorithms perform with the same input, providing a clear comparison of the output from each method.

## How to Run the Project

### Prerequisites

Before you can run this project, you'll need to have Node.js and pnpm installed on your machine. Additionally, you'll need an OpenAI API key to fetch embeddings for the text.

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/tsensei/TopK-vs-MMR-PoC.git
   cd TopK-vs-MMR-PoC
   ```
2. **Install dependencies:**
   ```bash
   pnpm install
   ```
3. **Running the Project**
   ```bash
   pnpm build
   pnpm start
   ```

### Contributing

Feel free to fork this project and submit pull requests. You can also open issues if you find bugs or have suggestions for improvements.

### Acknowledgments

This project was created as a PoC to demonstrate different text selection algorithms. It is built using OpenAI's API for embeddings and is intended for educational purposes. The top k and MMR codes were shimmed from llama-index's source
