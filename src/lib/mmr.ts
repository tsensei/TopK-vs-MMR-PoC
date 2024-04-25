export function similarity(embedding1: number[], embedding2: number[]): number {
  if (embedding1.length !== embedding2.length) {
    throw new Error("Embedding length mismatch");
  }

  // Function to compute the Euclidean norm of a vector
  function norm(x: number[]): number {
    let result = 0;
    for (let i = 0; i < x.length; i++) {
      result += x[i] * x[i];
    }
    return Math.sqrt(result);
  }

  // Compute the dot product of the two embeddings
  let dotProduct = 0;
  for (let i = 0; i < embedding1.length; i++) {
    dotProduct += embedding1[i] * embedding2[i];
  }

  // Compute the norms of each embedding
  const norm1 = norm(embedding1);
  const norm2 = norm(embedding2);

  // Return the cosine similarity, which is the dot product divided by the product of the norms
  return dotProduct / (norm1 * norm2);
}

/**
 * Retrieves the top K embeddings from a list based on their similarity to a specified query embedding.
 * This function calculates the similarity of each embedding in the provided list to the query embedding
 * and returns the K embeddings with the highest similarity scores. Optionally, a similarity cutoff can
 * be specified to filter out embeddings with scores below a certain threshold before selecting the top K.
 *
 * @param {number[]} queryEmbedding - The query vector against which all embeddings are compared.
 *                                    This should be an array of numbers representing the query embedding.
 * @param {number[][]} embeddings - An array of embeddings, where each embedding is itself an array of numbers.
 *                                  Each embedding is compared to the query embedding to determine its similarity.
 * @param {number} [similarityTopK=2] - The number of top similar embeddings to return. If more embeddings are
 *                                      requested than are available, only the available number will be returned.
 * @param {any[] | null} [embeddingIds=null] - An optional array of identifiers corresponding to each embedding.
 *                                             If provided, these identifiers are used in the output to indicate
 *                                             which embeddings have been selected. If null, the indices of the
 *                                             embeddings array are used as identifiers.
 * @param {number | null} [similarityCutoff=null] - An optional cutoff threshold for similarity scores.
 *                                                  Only embeddings with a similarity score above this threshold
 *                                                  will be considered for selection. If null, all embeddings are considered.
 *
 * @returns {[number[], any[]]} A tuple where the first element is an array of the top K similarity scores, and
 *                              the second element is an array of identifiers for the embeddings corresponding to
 *                              these scores. The scores and identifiers are aligned by index, providing a direct
 *                              correlation between an embedding's score and its identifier.
 *
 * @example
 * // Define a query vector and a set of embeddings
 * const query = [0.1, 0.2, 0.3];
 * const embeddings = [
 *   [0.1, 0.2, 0.3],
 *   [0.4, 0.5, 0.6],
 *   [0.7, 0.8, 0.9]
 * ];
 * const ids = ['id1', 'id2', 'id3'];
 * const topK = 2;
 * const cutoff = 0.5;
 *
 * // Call the function
 * const [scores, selectedIds] = getTopKEmbeddings(query, embeddings, topK, ids, cutoff);
 * console.log(scores); // Logs the top K similarity scores
 * console.log(selectedIds); // Logs the identifiers of the top K embeddings
 */
export function getTopKEmbeddings(
  queryEmbedding: number[],
  embeddings: number[][],
  similarityTopK: number = 2,
  embeddingIds: any[] | null = null,
  similarityCutoff: number | null = null
): [number[], any[]] {
  if (embeddingIds == null) {
    embeddingIds = Array.from({ length: embeddings.length }, (_, i) => i);
  }

  if (embeddingIds.length !== embeddings.length) {
    throw new Error(
      "getTopKEmbeddings: embeddings and embeddingIds length mismatch"
    );
  }

  const similarities: { similarity: number; id: number }[] = [];

  for (let i = 0; i < embeddings.length; i++) {
    const sim = similarity(queryEmbedding, embeddings[i]);
    if (similarityCutoff == null || sim > similarityCutoff) {
      similarities.push({ similarity: sim, id: embeddingIds[i] });
    }
  }

  similarities.sort((a, b) => b.similarity - a.similarity); // Reverse sort

  const resultSimilarities: number[] = [];
  const resultIds: any[] = [];

  for (let i = 0; i < similarityTopK; i++) {
    if (i >= similarities.length) {
      break;
    }
    resultSimilarities.push(similarities[i].similarity);
    resultIds.push(similarities[i].id);
  }

  return [resultSimilarities, resultIds];
}

/**
   * Retrieves the top K embeddings from a list, optimized for both relevance to a query embedding and diversity among the selected embeddings.
   * This function implements the Maximal Marginal Relevance (MMR) algorithm, which balances the trade-off between relevance (similarity to the query)
   * and diversity (dissimilarity among chosen embeddings). This method is particularly useful in scenarios like search engines, recommendation systems,
   * and clustering tasks where diversity is as crucial as relevance.
   * @param {number[]} queryEmbedding - A vector representing the query for which relevant and diverse embeddings are to be retrieved.
   * @param {number[][]} embeddings - An array of vectors, each vector is an embedding to be evaluated against the query.
   * @param {number | null} similarityTopK - The number of top embeddings to return. If null, all embeddings are considered.
   * @param {any[] | null} embeddingIds - Optional identifiers for each embedding. If null, indices of the embeddings array are used as IDs.
   * @param {number | null} mmrThreshold - A threshold to balance between relevance and diversity in the MMR calculation.
   If null, defaults to 0.5, giving equal weight to both relevance and diversity.
   * @returns {[number[], any[]]} A tuple where the first element is an array of similarity scores for the selected embeddings
  (reflecting both their similarity to the query and their diversity), and the second element
  is an array of identifiers for these embeddings. The scores and identifiers are aligned by index.
   * @example // Define a query vector and a set of embeddings
  const query = [0.1, 0.2, 0.3];
  const embeddings = [
    [0.1, 0.2, 0.3],
    [0.4, 0.5, 0.6],
    [0.7, 0.8, 0.9]
  ];
  const ids = ['id1', 'id2', 'id3'];
  const topK = 2;
  const mmrThresh = 0.5;
  
  // Call the function
  const [scores, selectedIds] = getTopKMMREmbeddings(query, embeddings, topK, ids, null, mmrThresh);
  console.log(scores); // Logs similarity scores
  console.log(selectedIds); // Logs identifiers of the top K embeddings
   */
export function getTopKMMREmbeddings(
  queryEmbedding: number[],
  embeddings: number[][],
  similarityTopK: number | null = null,
  embeddingIds: any[] | null = null,
  mmrThreshold: number | null = null
): [number[], any[]] {
  const threshold = mmrThreshold || 0.5;
  let similarityFn = similarity;

  if (embeddingIds === null || embeddingIds.length === 0) {
    embeddingIds = Array.from({ length: embeddings.length }, (_, i) => i);
  }

  const fullEmbedMap = new Map(embeddingIds.map((value, i) => [value, i]));
  const embedMap = new Map(fullEmbedMap);
  const embedSimilarity: Map<any, number> = new Map();
  let score: number = Number.NEGATIVE_INFINITY;
  let highScoreId: any | null = null;

  for (let i = 0; i < embeddings.length; i++) {
    const emb = embeddings[i];
    const similarity = similarityFn(queryEmbedding, emb);
    embedSimilarity.set(embeddingIds[i], similarity);
    if (similarity * threshold > score) {
      highScoreId = embeddingIds[i];
      score = similarity * threshold;
    }
  }

  const results: [number, any][] = [];

  const embeddingLength = embeddings.length;
  const similarityTopKCount = similarityTopK || embeddingLength;

  while (results.length < Math.min(similarityTopKCount, embeddingLength)) {
    results.push([score, highScoreId]);
    embedMap.delete(highScoreId);
    const recentEmbeddingId = highScoreId;
    score = Number.NEGATIVE_INFINITY;
    for (const embedId of Array.from(embedMap.keys())) {
      const overlapWithRecent = similarityFn(
        embeddings[embedMap.get(embedId)!],
        embeddings[fullEmbedMap.get(recentEmbeddingId)!]
      );
      if (
        threshold * embedSimilarity.get(embedId)! -
          (1 - threshold) * overlapWithRecent >
        score
      ) {
        score =
          threshold * embedSimilarity.get(embedId)! -
          (1 - threshold) * overlapWithRecent;
        highScoreId = embedId;
      }
    }
  }

  const resultSimilarities = results.map(([s, _]) => s);
  const resultIds = results.map(([_, n]) => n);

  return [resultSimilarities, resultIds];
}
