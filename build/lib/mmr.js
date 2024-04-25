export function similarity(embedding1, embedding2) {
    if (embedding1.length !== embedding2.length) {
        throw new Error("Embedding length mismatch");
    }
    function norm(x) {
        let result = 0;
        for (let i = 0; i < x.length; i++) {
            result += x[i] * x[i];
        }
        return Math.sqrt(result);
    }
    let dotProduct = 0;
    for (let i = 0; i < embedding1.length; i++) {
        dotProduct += embedding1[i] * embedding2[i];
    }
    const norm1 = norm(embedding1);
    const norm2 = norm(embedding2);
    return dotProduct / (norm1 * norm2);
}
export function getTopKEmbeddings(queryEmbedding, embeddings, similarityTopK = 2, embeddingIds = null, similarityCutoff = null) {
    if (embeddingIds == null) {
        embeddingIds = Array.from({ length: embeddings.length }, (_, i) => i);
    }
    if (embeddingIds.length !== embeddings.length) {
        throw new Error("getTopKEmbeddings: embeddings and embeddingIds length mismatch");
    }
    const similarities = [];
    for (let i = 0; i < embeddings.length; i++) {
        const sim = similarity(queryEmbedding, embeddings[i]);
        if (similarityCutoff == null || sim > similarityCutoff) {
            similarities.push({ similarity: sim, id: embeddingIds[i] });
        }
    }
    similarities.sort((a, b) => b.similarity - a.similarity);
    const resultSimilarities = [];
    const resultIds = [];
    for (let i = 0; i < similarityTopK; i++) {
        if (i >= similarities.length) {
            break;
        }
        resultSimilarities.push(similarities[i].similarity);
        resultIds.push(similarities[i].id);
    }
    return [resultSimilarities, resultIds];
}
export function getTopKMMREmbeddings(queryEmbedding, embeddings, similarityTopK = null, embeddingIds = null, mmrThreshold = null) {
    const threshold = mmrThreshold || 0.5;
    let similarityFn = similarity;
    if (embeddingIds === null || embeddingIds.length === 0) {
        embeddingIds = Array.from({ length: embeddings.length }, (_, i) => i);
    }
    const fullEmbedMap = new Map(embeddingIds.map((value, i) => [value, i]));
    const embedMap = new Map(fullEmbedMap);
    const embedSimilarity = new Map();
    let score = Number.NEGATIVE_INFINITY;
    let highScoreId = null;
    for (let i = 0; i < embeddings.length; i++) {
        const emb = embeddings[i];
        const similarity = similarityFn(queryEmbedding, emb);
        embedSimilarity.set(embeddingIds[i], similarity);
        if (similarity * threshold > score) {
            highScoreId = embeddingIds[i];
            score = similarity * threshold;
        }
    }
    const results = [];
    const embeddingLength = embeddings.length;
    const similarityTopKCount = similarityTopK || embeddingLength;
    while (results.length < Math.min(similarityTopKCount, embeddingLength)) {
        results.push([score, highScoreId]);
        embedMap.delete(highScoreId);
        const recentEmbeddingId = highScoreId;
        score = Number.NEGATIVE_INFINITY;
        for (const embedId of Array.from(embedMap.keys())) {
            const overlapWithRecent = similarityFn(embeddings[embedMap.get(embedId)], embeddings[fullEmbedMap.get(recentEmbeddingId)]);
            if (threshold * embedSimilarity.get(embedId) -
                (1 - threshold) * overlapWithRecent >
                score) {
                score =
                    threshold * embedSimilarity.get(embedId) -
                        (1 - threshold) * overlapWithRecent;
                highScoreId = embedId;
            }
        }
    }
    const resultSimilarities = results.map(([s, _]) => s);
    const resultIds = results.map(([_, n]) => n);
    return [resultSimilarities, resultIds];
}
//# sourceMappingURL=mmr.js.map