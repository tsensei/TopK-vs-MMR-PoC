import "dotenv/config";
import { OpenAIEmbeddings } from "@langchain/openai";
import { getTopKEmbeddings, getTopKMMREmbeddings } from "./lib/mmr.js";
const contextStringArray = [
    "AI beats human at Go with new strategies.",
    "Machine learning models now predict stock market trends.",
    "Deep learning applications in medical diagnosis.",
    "Advancements in neural networks for natural language processing.",
    "AI in autonomous driving: The future of transportation.",
    "The ethical considerations of AI in surveillance.",
    "Using AI to enhance virtual reality experiences.",
    "Challenges in AI regulation and policy making.",
    "AI-powered robotics in manufacturing.",
    "The role of big data in machine learning advancements.",
    "New programming languages for blockchain technology.",
    "Best practices for database security and management.",
    "Latest trends in web development technologies.",
    "Cloud computing: Cost-effective solutions for startups.",
    "How quantum computing could revolutionize encryption.",
    "Cybersecurity risks in an increasingly digital world.",
    "Emerging technologies in mobile app development.",
    "The impact of social media algorithms on public opinion.",
    "Computer graphics techniques for more realistic video games.",
    "Technological innovations in renewable energy.",
];
const queryString = "What are the latest advancements in AI?";
const embeddings = new OpenAIEmbeddings();
const queryEmbedding = await embeddings.embedQuery(queryString);
const contextEmbeddings = await embeddings.embedDocuments(contextStringArray);
function mapIdsToStringsAndScores(ids, scores, strings) {
    return ids.map((id, index) => `${strings[id]} (Score: ${scores[index].toFixed(2)})`);
}
const [topKSimilarities, topKIds] = getTopKEmbeddings(queryEmbedding, contextEmbeddings, 5);
console.log("# Results with top K #");
console.log(mapIdsToStringsAndScores(topKIds, topKSimilarities, contextStringArray));
const [mmrSimilarities, mmrIds] = getTopKMMREmbeddings(queryEmbedding, contextEmbeddings, 5);
console.log("# Results with MMR #");
console.log(mapIdsToStringsAndScores(mmrIds, mmrSimilarities, contextStringArray));
//# sourceMappingURL=app.js.map