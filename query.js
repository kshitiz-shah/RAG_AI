import * as dotenv from "dotenv";
dotenv.config();

import express from "express";
import cors from "cors";
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import { Pinecone } from "@pinecone-database/pinecone";
import { GoogleGenAI } from "@google/genai";

const app = express();
app.use(cors());
app.use(express.json()); // parse JSON bodies

// Initialize AI
const ai = new GoogleGenAI({});
const History = [];

/**
 * Step 1: Rephrase query into a standalone question
 */
async function transformQuery(question) {
  History.push({
    role: "user",
    parts: [{ text: question }],
  });

  const response = await ai.models.generateContent({
    model: "gemini-2.0-flash",
    contents: History,
    config: {
      systemInstruction: `You are a query rewriting expert. Based on the provided chat history, rephrase the "Follow Up user Question" into a complete, standalone question that can be understood without the chat history.
      Only output the rewritten question and nothing else.`,
    },
  });

  History.pop();
  return response.text;
}

/**
 * 
 * Step 2: Full Chat Flow
 */
async function chatting(question) {
  const queries = await transformQuery(question);

  // Create embeddings
  const embeddings = new GoogleGenerativeAIEmbeddings({
    apiKey: process.env.GEMINI_API_KEY,
    model: "text-embedding-004",
  });

  const queryVector = await embeddings.embedQuery(queries);

  // Pinecone init
  const pinecone = new Pinecone();
  const pineconeIndex = pinecone.Index(process.env.PINECONE_INDEX_NAME);

  // Search Pinecone
  const searchResults = await pineconeIndex.query({
    topK: 10,
    vector: queryVector,
    includeMetadata: true,
  });

  // Build context
  const context = searchResults.matches
    .map((match) => match.metadata.text)
    .join("\n\n---\n\n");

  // Push user question to history
  History.push({
    role: "user",
    parts: [{ text: question }],
  });

  // Generate answer
  const response = await ai.models.generateContent({
    model: "gemini-2.0-flash",
    contents: History,
    config: {
      systemInstruction: `You have to behave like a Data Structure and Algorithm Expert.
      You will be given a context of relevant information and a user question.
      Your task is to answer the user's question based ONLY on the provided context.
      If the answer is not in the context, you must say "I could not find the answer in the provided document."
      Keep your answers clear, concise, and educational.
      
      Context: ${context}`,
    },
  });

  // Save assistant response in History
  History.push({
    role: "model",
    parts: [{ text: response.text }],
  });

  return {
    answer: response.text,
    context,
  };
}

/**
 * API Route: Ask a Question
 */
app.post("/query", async (req, res) => {
  try {
    const { question } = req.body;
    if (!question) {
      return res.status(400).json({ success: false, error: "Question is required" });
    }

    const result = await chatting(question);
    res.json({ success: true, ...result });
  } catch (err) {
    console.error("Error:", err);
    res.status(500).json({ success: false, error: err.message });
  }
});

app.listen(5001, () => {
  console.log("Query server running at http://localhost:5001");
});
