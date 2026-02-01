import * as dotenv from "dotenv";
dotenv.config();

import express from "express";
import cors from "cors";
import multer from "multer";
import fs from "fs";

import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import { Pinecone } from "@pinecone-database/pinecone";
import { PineconeStore } from "@langchain/pinecone";

const app = express();
app.use(cors());

// Multer setup for file uploads
const upload = multer({ dest: "uploads/" });

/**
 * Upload and index a PDF
 */
app.post("/upload", upload.single("file"), async (req, res) => {
  try {
    const filePath = req.file.path; // uploaded file path
    console.log("PDF uploaded:", filePath);

    // Load PDF
    const pdfLoader = new PDFLoader(filePath);
    const rawDocs = await pdfLoader.load();
     console.log("✅ PDF loaded");

    // Split into chunks
    const textSplitter = new RecursiveCharacterTextSplitter({
      chunkSize: 1000,
      chunkOverlap: 200,
    });
    const chunkedDocs = await textSplitter.splitDocuments(rawDocs);
     console.log("✅ Chunking completed");

    // Embeddings
    const embeddings = new GoogleGenerativeAIEmbeddings({
      apiKey: process.env.GEMINI_API_KEY,
      model: "text-embedding-004",
    });

    // Pinecone init
    const pinecone = new Pinecone();
    const pineconeIndex = pinecone.Index(process.env.PINECONE_INDEX_NAME);

    // Store in Pinecone
    await PineconeStore.fromDocuments(chunkedDocs, embeddings, {
      pineconeIndex,
      maxConcurrency: 5,
    });

    console.log("🎉 Indexing success");
    res.json({ success: true, message: "PDF indexed successfully" });

    // Cleanup uploaded file
    fs.unlinkSync(filePath);
  } catch (err) {
    console.error("❌ Error:", err);
    res.status(500).json({ success: false, error: err.message });
  }
});

app.listen(5000, () => {
  console.log("🚀 Server running on http://localhost:5000");
});
