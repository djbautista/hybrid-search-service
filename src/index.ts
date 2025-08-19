import 'dotenv/config';

import express from 'express';
import OpenAI from 'openai';
import { Pool } from 'pg';

const app = express();
app.use(express.json({ limit: "2mb" }));

const pool = new Pool({
  connectionString: process.env.DATABASE_URL
});
const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

const EMBEDDING_MODEL = "text-embedding-3-small";
const EMBEDDING_DIM = 1536;

app.get("/health", (_req, res) => {
  res.json({ status: "ok", date: new Date().toISOString() });
});

app.post("/hybrid-search", async (req, res) => {
  try {
    const { query, matchCount = 10, fullTextWeight = 1, semanticWeight = 1, rrfK = 50 } = req.body ?? {};

    if (!query || typeof query !== "string") {
      return res.status(400).json({ error: "Falta `query` (string)" });
    }

    const emb = await openai.embeddings.create({
      model: EMBEDDING_MODEL,
      input: query,
      dimensions: EMBEDDING_DIM
    });
    const vector = emb.data[0].embedding;

    const { rows } = await pool.query(
      `SELECT *
         FROM hybrid_search(
           $1::text,
           $2::vector(${EMBEDDING_DIM}),
           $3::int,
           $4::float,
           $5::float,
           $6::int
         )`,
      [query, vector, matchCount, fullTextWeight, semanticWeight, rrfK]
    );

    res.json({ data: rows });
  } catch (e: any) {
    console.error("hybrid-search error:", e?.message || e);
    res.status(500).json({ error: "Error interno" });
  }
});

const port = Number(process.env.PORT) || 8080;

app.listen(port, "0.0.0.0", () => {
  console.log(`Hybrid search service on :${port}`);
});
