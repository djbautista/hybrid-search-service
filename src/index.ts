import 'dotenv/config';

import express from 'express';
import OpenAI from 'openai';
import { Pool } from 'pg';

const app = express();
app.use(express.json());

const pool = new Pool({ connectionString: process.env.DATABASE_URL });
const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

const EMBEDDING_MODEL = 'text-embedding-3-small';

app.get('/health', (_req, res) => {
  res.json({ status: 'ok', date: new Date().toISOString() });
});

app.post('/hybrid-search', async (req, res) => {
  try {
    const { query, matchCount = 10, alpha = 0.5 } = req.body ?? {};
    if (!query || typeof query !== 'string') {
      return res.status(400).json({ error: 'Falta `query` (string)' });
    }

    const emb = await openai.embeddings.create({ model: EMBEDDING_MODEL, input: query });
    const vector = emb.data[0].embedding;

    const { rows } = await pool.query(
      `SELECT id, content, rank
       FROM hybrid_search($1::vector(1536), $2::text, $3::int, $4::float)`,
      [vector, query, matchCount, alpha]
    );
    res.json({ data: rows });
  } catch (e: any) {
    console.error(e);
    res.status(500).json({ error: 'Error interno' });
  }
});

const port = process.env.PORT || 8080;
app.listen(port, () => console.log(`Hybrid search service on :${port}`));
