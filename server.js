require("dotenv").config();
const express = require("express");
const multer = require("multer");
const cors = require("cors");
const { Pool } = require("pg");

const app = express();
const upload = multer({ storage: multer.memoryStorage() }); // store image in memory

app.use(cors());
app.use(express.json());

// Connect to Neon database (optional — server still starts without it)
const db = process.env.DATABASE_URL
  ? new Pool({ connectionString: process.env.DATABASE_URL, ssl: true })
  : null;

async function setupDB() {
  if (!db) {
    console.warn(
      "⚠️  DATABASE_URL not set — uploads will work but receipts will not be saved"
    );
    return;
  }
  await db.query(`
    CREATE TABLE IF NOT EXISTS receipts (
      id SERIAL PRIMARY KEY,
      merchant TEXT,
      amount NUMERIC,
      date TEXT,
      category TEXT,
      tax NUMERIC,
      raw_text TEXT,
      created_at TIMESTAMP DEFAULT NOW()
    )
  `);
  console.log("✅ Database ready");
}

// Competition requirement: Gemma 4 only (do not swap models)
const VISION_MODEL = "google/gemma-4-26b-a4b-it:free";

// One OpenRouter call at a time — avoids burning free-tier quota with parallel uploads
let visionQueue = Promise.resolve();
function runVisionQueued(fn) {
  const run = visionQueue.then(fn, fn);
  visionQueue = run.catch(() => {});
  return run;
}

const RECEIPT_PROMPT = `Look at this receipt image and extract the data.
Reply ONLY with a JSON object, nothing else:
{
  "merchant": "store name",
  "amount": 12.50,
  "date": "2026-05-14",
  "category": "Food & Dining",
  "tax": 1.10
}
For category use one of: Food & Dining, Transport, Shopping, Healthcare, Groceries, Entertainment, Other`;

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function openRouterErrorMessage(data) {
  return (
    data?.error?.metadata?.raw ||
    (typeof data?.error === "string"
      ? data.error
      : data?.error?.message || "OpenRouter API error")
  );
}

function normalizeReceipt(receipt) {
  const out = { ...receipt };
  if (typeof out.amount === "string") {
    out.amount = parseFloat(out.amount.replace(/[^0-9.-]/g, "")) || 0;
  }
  if (typeof out.tax === "string") {
    out.tax = parseFloat(out.tax.replace(/[^0-9.-]/g, "")) || 0;
  }
  return out;
}

function parseReceiptJson(rawText) {
  const stripped = rawText.replace(/```json|```/gi, "").trim();
  let receipt;
  try {
    receipt = JSON.parse(stripped);
  } catch {
    const match = stripped.match(/\{[\s\S]*\}/);
    if (!match) throw new Error("Could not parse receipt JSON from model response");
    receipt = JSON.parse(match[0]);
  }
  return normalizeReceipt(receipt);
}

async function callOpenRouterVision(base64Image, mimeType) {
  const maxAttempts = 8;
  const retryDelaysMs = [8000, 12000, 15000, 18000, 22000, 26000, 30000, 35000];
  let lastError = null;

  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 90000);

    try {
      console.log(
        `🤖 Step 3: OpenRouter attempt ${attempt}/${maxAttempts} (${VISION_MODEL})`
      );

      const response = await fetch(
        "https://openrouter.ai/api/v1/chat/completions",
        {
          method: "POST",
          signal: controller.signal,
          headers: {
            Authorization: `Bearer ${process.env.OPENROUTER_API_KEY}`,
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:3000",
            "X-Title": "ReceiptMind"
          },
          body: JSON.stringify({
            model: VISION_MODEL,
            messages: [
              {
                role: "user",
                content: [
                  {
                    type: "image_url",
                    image_url: {
                      url: `data:${mimeType};base64,${base64Image}`
                    }
                  },
                  { type: "text", text: RECEIPT_PROMPT }
                ]
              }
            ]
          })
        }
      );

      const data = await response.json();
      console.log(
        "📦 Step 5: OpenRouter status",
        response.status,
        data.error ? openRouterErrorMessage(data) : "ok"
      );

      if (response.status === 429 || data?.error?.code === 429) {
        lastError = {
          status: 429,
          message:
            openRouterErrorMessage(data) ||
            "Gemma 4 (free) is busy on OpenRouter. The server will keep retrying automatically."
        };
        if (attempt < maxAttempts) {
          const waitMs = retryDelaysMs[attempt - 1] || 30000;
          console.log(
            `⏳ Gemma 4 rate limited — retry ${attempt + 1}/${maxAttempts} in ${waitMs / 1000}s...`
          );
          await sleep(waitMs);
          continue;
        }
        return {
          ...lastError,
          message:
            "Gemma 4 is still rate-limited after several retries. Wait 2–3 minutes, upload one receipt at a time, or add a Google AI key in OpenRouter settings (BYOK) for higher limits."
        };
      }

      if (data.error || !response.ok) {
        return {
          status: response.status >= 400 ? response.status : 502,
          message: openRouterErrorMessage(data)
        };
      }

      if (!data.choices?.[0]?.message?.content) {
        return { status: 502, message: "No response from model" };
      }

      return { status: 200, rawText: data.choices[0].message.content, data };
    } catch (fetchErr) {
      if (fetchErr.name === "AbortError") {
        return {
          status: 504,
          message: "Gemma 4 timed out after 90 seconds. Try a smaller image."
        };
      }
      throw fetchErr;
    } finally {
      clearTimeout(timeout);
    }
  }

  return lastError || { status: 502, message: "OpenRouter request failed" };
}

app.get("/api/health", (_req, res) => {
  res.json({
    ok: true,
    app: "ReceiptMind",
    database: Boolean(db),
    model: VISION_MODEL,
    routes: ["POST /upload", "GET /receipts", "GET /api/health"]
  });
});

// Main route: upload receipt image → Gemma 4 reads it → save to DB (verbose logging)
app.post("/upload", upload.single("receipt"), async (req, res) => {
  try {
    console.log(
      "📥 Step 1: File received",
      req.file?.originalname,
      req.file?.size,
      "bytes"
    );

    if (!req.file) {
      return res.status(400).json({ error: "No file uploaded" });
    }

    if (!process.env.OPENROUTER_API_KEY) {
      return res.status(500).json({
        error: "OPENROUTER_API_KEY is missing from .env"
      });
    }

    const base64Image = req.file.buffer.toString("base64");
    const mimeType = req.file.mimetype || "image/jpeg";
    console.log(
      "📸 Step 2: Image converted to base64, length:",
      base64Image.length
    );

    const vision = await runVisionQueued(() =>
      callOpenRouterVision(base64Image, mimeType)
    );
    if (vision.status !== 200) {
      console.error("❌ Vision API failed:", vision.message);
      const httpStatus = vision.status === 429 ? 429 : vision.status || 502;
      return res.status(httpStatus).json({
        error: vision.message,
        retryable: httpStatus === 429
      });
    }

    const rawText = vision.rawText;
    console.log("💬 Step 6: Gemma said:", rawText);

    let receipt;
    try {
      receipt = parseReceiptJson(rawText);
    } catch (parseErr) {
      console.error("❌ Step 7 FAILED:", parseErr.message, rawText);
      return res.status(500).json({
        error: "Could not parse receipt data from model",
        raw: rawText
      });
    }

    console.log("✅ Step 7: Parsed receipt:", receipt);

    if (db) {
      try {
        await db.query(
          `INSERT INTO receipts (merchant, amount, date, category, tax, raw_text)
           VALUES ($1, $2, $3, $4, $5, $6)`,
          [
            receipt.merchant,
            receipt.amount,
            receipt.date,
            receipt.category,
            receipt.tax,
            rawText
          ]
        );
        console.log("✅ Step 8: Saved to database!");
      } catch (dbErr) {
        console.error("❌ Step 8 DB error:", dbErr.message);
        return res.status(500).json({
          error: `Receipt read OK but database save failed: ${dbErr.message}`
        });
      }
    } else {
      console.log("⚠️ Step 8: Skipped DB save (no DATABASE_URL)");
    }

    res.json({ success: true, receipt, savedToDb: Boolean(db) });
  } catch (err) {
    console.error("💥 Unexpected error:", err);
    res.status(500).json({ error: err.message });
  }
});

// Get all receipts for the dashboard
app.get("/receipts", async (req, res) => {
  if (!db) {
    return res.json([]);
  }
  try {
    const result = await db.query(
      "SELECT * FROM receipts ORDER BY created_at DESC"
    );
    res.json(result.rows);
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: err.message });
  }
});

// Static files last so /upload and /receipts are never shadowed
app.use(express.static("public")); // serves your index.html

const PORT = process.env.PORT || 3000;

setupDB()
  .then(() => {
    app.listen(PORT, () =>
      console.log(`🚀 ReceiptMind running at http://localhost:${PORT}`)
    );
  })
  .catch((err) => {
    console.error(err);
    process.exit(1);
  });
