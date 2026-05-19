// Load your API key from .env
require("dotenv").config();

async function askGemma(question) {
  const response = await fetch("https://openrouter.ai/api/v1/chat/completions", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${process.env.OPENROUTER_API_KEY}`,
      "Content-Type": "application/json",
      "HTTP-Referer": "http://localhost:3000",
      "X-Title": "gemma4-app"
    },
    body: JSON.stringify({
      model: "google/gemma-4-31b-it:free",  // ← free Gemma 4 model
      messages: [
        { role: "user", content: question }
      ]
    })
  });

  const data = await response.json();
  if (!response.ok) {
    const err =
      typeof data?.error === "string"
        ? data.error
        : data?.error?.message || JSON.stringify(data);
    throw new Error(err);
  }
  return data.choices?.[0]?.message?.content;
}

module.exports = { askGemma };

if (require.main === module) {
  const question =
    process.argv.slice(2).join(" ") || "Say hello in one short sentence.";
  askGemma(question)
    .then((text) => console.log(text))
    .catch((err) => {
      console.error(err);
      process.exit(1);
    });
}
