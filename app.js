// ================= CONFIG =================
const TOTAL_CHUNKS = 17;          // поставь своё число
const USE_SINGLE_FILE = false;
const ONNX_URL =
  "https://huggingface.co/iammik3e/recsys-minilm/resolve/main/model.onnx";

// ================= STATE =================
let recipes = [];
let session = null;
let tokenizer = null;
let ort = null;

// ================= UI =================
const loading = document.getElementById("loading");
const results = document.getElementById("results");
const searchBtn = document.getElementById("searchBtn");
const ingredientsInput = document.getElementById("ingredientsInput");
const topN = document.getElementById("topN");

// ================= ONNX =================
async function loadONNXRuntime() {
  const m = await import(
    "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.2/dist/ort.wasm.min.js"
  );
  ort = m.default || m;
}

async function loadModel() {
  await loadONNXRuntime();
  session = await ort.InferenceSession.create(ONNX_URL, {
    executionProviders: ["wasm"],
  });
}

// ================= TOKENIZER =================
async function loadTokenizer() {
  const text = await fetch("model/vocab.txt").then((r) => r.text());
  const vocab = {};
  text.split("\n").forEach((t, i) => (vocab[t.trim()] = i));
  tokenizer = { vocab };
}

function tokenize(text) {
  text = text.toLowerCase().replace(/[^a-z0-9 ]+/g, " ");
  const tokens = text.split(" ").filter(Boolean);
  const ids = tokens.map((t) => tokenizer.vocab[t] ?? tokenizer.vocab["[UNK]"]);
  const cls = tokenizer.vocab["[CLS]"] ?? 0;
  return { ids: [cls, ...ids].slice(0, 128), len: Math.min(ids.length + 1, 128) };
}

function makeTensor(ids) {
  const arr = new BigInt64Array(128);
  ids.forEach((v, i) => (arr[i] = BigInt(v)));
  return new ort.Tensor("int64", arr, [1, 128]);
}

// ================= EMBEDDING =================
async function embed(text) {
  const tok = tokenize(text);
  const mask = new BigInt64Array(128);
  for (let i = 0; i < tok.len; i++) mask[i] = 1n;

  const out = await session.run({
    input_ids: makeTensor(tok.ids),
    attention_mask: new ort.Tensor("int64", mask, [1, 128]),
    token_type_ids: new ort.Tensor(
      "int64",
      new BigInt64Array(128),
      [1, 128]
    ),
  });

  const key = Object.keys(out)[0];
  const data = out[key].data;
  const hidden = out[key].dims[2];

  const emb = new Array(hidden).fill(0);
  for (let i = 0; i < hidden; i++) {
    for (let j = 0; j < tok.len; j++) {
      emb[i] += data[j * hidden + i];
    }
    emb[i] /= tok.len;
  }
  return emb;
}

// ================= DATA =================
async function loadChunks() {
  const parts = await Promise.all(
    Array.from({ length: TOTAL_CHUNKS }, (_, i) =>
      fetch(`chunks/part${i + 1}.json`)
        .then((r) => r.json())
        .catch(() => [])
    )
  );
  recipes = parts.flat();
}

// ================= UTILS =================
function cosine(a, b) {
  let d = 0,
    na = 0,
    nb = 0;
  for (let i = 0; i < a.length; i++) {
    d += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  return d / (Math.sqrt(na) * Math.sqrt(nb));
}

function getRecipeImage(r) {
  return r.image ? `images/${r.image}` : "placeholder.jpg";
}

// ================= SEARCH =================
async function recommend() {
  const query = ingredientsInput.value.trim();
  if (!query) return;

  loading.textContent = "Searching…";

  const userEmb = await embed(`Ingredients: ${query}`);

  const scored = recipes
    .map((r) => ({
      r,
      score: cosine(userEmb, r.embedding),
    }))
    .sort((a, b) => b.score - a.score)
    .slice(0, parseInt(topN.value));

  results.innerHTML = "";
  scored.forEach(({ r, score }) => {
    results.innerHTML += `
      <div class="recipe-card">
        <img src="${getRecipeImage(r)}" alt="${r.title}">
        <div class="card-body">
          <h3>${r.title}</h3>
          <div class="score">Similarity: ${score.toFixed(3)}</div>
          <div class="ingredients">
            ${r.ingredients.join(", ")}
          </div>
          <details>
            <summary>Instructions</summary>
            <p>${r.instructions}</p>
          </details>
        </div>
      </div>
    `;
  });

  loading.textContent = "";
}

// ================= INIT =================
async function init() {
  loading.textContent = "Loading…";
  await loadTokenizer();
  await loadModel();
  await loadChunks();
  loading.textContent = "Ready ✓";
}

searchBtn.onclick = recommend;
init();
