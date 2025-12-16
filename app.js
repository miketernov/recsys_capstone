// ================= CONFIG =================
const TOTAL_CHUNKS = 8;
const USE_SINGLE_FILE = false;
const RECIPES_FILE = "recipes_all.json";

const ONNX_URL =
  "https://huggingface.co/iammik3e/recsys-minilm/resolve/main/model.onnx";

const DIET_EXCLUSIONS = {
  vegetarian: ["chicken", "beef", "pork", "fish", "meat", "bacon", "ham"],
  vegan: [
    "meat", "chicken", "beef", "fish",
    "milk", "cheese", "egg", "butter", "cream"
  ],
  nopork: ["pork", "bacon", "ham"]
};

// ================= STATE =================
let recipes = [];
let session = null;
let tokenizer = null;
let ort = null;
let ready = false;

// ================= UI =================
const loading = document.getElementById("loading");
const results = document.getElementById("results");
const searchBtn = document.getElementById("searchBtn");
const ingredientsInput = document.getElementById("ingredientsInput");
const topN = document.getElementById("topN");

const dietVegetarian = document.getElementById("dietVegetarian");
const dietVegan = document.getElementById("dietVegan");
const dietNoPork = document.getElementById("dietNoPork");

const loader = document.getElementById("loader");
const loaderText = document.getElementById("loaderText");
const progressFill = document.getElementById("progressFill");

function setProgress(percent, text) {
  loader.classList.remove("hidden");
  progressFill.style.width = percent + "%";
  if (text) loaderText.textContent = text;
}

function hideLoader() {
  loader.classList.add("hidden");
}

// ================= ONNX RUNTIME (СТАРЫЙ РАБОЧИЙ ВАРИАНТ) =================
async function loadONNXRuntime() {
  try {
    const m = await import(
      "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.2/dist/ort.wasm.min.js"
    );
    ort = m.default || m;
    if (ort?.InferenceSession) return ort;
  } catch {}

  return new Promise((resolve, reject) => {
    const s = document.createElement("script");
    s.src =
      "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.2/dist/ort.wasm.min.js";
    s.onload = () => window.ort ? resolve(window.ort) : reject();
    document.head.appendChild(s);
  });
}

// ================= TOKENIZER =================
async function loadTokenizer() {
  const text = await fetch("model/vocab.txt").then(r => r.text());
  const vocab = {};
  text.split("\n").forEach((t, i) => {
    vocab[t.trim()] = i;
  });
  tokenizer = { vocab };
}

function tokenize(text) {
  text = text.toLowerCase().replace(/[^a-z0-9 ]+/g, " ");
  const tokens = text.split(" ").filter(Boolean);
  const ids = tokens.map(t => tokenizer.vocab[t] ?? tokenizer.vocab["[UNK]"]);
  const cls = tokenizer.vocab["[CLS]"] ?? 0;

  return {
    ids: [cls, ...ids].slice(0, 128),
    len: Math.min(ids.length + 1, 128)
  };
}

function makeTensor(ids) {
  const arr = new BigInt64Array(128);
  ids.forEach((v, i) => (arr[i] = BigInt(v)));
  return new ort.Tensor("int64", arr, [1, 128]);
}

// ================= MODEL =================
async function loadModel() {
  ort = await loadONNXRuntime();
  session = await ort.InferenceSession.create(ONNX_URL, {
    executionProviders: ["wasm"]
  });
}

// ================= DATA =================
async function loadChunks() {
  if (USE_SINGLE_FILE) {
    recipes = await fetch(RECIPES_FILE).then(r => r.json());
    return;
  }

  const parts = await Promise.all(
    Array.from({ length: TOTAL_CHUNKS }, (_, i) =>
      fetch(`chunks/part${i + 1}.json`)
        .then(r => r.json())
        .catch(() => [])
    )
  );

  recipes = parts.flat();
}

// ================= EMBEDDING =================
async function embed(text) {
  const tok = tokenize("Ingredients: " + text);
  const mask = new BigInt64Array(128);
  for (let i = 0; i < tok.len; i++) mask[i] = 1n;

  const out = await session.run({
    input_ids: makeTensor(tok.ids),
    attention_mask: new ort.Tensor("int64", mask, [1, 128]),
    token_type_ids: new ort.Tensor(
      "int64",
      new BigInt64Array(128),
      [1, 128]
    )
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

// ================= UTILS =================
function cosine(a, b) {
  let d = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) {
    d += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  return d / (Math.sqrt(na) * Math.sqrt(nb));
}

function recipeText(r) {
  const t = (r.title || "").toLowerCase();
  const i = Array.isArray(r.ingredients)
    ? r.ingredients.join(" ").toLowerCase()
    : "";
  return `${t} ${i}`;
}

function getRecipeImage(r) {
  if (r.image) {
    return `images/${r.image}.jpg`;
  }
  return "placeholder.jpg";
}

// ================= SEARCH =================
async function recommend() {
  if (!ready) return;

  const query = ingredientsInput.value.trim();
  if (!query) return;

  let excluded = [];
  if (dietVegetarian.checked) excluded.push(...DIET_EXCLUSIONS.vegetarian);
  if (dietVegan.checked) excluded.push(...DIET_EXCLUSIONS.vegan);
  if (dietNoPork.checked) excluded.push(...DIET_EXCLUSIONS.nopork);

  loading.textContent = "Encoding…";
  results.innerHTML = "";

  const userEmb = await embed(query);

  const filtered = excluded.length
    ? recipes.filter(r =>
        !excluded.some(e => recipeText(r).includes(e))
      )
    : recipes;

  const n = parseInt(topN.value, 10);

  const scored = filtered
    .map(r => ({ r, score: cosine(userEmb, r.embedding) }))
    .sort((a, b) => b.score - a.score)
    .slice(0, n);

  results.innerHTML = "";

  scored.forEach(({ r, score }) => {
    results.innerHTML += `
      <div class="recipe-card">
        <img
          src="${getRecipeImage(r)}"
          onerror="this.src='placeholder.jpg'"
        >
        <div class="card-body">
          <h3>${r.title}</h3>
          <div class="score">Similarity: ${score.toFixed(3)}</div>

          <div class="ingredients">
            <strong>Ingredients:</strong><br>
            ${r.ingredients.join(", ")}
          </div>

          <details>
            <summary>Recipe</summary>
            <p class="instructions">${r.instructions}</p>
          </details>
        </div>
      </div>
    `;
  });

  loading.textContent = "";
}

// ================= INIT =================
async function init() {
  setProgress(10, "Loading tokenizer…");
  await loadTokenizer();

  setProgress(40, "Loading ML model…");
  await loadModel();

  setProgress(70, "Loading recipes…");
  await loadChunks();

  setProgress(100, "Ready ✓");
  ready = true;

  setTimeout(hideLoader, 500);
}

searchBtn.onclick = recommend;
ingredientsInput.addEventListener("keydown", e => {
  if (e.key === "Enter") recommend();
});

init();
