// ================= CONFIG =================
const TOTAL_CHUNKS = 8;          // поставь своё реальное число чанков
const USE_SINGLE_FILE = false;    // если сделаешь один файл — можно true
const RECIPES_FILE = "recipes_all.json";

const ONNX_URL =
  "https://huggingface.co/iammik3e/recsys-minilm/resolve/main/model.onnx";

// Diet exclusions (как раньше)
const DIET_EXCLUSIONS = {
  vegetarian: ["chicken", "beef", "pork", "fish", "meat", "bacon", "ham"],
  vegan: [
    "meat", "chicken", "beef", "fish",
    "milk", "cheese", "egg", "butter", "cream", "yogurt", "honey"
  ],
  nopork: ["pork", "bacon", "ham", "prosciutto", "salami", "pepperoni"]
};

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

const dietVegetarian = document.getElementById("dietVegetarian");
const dietVegan = document.getElementById("dietVegan");
const dietNoPork = document.getElementById("dietNoPork");

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
    token_type_ids: new ort.Tensor("int64", new BigInt64Array(128), [1, 128]),
  });

  const key = Object.keys(out)[0];
  const data = out[key].data;
  const hidden = out[key].dims[2];

  // mean pooling по токенам
  const emb = new Array(hidden).fill(0);
  for (let i = 0; i < hidden; i++) {
    for (let j = 0; j < tok.len; j++) emb[i] += data[j * hidden + i];
    emb[i] /= tok.len;
  }
  return emb;
}

// ================= DATA =================
async function loadChunks() {
  if (USE_SINGLE_FILE) {
    recipes = await fetch(RECIPES_FILE).then((r) => r.json());
    return;
  }

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
  let d = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) {
    d += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  return d / (Math.sqrt(na) * Math.sqrt(nb));
}

function normalizeText(s) {
  return (s || "").toLowerCase();
}

function recipeTextForFilter(r) {
  // фильтруем по ингредиентам + названию (надёжнее)
  const title = normalizeText(r.title);
  const ing = Array.isArray(r.ingredients) ? r.ingredients.join(" ") : "";
  return `${title} ${normalizeText(ing)}`;
}

function getRecipeImage(r) {
  // локальная картинка из папки images/
  return r.image ? `images/${r.image}` : "placeholder.jpg";
}

// ================= SEARCH =================
async function recommend() {
  const query = ingredientsInput.value.trim();
  if (!query) return;

  // exclusions
  let excluded = [];
  if (dietVegetarian.checked) excluded.push(...DIET_EXCLUSIONS.vegetarian);
  if (dietVegan.checked) excluded.push(...DIET_EXCLUSIONS.vegan);
  if (dietNoPork.checked) excluded.push(...DIET_EXCLUSIONS.nopork);

  loading.textContent = "Encoding…";
  results.innerHTML = "";

  // эмбедим запрос (как раньше)
  const userEmb = await embed(`Ingredients: ${query}`);

  loading.textContent = "Scoring…";

  // фильтрация по диетам
  const filtered = excluded.length
    ? recipes.filter((r) => {
        const text = recipeTextForFilter(r);
        return !excluded.some((e) => text.includes(e));
      })
    : recipes;

  const n = parseInt(topN.value, 10);

  const scored = filtered
    .map((r) => ({ r, score: cosine(userEmb, r.embedding) }))
    .sort((a, b) => b.score - a.score)
    .slice(0, n);

  // рендер
  results.innerHTML = "";
  scored.forEach(({ r, score }) => {
    const title = r.title || "Untitled recipe";
    const ing = Array.isArray(r.ingredients) ? r.ingredients.join(", ") : "";
    const instr = (r.instructions || "").trim();

    results.innerHTML += `
      <div class="recipe-card">
        <img
          src="${getRecipeImage(r)}"
          alt="${title}"
          onerror="this.onerror=null;this.src='placeholder.jpg';"
        >
        <div class="card-body">
          <h3>${title}</h3>
          <div class="score">Similarity: ${score.toFixed(3)}</div>
          <div class="ingredients">${ing}</div>

          ${instr ? `
            <details>
              <summary>Instructions</summary>
              <p class="instructions">${instr}</p>
            </details>
          ` : ""}
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
  loading.textContent = `Ready ✓ (${recipes.length} recipes)`;
}

searchBtn.onclick = recommend;

// enter = поиск
ingredientsInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") recommend();
});

init();
