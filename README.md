# 🧬 SPARCe: Sparse Representation–Attribution Correlator ✨

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

> *Where single cells meet sparse autoencoders and magic happens!* 🪄

**SPARCe** is your friendly neighborhood data science wizard for diving deep into single-cell RNA sequencing data from the CZ CELLxGENE Census. Think of it as a Swiss Army knife for understanding what makes cells tick, powered by sparse representations and sprinkled with a generous dose of attribution analysis! 🔬⚡

## 🚀 What Makes SPARCe Special?

- **⚡ UV-Powered Speed**: Lightning-fast Python package management that doesn't make you wait for coffee ☕
- **🌍 CELLxGENE Universe Access**: Direct pipeline to the vast cosmos of single-cell RNA-seq data
- **🔬 Smart Data Pipeline**: Because messy data is nobody's friend
- **📊 Jupyter Magic**: Interactive notebooks that make data exploration feel like play
- **🧠 SAE Feature Analysis**: Sparse autoencoders meet biological insights in the most delightful way
- **🎨 Beautiful Visualizations**: Seaborn plots so pretty they belong in an art gallery

## 🏃‍♀️ Quick Start (Because Who Has Time to Wait?)

### 📋 What You'll Need

- Python 3.13+ (the shiny new version! ✨)
- UV package manager (your new best friend 🤝)
- Internet connection (for downloading the entire universe of cells 🌌)
- A sense of adventure! 🗺️

### 🛠️ Installation (It's Easier Than Making Toast!)

1. **Get UV on your side** (if you haven't already joined the UV revolution):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   export PATH="$HOME/.local/bin:$PATH"
   # 🎉 Welcome to the UV family!
   ```

2. **Clone this beauty and make it yours**:
   ```bash
   git clone <repository-url>
   cd AstroBio
   # 🏠 Welcome home!
   ```

3. **Let UV work its magic** (sit back and watch the dependencies dance):
   ```bash
   uv sync
   # ✨ *Poof!* Everything you need is now ready
   ```

4. **Take it for a test drive**:
   ```bash
   uv run scripts/simple_cellxgene_test.py
   # 🚗💨 Vroom vroom! Let's see if everything works!
   ```

### 🎯 Ready, Set, Science!

- **🧪 Test the waters**: `uv run scripts/simple_cellxgene_test.py`
- **📥 Download the cellular cosmos**: `uv run scripts/download_cellxgene_data.py`
- **🚀 Launch into analysis**: `uv run jupyter lab notebooks/`
- **🔍 Decode your SAE features**: `uv run scripts/sae_interpretation_analysis.py --input-dir results/your_analysis --output-dir results/interpretation_magic`

## 📝 Adventure Checklist

 - [x] 🎉 Set up UV environment management (Done and dusted!)
 - [x] 🔗 Add CELLxGENE Census integration (Connected to the mothership!)
 - [x] 📥 Create data download scripts (Data flows like a river!)
 - [x] 🧠 Build SAE interpretation tools (Mind-reading for features!)
 - [x] 🎨 Generate beautiful visualizations (Art meets science!)
 - [ ] 🏗️ Structure src better (Organization is key!)
 - [ ] 🔍 Add ruff/linter settings (Clean code, happy life!)
 - [ ] 📊 Improve org data (More structure, more fun!)
 - [ ] 🤖 Autorunning test (Because automation is magic!)

## 🎭 Scripts Overview (Your Digital Toolbox!)

### 🧪 Data Download Scripts (The Data Hunters!)

#### `scripts/simple_cellxgene_test.py`
*The friendly neighborhood scout* 🕵️‍♀️
- Tests if your packages are playing nice together
- Creates sample data faster than you can say "mitochondria"
- Connects to CELLxGENE Census (with timeout protection!)
- Generates mock data if the census is being shy
- **Cast the spell**: `uv run scripts/simple_cellxgene_test.py`

#### `scripts/download_cellxgene_data.py`
*The data wrangler extraordinaire* 🤠
- Queries cell metadata like a detective
- Downloads gene expression data with style
- Filters for the coolest cells (neurons, microglia, you name it!)
- Saves everything in multiple formats (because options are good!)
- **Summon the data**: `uv run scripts/download_cellxgene_data.py`

#### `scripts/sae_interpretation_analysis.py`
*The feature whisperer* 🔮
- Decodes what your SAE features actually mean
- Creates stunning visualizations that tell stories
- Computes feature specificity scores (how picky are your features?)
- Maps attribution leakage (where do biological signals spread?)
- **Unlock the secrets**: `uv run scripts/sae_interpretation_analysis.py --input-dir results/your_analysis`

### 📁 Data Treasures (What You'll Find in Your Digital Chest!)

The scripts sprinkle their magic across `data/raw/`:
- 📊 `test_data.csv`: Your practice playground
- 🧬 `cell_metadata_*.csv`: The cellular who's who directory
- 🗂️ `expression_data_*.h5ad`: Gene expression goldmines in AnnData format
- 📋 `gene_info.csv`: Your gene annotation cheat sheet
- 📈 `download_summary.txt`: The "what just happened?" file

### 🎨 Interpretation Artworks (Beauty Meets Science!)

When you run the interpretation analysis, you'll get:
- 🎯 `feature_specificity_analysis.png`: How selective are your features?
- 🌊 `attribution_leakage_analysis.png`: Where do biological signals flow?
- 🔥 `feature_attribution_heatmap.png`: The grand overview of everything
- 🗺️ `feature_landscape_analysis.png`: A PCA journey through feature space
- 📝 `interpretation_report.md`: Your personalized feature biography

## ⚡ UV Environment Details (Your Python Superpower!)

This project rides the UV wave for blazing-fast Python package management that makes conda look like it's stuck in traffic! 🚀

**🔬 Core Data Science Arsenal:**
- `pandas`, `numpy`, `scipy`: The holy trinity of data manipulation
- `matplotlib`, `seaborn`: For plots that make Excel jealous
- `scikit-learn`: Machine learning that actually learns
- `jupyterlab`, `notebook`: Your interactive playground

**🧬 Single-cell Superpowers:**
- `cellxgene-census`: Your VIP pass to cellular data heaven
- `anndata`: Because annotated data is happy data
- `scanpy`: The Swiss Army knife of single-cell analysis

**🛠️ Development Magic:**
- `ruff`: Lightning-fast linting (faster than you can say "PEP 8")
- `pytest`: Testing that doesn't test your patience
- `python-dotenv`: Environment variables made easy

### 🎮 UV Commands (Your Cheat Codes!)

```bash
# 🔄 Sync your universe with reality
uv sync

# ➕ Add a new package to your arsenal
uv add package-name

# 🏃‍♂️ Run with the power of UV
uv run script.py

# 🧑‍💻 Get development superpowers
uv sync --dev

# 👀 See what's in your toolkit
uv pip list

# ⬆️ Level up everything
uv lock --upgrade
```

## 🆘 Troubleshooting (When Things Get Spicy!) 🌶️

### 🐛 Common Plot Twists

1. **😱 TileDB Context Error**: When the Census decides to play hard to get:
   - Usually just network hiccups or system quirks
   - Our test script is prepared with backup mock data (because we plan ahead!)
   - Try a different network or take a coffee break ☕

2. **💾 Disk Space Drama**: When your hard drive throws a tantrum:
   - Keep an eye on space with `df -h` (knowledge is power!)
   - Clean house with `uv clean` (Marie Kondo for Python!)
   - Start small with test datasets (baby steps!)

3. **📦 Import Errors**: When packages refuse to cooperate:
   - Double-check your UV environment is active
   - Run `uv sync` to restore harmony
   - Verify Python version compatibility (3.13+ is our happy place!)

### 🚀 Performance Pro Tips

- Use `UV_LINK_MODE=copy` for shared filesystems (sharing is caring!)
- Set reasonable timeouts (patience is a virtue, but not infinite!)
- Cache your downloads (because time is precious!)
- Start with small datasets for testing (crawl before you sprint!)

## 🏗️ Project Organization (A Beautiful Mind Palace!)

```
├── LICENSE            <- MIT license (sharing is caring! 🤝)
├── README.md          <- You are here! 📍
├── pyproject.toml     <- The magic configuration scroll ✨
├── data
│   ├── external       <- Third-party treasures 💎
│   ├── interim        <- Work-in-progress masterpieces 🎨
│   ├── processed      <- Polished data diamonds 💍
│   └── raw            <- Fresh-from-the-source data (CELLxGENE bounty!) 🌊
│
├── models             <- Where AI dreams come true 🤖
├── notebooks          <- Interactive wonderlands 📓✨
├── results            <- Your scientific discoveries! 🏆
│   ├── figures        <- Pretty pictures that tell stories 🖼️
│   └── sae_interpretation <- Feature interpretation magic! 🔮
│
├── scripts            <- Your command-line superpowers 🦸‍♀️
│   ├── download_cellxgene_data.py      <- The data summoner 🧙‍♂️
│   ├── simple_cellxgene_test.py        <- The environment validator 🛡️
│   ├── sae_attribution_pipeline.py    <- The full analysis wizard 🪄
│   └── sae_interpretation_analysis.py <- The feature decoder 🔍
│
└── src                <- Your custom code sanctuary 🏛️
    ├── __init__.py                    <- The Python module maker ⚙️
    ├── feature_attribution_analysis.py <- Statistical magic toolkit 📊
    ├── models.py                      <- AI architecture blueprints 🏗️
    ├── sae_feature_interpretation.py  <- Feature storytelling engine 📚
    └── utils.py                       <- Utility spells collection ✨
```

## 🎬 Getting Started with Analysis (Your Scientific Adventure Begins!)

1. **🧪 Test your scientific setup**:
   ```bash
   uv run scripts/simple_cellxgene_test.py
   # Is everything working? Let's find out! 🤞
   ```

2. **📊 Summon your data**:
   ```bash
   uv run scripts/download_cellxgene_data.py
   # Downloading the cellular universe... ⬇️🌌
   ```

3. **🚀 Launch into interactive exploration**:
   ```bash
   uv run jupyter lab notebooks/
   # Time to play with data! 🎮
   ```

4. **📈 Run the full SAE analysis pipeline**:
   ```bash
   uv run scripts/sae_attribution_pipeline.py --sample-size 1000 --output-dir results/my_awesome_analysis
   # Let the magic happen! ✨🔬
   ```

5. **🎨 Create beautiful interpretation visualizations**:
   ```bash
   uv run scripts/sae_interpretation_analysis.py --input-dir results/my_awesome_analysis --output-dir results/interpretation_art
   # Transform data into art! 🎨📊
   ```

6. **🕵️‍♀️ Investigate your results**:
   ```bash
   ls -la data/raw/
   cat data/raw/download_summary.txt
   # What treasures did we find? 💎
   ```

## 🤝 Contributing (Join Our Scientific Adventure!)

Want to make SPARCe even more awesome? We'd love to have you aboard! 🎉

1. 🍴 Fork the repository (make it yours!)
2. 🌟 Create a feature branch (`git checkout -b feature/amazing-discovery`)
3. 🧪 Make your changes and test with UV: `uv run scripts/simple_cellxgene_test.py`
4. 💾 Commit your brilliance (`git commit -m 'Add mind-blowing feature'`)
5. 🚀 Push to your branch (`git push origin feature/amazing-discovery`)
6. 🎯 Open a Pull Request and share your magic!

*Every contribution makes the single-cell analysis world a little bit brighter!* ✨

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for all the legal details. TL;DR: Have fun with it! 🎉

---

*Built with ❤️, lots of ☕, and a healthy dose of 🧬 curiosity. Happy analyzing!* 🚀
