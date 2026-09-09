# 1. Install dependencies
pip install -r requirements.txt

# 2. Ingest corpus
python main.py --ingest-all

# 3. Query the Personal Brain
python main.py --query "Why did I move from Drosophila to yeast?"

# 4. Run the 12 Benchmark Questions
python main.py --benchmark --output-json benchmark_results.json

# 5. Run Automated Test Suite
python -m unittest tests/test_modules.py
