from pathlib import Path
import duckdb

root = Path(__file__).resolve().parent  # adjust if running elsewhere
pattern = str(root / "parquet" / "cleaned" / "*" / "*" / "*.parquet")

con = duckdb.connect("duckdb/app.db")
con.execute(f"""
CREATE VIEW IF NOT EXISTS cleaned_all AS
SELECT * FROM read_parquet('{pattern}');
""")
con.close()
