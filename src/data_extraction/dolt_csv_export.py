import os
import subprocess
import concurrent.futures
import time

# Config
DATABASES = [
    {
        'name': 'stocks',
        'path': os.path.join('data', 'raw', 'stocks'),
        'tables': ['dividend', 'ohlcv', 'split', 'symbol']
    },
    {
        'name': 'options',
        'path': os.path.join('data', 'raw', 'options'),
        'tables': ['option_chain', 'volatility_history']
    }
]

OUTPUT_DIR = os.path.join('csv')

# Functions
def export_table(db_config, table):
    repo_path = os.path.abspath(db_config['path'])
    output_dir = OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{table}.csv")
    
    if os.path.exists(output_path):
        print(f"⚠️  Removing existing file: {output_path}")
        os.remove(output_path)
    
    cmd = ["dolt", "table", "export", table, output_path]
    print(f"🚀 Exporting table '{table}' from repository '{db_config['name']}'...")
    
    try:
        subprocess.run(cmd, cwd=repo_path, check=True)
        print(f"✅ Exported {table} to {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error exporting table {table}: {e}")
        return

def export_database(db_config):
    """Export all tables for a given database configuration."""
    for table in db_config['tables']:
        export_table(db_config, table)

# Main
def main():
    start_time = time.time()
    print("⏳ Starting Dolt native exports...\n")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(DATABASES)) as executor:
        futures = [executor.submit(export_database, db_config) for db_config in DATABASES]
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                print(f"❌ Database export generated an exception: {exc}")
    
    elapsed = time.time() - start_time
    print(f"\n🎉 All exports completed in {elapsed:.2f} seconds.")

if __name__ == '__main__':
    main()
