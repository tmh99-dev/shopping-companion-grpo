"""Import a prepared Stage-2 dataset and validate its three data files."""
import argparse
import json
from pathlib import Path
import shutil
import tempfile

from shopping_agent.data.validation import validate_data

ROOT = Path(__file__).resolve().parents[2]




def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-source', type=Path, required=True)
    parser.add_argument('--test-source', type=Path, required=True)
    parser.add_argument('--products-source', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, default=ROOT/'data')
    args = parser.parse_args()
    sources = {'train.parquet': args.train_source, 'test.parquet': args.test_source,
               'products.jsonl': args.products_source}
    if any((args.output_root/name).exists() for name in sources):
        raise FileExistsError('Use a fresh output directory; existing data is preserved.')
    with tempfile.TemporaryDirectory() as tmp:
        staging = Path(tmp)
        for name in ('train.parquet', 'test.parquet'):
            shutil.copy2(sources[name], staging/name)
        report = validate_data(staging, args.products_source)
    args.output_root.mkdir(parents=True, exist_ok=True)
    for name, source in sources.items():
        shutil.copy2(source, args.output_root/name)
    print(json.dumps({'questions': report['counts'], 'products': report['products']}))


if __name__ == '__main__':
    main()
