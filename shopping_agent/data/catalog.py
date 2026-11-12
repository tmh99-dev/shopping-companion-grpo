"""data.catalog: extracted implementation; shared by training and evaluation."""
from __future__ import annotations

from pathlib import Path
from shopping_agent.tools.product import ProductViewMixin
from shopping_agent.tools.search import BM25Okapi
from shopping_agent.tools.search import ProductSearchMixin
from shopping_agent.tools.search import terms
from shopping_agent.tools.voucher import ProductQuoteMixin
import json


class ProductStore(ProductSearchMixin, ProductViewMixin, ProductQuoteMixin):
    def __init__(self, path: Path):
        self.rows = [json.loads(line) for line in path.open(encoding="utf-8")]
        self.by_id = {str(row["id"]): row for row in self.rows}
        if len(self.rows) != len(self.by_id):
            raise ValueError("Duplicate product IDs")
        self.index = BM25Okapi([terms(row.get("contents", "")) for row in self.rows])
