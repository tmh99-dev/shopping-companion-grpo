"""tools.product: extracted implementation; shared by training and evaluation."""
from __future__ import annotations

from typing import Any


class ProductViewMixin:
    @staticmethod
    def public_summary(row: dict[str, Any]) -> dict[str, Any]:
        product = row["product"]
        return {
            "product_id": str(row["id"]),
            "name": product.get("product_name", ""),
            "category": product.get("category", ""),
            "price": product.get("price"),
            "seller_id": str(product.get("seller_id", "")),
        }

    def view(self, arguments: dict[str, Any]) -> dict[str, Any]:
        product_id = str(arguments.get("product_id", ""))
        row = self.by_id.get(product_id)
        if row is None:
            return {"error": f"unknown product_id {product_id}"}
        product = row["product"]
        return {
            **self.public_summary(row),
            "attributes": product.get("attributes", {}),
            "options": product.get("options", []),
            "description_excerpt": str(product.get("description", ""))[:800],
            "ignored_arguments": sorted(set(arguments) - {"product_id"}),
            "notice": "Attributes describe the product; options are variants. An unselected option is not a purchased specification. A wanted feature without visible evidence does not satisfy the requirement.",
        }

    def invalid_variant_selection(self, selection: dict[str, Any]) -> bool:
        if "variant_index" not in selection:
            return False
        index = selection["variant_index"]
        options = self.by_id[str(selection["product_id"])]["product"].get("options", [])
        return (isinstance(index, bool) or not isinstance(index, int)
                or not isinstance(options, list) or not 0 <= index < len(options))
