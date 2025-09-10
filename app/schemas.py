from pydantic import BaseModel
from typing import List, Optional


class ProductItem(BaseModel):
	name: str
	url: str
	price: Optional[str] = None
	snippet: Optional[str] = None
	image_url: Optional[str] = None


class AnalyzeResponse(BaseModel):
	diagnosis: str
	skin_type: str
	explanation: str
	products: List[ProductItem]
