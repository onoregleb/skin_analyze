from pydantic import BaseModel
from typing import List, Optional, Dict


class ProductItem(BaseModel):
	name: str
	url: str
	price: Optional[str | float] = None
	snippet: Optional[str] = None
	image_url: Optional[str] = None


class AnalyzeResponse(BaseModel):
	diagnosis: str
	skin_type: str
	explanation: str
	products: List[ProductItem]
	# Extended fields for intermediate visibility
	medgemma_summary: str = ""
	tool_products: List[ProductItem] = []
	timings: Dict[str, float] = {}


class SkinAnalysisResponse(BaseModel):
	mode: str
	summary: str
	description: str
	timings: Dict[str, float] = {}
