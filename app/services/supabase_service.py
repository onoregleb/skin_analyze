# app/services/supabase_service.py
from typing import Dict, Any, Optional, List
import json
from datetime import datetime
from supabase import create_client, Client
from pydantic import BaseModel
import os

# Модели для работы с базой данных
class SkinAnalysisJobCreate(BaseModel):
    job_id: str
    image_url: str
    user_text: Optional[str] = None
    mode: str = "basic"

class SkinAnalysisJobUpdate(BaseModel):
    status: Optional[str] = None
    progress: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    timings: Optional[Dict[str, float]] = None

class SkinAnalysisResult(BaseModel):
    job_id: str  # UUID из таблицы jobs
    diagnosis: Optional[str] = None
    skin_type: Optional[str] = None
    explanation: Optional[str] = None
    medgemma_summary: Optional[str] = None
    planning_data: Optional[Dict[str, Any]] = None
    final_result: Optional[Dict[str, Any]] = None

class RecommendedProduct(BaseModel):
    job_id: str  # UUID из таблицы jobs
    product_name: Optional[str] = None
    brand: Optional[str] = None
    description: Optional[str] = None
    price: Optional[float] = None
    currency: str = "USD"
    product_url: Optional[str] = None
    image_url: Optional[str] = None
    category: Optional[str] = None
    ingredients: Optional[List[str]] = None
    benefits: Optional[List[str]] = None
    suitable_for_skin_type: Optional[str] = None
    recommendation_order: int = 1

class SupabaseService:
    def __init__(self):
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_KEY")
        
        if not supabase_url or not supabase_key:
            raise ValueError("SUPABASE_URL and SUPABASE_KEY environment variables are required")
        
        self.client: Client = create_client(supabase_url, supabase_key)
    
    async def create_job(self, job_data: SkinAnalysisJobCreate) -> Dict[str, Any]:
        """Создать новую задачу анализа"""
        try:
            response = self.client.table("skin_analysis_jobs").insert({
                "job_id": job_data.job_id,
                "image_url": job_data.image_url,
                "user_text": job_data.user_text,
                "mode": job_data.mode,
                "status": "in_progress"
            }).execute()
            
            return response.data[0] if response.data else {}
        except Exception as e:
            print(f"Error creating job in Supabase: {e}")
            raise
    
    async def update_job(self, job_id: str, update_data: SkinAnalysisJobUpdate) -> Dict[str, Any]:
        """Обновить статус задачи"""
        try:
            update_dict = {}
            
            if update_data.status is not None:
                update_dict["status"] = update_data.status
            if update_data.progress is not None:
                update_dict["progress"] = json.dumps(update_data.progress)
            if update_data.error_message is not None:
                update_dict["error_message"] = update_data.error_message
            if update_data.timings is not None:
                update_dict["timings"] = json.dumps(update_data.timings)
            
            response = self.client.table("skin_analysis_jobs").update(
                update_dict
            ).eq("job_id", job_id).execute()
            
            return response.data[0] if response.data else {}
        except Exception as e:
            print(f"Error updating job in Supabase: {e}")
            raise
    
    async def get_job_by_job_id(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Получить задачу по job_id"""
        try:
            response = self.client.table("skin_analysis_jobs").select("*").eq(
                "job_id", job_id
            ).execute()
            
            return response.data[0] if response.data else None
        except Exception as e:
            print(f"Error getting job from Supabase: {e}")
            return None
    
    async def save_analysis_result(self, result_data: SkinAnalysisResult) -> Dict[str, Any]:
        """Сохранить результат анализа"""
        try:
            # Получаем UUID задачи по job_id
            job = await self.get_job_by_job_id(result_data.job_id)
            if not job:
                raise ValueError(f"Job with job_id {result_data.job_id} not found")
            
            job_uuid = job["id"]
            
            response = self.client.table("skin_analysis_results").insert({
                "job_id": job_uuid,
                "diagnosis": result_data.diagnosis,
                "skin_type": result_data.skin_type,
                "explanation": result_data.explanation,
                "medgemma_summary": result_data.medgemma_summary,
                "planning_data": json.dumps(result_data.planning_data) if result_data.planning_data else None,
                "final_result": json.dumps(result_data.final_result) if result_data.final_result else None
            }).execute()
            
            return response.data[0] if response.data else {}
        except Exception as e:
            print(f"Error saving analysis result to Supabase: {e}")
            raise
    
    async def save_recommended_products(self, products: List[RecommendedProduct]) -> List[Dict[str, Any]]:
        """Сохранить рекомендованные продукты"""
        if not products:
            return []
        
        try:
            # Получаем UUID задачи по job_id первого продукта
            job = await self.get_job_by_job_id(products[0].job_id)
            if not job:
                raise ValueError(f"Job with job_id {products[0].job_id} not found")
            
            job_uuid = job["id"]
            
            products_data = []
            for i, product in enumerate(products):
                products_data.append({
                    "job_id": job_uuid,
                    "product_name": product.product_name,
                    "brand": product.brand,
                    "description": product.description,
                    "price": product.price,
                    "currency": product.currency,
                    "product_url": product.product_url,
                    "image_url": product.image_url,
                    "category": product.category,
                    "ingredients": product.ingredients,
                    "benefits": product.benefits,
                    "suitable_for_skin_type": product.suitable_for_skin_type,
                    "recommendation_order": i + 1
                })
            
            response = self.client.table("recommended_products").insert(products_data).execute()
            return response.data or []
        except Exception as e:
            print(f"Error saving products to Supabase: {e}")
            raise
    
    async def get_full_analysis_result(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Получить полный результат анализа с продуктами"""
        try:
            # Получаем основную информацию о задаче
            job = await self.get_job_by_job_id(job_id)
            if not job:
                return None
            
            job_uuid = job["id"]
            
            # Получаем результат анализа
            result_response = self.client.table("skin_analysis_results").select("*").eq(
                "job_id", job_uuid
            ).execute()
            
            # Получаем продукты
            products_response = self.client.table("recommended_products").select("*").eq(
                "job_id", job_uuid
            ).order("recommendation_order").execute()
            
            result = result_response.data[0] if result_response.data else {}
            products = products_response.data or []
            
            return {
                "job": job,
                "result": result,
                "products": products
            }
        except Exception as e:
            print(f"Error getting full analysis result from Supabase: {e}")
            return None
    
    async def cleanup_old_jobs(self, days: int = 30):
        """Удалить старые задачи (опционально)"""
        try:
            from datetime import datetime, timedelta
            cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
            
            response = self.client.table("skin_analysis_jobs").delete().lt(
                "created_at", cutoff_date
            ).execute()
            
            return len(response.data) if response.data else 0
        except Exception as e:
            print(f"Error cleaning up old jobs: {e}")
            return 0

# Создание глобального экземпляра сервиса
supabase_service = SupabaseService()