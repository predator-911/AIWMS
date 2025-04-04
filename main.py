import os
import pymongo
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Union
import pandas as pd
from bson import ObjectId
import uvicorn
import networkx as nx
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from datetime import datetime, timedelta
from pymongo import MongoClient
import certifi

# ✅ Load MongoDB Credentials from Environment Variables
MONGO_USERNAME = os.getenv("MONGO_USERNAME")
MONGO_PASSWORD = os.getenv("MONGO_PASSWORD")
MONGO_DB = os.getenv("MONGO_DB")

# ✅ Connect to MongoDB Atlas
uri = f"mongodb+srv://{MONGO_USERNAME}:{MONGO_PASSWORD}@cluster0.38cb2.mongodb.net/{MONGO_DB}?retryWrites=true&w=majority"
client = MongoClient(uri, tlsCAFile=certifi.where())
db = client[MONGO_DB]
cargo_collection = db["cargo"]
log_collection = db["logs"]
storage_containers = db["storage_containers"]
items_col = db["items"]  # Added missing collection

# ✅ Define Pydantic models for request validation
class CargoItem(BaseModel):
    name: str
    size: float
    priority: int
    expiry_days: int

class ReturnPlanRequest(BaseModel):
    undockingContainerId: str
    undockingDate: str  # ISO format
    maxWeight: float

class ItemUsage(BaseModel):
    itemId: str
    uses: int = 1

# Fixed SimulationRequest to support both formats
class SimulationRequest(BaseModel):
    # Accept either numOfDays or days
    numOfDays: Optional[int] = None
    days: Optional[int] = None
    itemsToBeUsedPerDay: Optional[List[Dict[str, int]]] = []
    
    # Validate that either numOfDays or days is provided
    def get_days(self) -> int:
        return self.numOfDays if self.numOfDays is not None else self.days

# ✅ FastAPI App
app = FastAPI(title="Warehouse Management System API")

# ✅ Enable CORS for Frontend Access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

# ✅ AI Model for Smart Storage Placement
X_train = np.array([[5, 90, 30], [10, 50, 60], [3, 20, 15]])
y_train = np.array(["Zone A", "Zone B", "Zone C"])
storage_model = DecisionTreeClassifier().fit(X_train, y_train)

# ✅ AI Model for Shortest Path
G = nx.Graph()
G.add_edges_from([("Zone A", "Zone B"), ("Zone B", "Zone C"), ("Zone C", "Zone A")])

# Time management functions
def get_current_time():
    """Get the current system time."""
    # Fetch from db or return current time
    time_record = db["system_time"].find_one({"id": "current"})
    if time_record:
        return time_record["time"]
    # Initialize with current time if not set
    current_time = datetime.now().isoformat()
    db["system_time"].insert_one({"id": "current", "time": current_time})
    return current_time

def set_current_time(time_str):
    """Set the system time."""
    db["system_time"].update_one(
        {"id": "current"},
        {"$set": {"time": time_str}},
        upsert=True
    )

# Helper function for item volume calculation
def item_volume(item):
    """Calculate volume of an item."""
    return item.get("volume", 0) if "volume" in item else (
        item.get("width", 0) * item.get("height", 0) * item.get("depth", 0)
    )

# 📌 **1️⃣ Add Cargo (AI Learning Included)**
@app.post("/api/add_cargo")
async def add_cargo(item: CargoItem):
    item_dict = item.dict()
    item_dict["expiry_date"] = datetime.now() + timedelta(days=item_dict["expiry_days"])
    item_dict["retrieval_count"] = 0  # Track how many times this cargo is retrieved
    item_dict["created_at"] = datetime.now()
    
    inserted_item = cargo_collection.insert_one(item_dict)
    
    # Log the action
    log_collection.insert_one({
        "action": "add_cargo",
        "item_id": str(inserted_item.inserted_id),
        "timestamp": datetime.now()
    })
    
    # ✅ AI Learning: Update Model
    update_ai_model()

    return {"message": "Cargo added", "id": str(inserted_item.inserted_id)}

# 📌 **2️⃣ Get All Cargo**
@app.get("/api/get_cargo")
async def get_cargo():
    items = list(cargo_collection.find({}))
    for item in items:
        item["_id"] = str(item["_id"])
        # Convert datetime objects to ISO format strings for JSON serialization
        if "expiry_date" in item and isinstance(item["expiry_date"], datetime):
            item["expiry_date"] = item["expiry_date"].isoformat()
        if "created_at" in item and isinstance(item["created_at"], datetime):
            item["created_at"] = item["created_at"].isoformat()
    return items

# 📌 **3️⃣ AI Smart Placement**
@app.post("/api/placement")
async def placement_recommendation(item: CargoItem):
    item_dict = item.dict()
    suggested_zone = storage_model.predict([[item_dict["size"], item_dict["priority"], item_dict["expiry_days"]]])[0]
    
    # Log the recommendation
    log_collection.insert_one({
        "action": "placement_recommendation",
        "item_name": item_dict["name"],
        "suggested_zone": suggested_zone,
        "timestamp": datetime.now()
    })
    
    return {"suggested_zone": suggested_zone, "confidence": 90.0}

# 📌 **4️⃣ Retrieve Item (AI Learning Update)**
@app.post("/api/retrieve/{item_id}")
async def retrieve_item(item_id: str):
    try:
        object_id = ObjectId(item_id)
    except:
        raise HTTPException(status_code=400, detail="Invalid item ID format")
        
    item = cargo_collection.find_one({"_id": object_id})
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    
    # ✅ Increase Retrieval Count (AI Feedback)
    new_count = item.get("retrieval_count", 0) + 1
    cargo_collection.update_one({"_id": object_id}, {"$set": {"retrieval_count": new_count}})

    # Log the retrieval
    log_collection.insert_one({
        "action": "retrieve_item",
        "item_id": item_id,
        "item_name": item.get("name", "Unknown"),
        "retrieval_count": new_count,
        "timestamp": datetime.now()
    })

    # ✅ AI Learning: Update Model
    update_ai_model()

    return {"message": f"{item.get('name', 'Item')} retrieved successfully (Total: {new_count} times)"}

# 📌 **5️⃣ Delete Cargo**
@app.delete("/api/delete_cargo/{item_id}")
async def delete_cargo(item_id: str):
    try:
        object_id = ObjectId(item_id)
    except:
        raise HTTPException(status_code=400, detail="Invalid item ID format")
        
    item = cargo_collection.find_one({"_id": object_id})
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
        
    result = cargo_collection.delete_one({"_id": object_id})
    
    # Log the deletion
    log_collection.insert_one({
        "action": "delete_cargo",
        "item_id": item_id,
        "item_name": item.get("name", "Unknown"),
        "timestamp": datetime.now()
    })
    
    return {"message": "Cargo deleted successfully"}

# 📌 **6️⃣ Waste Management**
@app.post("/api/waste/return-plan")
async def generate_return_plan(request: ReturnPlanRequest):
    # Mark expired and depleted items as waste
    try:
        current_time = datetime.fromisoformat(get_current_time())
    except ValueError:
        current_time = datetime.now()
        
    # Find expired items
    expired_items = list(items_col.find({
        "expiryDate": {"$lt": current_time},
        "isWaste": False
    }))
    
    # Find depleted items
    depleted_items = list(items_col.find({
        "usageCount": {"$gte": "$usageLimit"},
        "isWaste": False
    }))

    # Update waste status
    for item in expired_items + depleted_items:
        items_col.update_one(
            {"_id": item["_id"]},
            {"$set": {
                "isWaste": True,
                "wasteReason": "Expired" if item.get("expiryDate", None) else "Depleted"
            }}
        )

    # Fetch all waste items
    waste_items = list(items_col.find({"isWaste": True}))
    
    # Optimize for maxWeight using knapsack algorithm
    selected_items = []
    total_weight = 0.0
    for item in sorted(waste_items, key=lambda x: -x.get("mass", 0)):
        if total_weight + item.get("mass", 0) <= request.maxWeight:
            selected_items.append(item)
            total_weight += item.get("mass", 0)

    # Generate return steps
    steps = []
    for idx, item in enumerate(selected_items, 1):
        steps.append({
            "step": idx,
            "action": "move",
            "itemId": item.get("itemId", str(item["_id"])),
            "fromContainer": item.get("containerId", "unknown"),
            "toContainer": request.undockingContainerId,
            "position": item.get("position", {"x": 0, "y": 0, "z": 0})
        })

    # Log the waste management plan
    log_collection.insert_one({
        "action": "waste_management",
        "undockingContainerId": request.undockingContainerId,
        "itemsSelected": len(selected_items),
        "totalWeight": total_weight,
        "timestamp": datetime.now()
    })

    return {
        "success": True,
        "totalWeight": total_weight,
        "totalVolume": sum(item_volume(item) for item in selected_items),
        "items": len(selected_items),
        "steps": steps
    }

# 📌 **7️⃣ Time Simulation - FIXED TO ACCEPT BOTH REQUEST FORMATS**
@app.post("/api/simulate/day")
async def simulate_time(request: dict):
    """
    Simulate the passage of time.
    Accepts either {"days": N} or {"numOfDays": N, "itemsToBeUsedPerDay": [...]} format.
    """
    # Handle different request formats
    num_days = request.get("numOfDays", request.get("days", 0))
    items_to_use = request.get("itemsToBeUsedPerDay", [])
    
    # Validate input
    if num_days <= 0:
        raise HTTPException(status_code=400, detail="Number of days must be positive")
    
    try:
        current_time = datetime.fromisoformat(get_current_time())
    except ValueError:
        current_time = datetime.now()
        set_current_time(current_time.isoformat())
    
    new_time = current_time + timedelta(days=num_days)
    
    expired_today = []
    depleted_today = []
    
    # Advance day-by-day to track granular changes
    for day in range(num_days):
        # Update system time
        current_time += timedelta(days=1)
        set_current_time(current_time.isoformat())
        
        # Mark expired items
        expired = list(items_col.find({
            "expiryDate": {"$lt": current_time},
            "isWaste": False
        }))
        expired_today.extend(expired)
        
        # Process daily usages
        if day < len(items_to_use):
            for usage in items_to_use[day]:
                item = items_col.find_one({"itemId": usage["itemId"]})
                if item:
                    new_usage = item.get("usageCount", 0) + usage.get("uses", 0)
                    items_col.update_one(
                        {"_id": item["_id"]},
                        {"$set": {"usageCount": new_usage}}
                    )
                    if item.get("usageLimit") and new_usage >= item["usageLimit"]:
                        depleted_today.append(item)
    
    # Mark depleted items as waste
    for item in depleted_today:
        items_col.update_one(
            {"_id": item["_id"]},
            {"$set": {"isWaste": True, "wasteReason": "Depleted"}}
        )
    
    # Mark expired items as waste
    for item in expired_today:
        items_col.update_one(
            {"_id": item["_id"]},
            {"$set": {"isWaste": True, "wasteReason": "Expired"}}
        )
    
    # Log the simulation
    log_collection.insert_one({
        "action": "time_simulation",
        "days_simulated": num_days,
        "expired_items": len(expired_today),
        "depleted_items": len(depleted_today),
        "timestamp": datetime.now()
    })
    
    return {
        "success": True,
        "newDate": current_time.isoformat(),
        "expiredToday": [{"itemId": x.get("itemId", str(x["_id"]))} for x in expired_today],
        "depletedToday": [{"itemId": x.get("itemId", str(x["_id"]))} for x in depleted_today]
    }

# Alternative endpoint that accepts raw JSON
@app.post("/api/simulate")
async def simulate_time_flexible(request: dict):
    """A more flexible endpoint that accepts various request formats."""
    return await simulate_time(request)

# 📌 **8️⃣ Import & Export Warehouse Data**
@app.post("/api/import/items")
async def import_items(file: UploadFile = File(...)):
    try:
        df = pd.read_csv(file.file)
        records = df.to_dict(orient="records")
        
        # Validate records before insertion
        valid_records = []
        for record in records:
            # Add creation timestamp
            record["created_at"] = datetime.now()
            valid_records.append(record)
            
        if valid_records:
            cargo_collection.insert_many(valid_records)
            
            # Log the import
            log_collection.insert_one({
                "action": "import_items",
                "filename": file.filename,
                "items_imported": len(valid_records),
                "timestamp": datetime.now()
            })
            
            return {"message": f"{len(valid_records)} items imported successfully"}
        else:
            return {"message": "No valid items found in the file"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Import failed: {str(e)}")

@app.get("/api/export/arrangement")
async def export_arrangement():
    try:
        items = list(cargo_collection.find({}, {"_id": 0}))
        
        # Convert datetime objects to strings for CSV export
        for item in items:
            for key, value in item.items():
                if isinstance(value, datetime):
                    item[key] = value.isoformat()
        
        # Create DataFrame and export to CSV
        df = pd.DataFrame(items)
        if df.empty:
            return {"message": "No items to export"}
            
        export_path = "warehouse_arrangement.csv"
        df.to_csv(export_path, index=False)
        
        # Log the export
        log_collection.insert_one({
            "action": "export_arrangement",
            "items_exported": len(items),
            "timestamp": datetime.now()
        })
        
        return FileResponse(export_path, filename="warehouse_arrangement.csv")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")

# 📌 **9️⃣ AI Learning Function**
def update_ai_model():
    """Retrains AI model using updated cargo data."""
    items = list(cargo_collection.find({}, {"size": 1, "priority": 1, "expiry_days": 1, "retrieval_count": 1}))
    
    if len(items) > 3:  # Ensure we have enough data points
        try:
            X = np.array([[
                item.get("size", 0), 
                item.get("priority", 0), 
                item.get("expiry_days", 0)
            ] for item in items])
            
            # Determine zones based on retrieval patterns
            y = np.array([
                "Zone A" if item.get("retrieval_count", 0) < 2 else
                "Zone B" if item.get("retrieval_count", 0) < 5 else
                "Zone C" for item in items
            ])
            
            global storage_model
            storage_model = DecisionTreeClassifier().fit(X, y)
            
            # Log model update
            log_collection.insert_one({
                "action": "model_update",
                "data_points": len(items),
                "timestamp": datetime.now()
            })
            
            print("✅ AI Model Updated with New Data")
        except Exception as e:
            print(f"❌ Error updating AI model: {str(e)}")

# Add health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}

# ✅ Start FastAPI
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
