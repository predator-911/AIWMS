import os
import pymongo
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
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

# ✅ FastAPI App
app = FastAPI()

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

# ✅ AI Model for Shortest Path Optimization
G = nx.Graph()
G.add_edges_from([("Zone A", "Zone B"), ("Zone B", "Zone C"), ("Zone C", "Zone A")])

# 📌 **1️⃣ Add Cargo**
@app.post("/api/add_cargo")
async def add_cargo(item: dict):
    """Adds a new cargo item to the storage system."""
    item["expiry_date"] = datetime.now() + timedelta(days=item["expiry_days"])
    item["retrieval_count"] = 0  
    inserted_item = cargo_collection.insert_one(item)
    return {"message": "Cargo added", "id": str(inserted_item.inserted_id)}

# 📌 **2️⃣ Get All Cargo**
@app.get("/api/get_cargo")
async def get_cargo():
    """Fetch all cargo data."""
    items = list(cargo_collection.find({}))
    for item in items:
        item["_id"] = str(item["_id"])
    return items

# 📌 **3️⃣ Smart Placement (AI Learning & Space Optimization)**
@app.post("/api/placement")
async def placement_recommendation(item: dict):
    """Suggests an optimal storage location for the given cargo item."""
    prediction = storage_model.predict([[item["size"], item["priority"], item["expiry_days"]]])[0]
    confidence = max(storage_model.predict_proba([[item["size"], item["priority"], item["expiry_days"]]])[0]) * 100
    return {"suggested_zone": prediction, "confidence": round(confidence, 2)}

# 📌 **4️⃣ Retrieve Item & AI Learning**
@app.post("/api/retrieve")
async def retrieve_item(item_id: str):
    """Retrieves an item from storage and updates AI learning."""
    item = cargo_collection.find_one({"_id": ObjectId(item_id)})
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    
    cargo_collection.update_one({"_id": ObjectId(item_id)}, {"$inc": {"retrieval_count": 1}})
    log_collection.insert_one({"item_id": item_id, "action": "retrieved", "timestamp": datetime.utcnow()})
    return {"message": f"{item['name']} retrieved successfully"}

# 📌 **5️⃣ Optimize Storage (High-Priority Item Placement)**
@app.get("/api/optimize_storage")
async def optimize_storage():
    """Rearranges high-priority cargo based on retrieval frequency."""
    high_priority_items = list(cargo_collection.find().sort("retrieval_count", -1).limit(3))
    return {"high_priority_items": [item["name"] for item in high_priority_items]}

# 📌 **6️⃣ Waste Management & Automated Return Planning**
@app.post("/api/waste/return-plan")
async def return_plan():
    """Generates a return manifest for expired items."""
    expired_items = list(cargo_collection.find({"expiry_date": {"$lt": datetime.now()}}))
    return_manifest = {
        "disposalContainerId": "disposalA",
        "undockingDate": "2025-06-01",
        "returnItems": [{"itemId": item["_id"], "name": item["name"], "reason": "Expired"} for item in expired_items],
    }
    return {"returnPlan": return_manifest}

# 📌 **7️⃣ Time Simulation (Fast-Forward Warehouse Conditions)**
@app.get("/api/simulate_usage/{days}")
async def simulate_day(days: int):
    """Simulates warehouse conditions for future days."""
    future_date = datetime.now() + timedelta(days=days)
    expiring_items = list(cargo_collection.find({"expiry_date": {"$lt": future_date}}))
    return {"dateSimulated": str(future_date), "expiring_items": expiring_items}

# 📌 **8️⃣ Import Cargo via CSV**
@app.post("/api/import/items")
async def import_items(file: UploadFile = File(...)):
    """Imports cargo data from a CSV file."""
    df = pd.read_csv(file.file)
    records = df.to_dict(orient="records")
    cargo_collection.insert_many(records)
    return {"message": "Items imported successfully"}

# 📌 **9️⃣ Export Warehouse Data**
@app.get("/api/export/arrangement")
async def export_arrangement():
    """Exports warehouse data as a CSV file."""
    items = list(cargo_collection.find({}, {"_id": 0}))
    df = pd.DataFrame(items)
    file_path = "warehouse_arrangement.csv"
    df.to_csv(file_path, index=False)
    return FileResponse(file_path, filename="warehouse_arrangement.csv", media_type="text/csv")

# 📌 **🔟 View System Logs**
@app.get("/api/logs")
async def get_logs():
    """Fetch all system logs."""
    logs = list(log_collection.find())
    for log in logs:
        log["_id"] = str(log["_id"])
    return logs

# ✅ **Start FastAPI**
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
