import os
import pymongo
from fastapi import FastAPI, HTTPException
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

# ✅ FastAPI App
app = FastAPI()

# ✅ Enable CORS for Frontend Access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all domains
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods
    allow_headers=["*"],  # Allows all headers
)

# ✅ AI Model for Smart Storage Placement
X_train = np.array([[5, 90, 30], [10, 50, 60], [3, 20, 15]])
y_train = np.array(["Zone A", "Zone B", "Zone C"])
storage_model = DecisionTreeClassifier().fit(X_train, y_train)

# ✅ AI Model for Shortest Path
G = nx.Graph()
G.add_edges_from([("Zone A", "Zone B"), ("Zone B", "Zone C"), ("Zone C", "Zone A")])

# 📌 **1️⃣ Add Cargo**
@app.post("/api/add_cargo")
async def add_cargo(item: dict):
    item["expiry_date"] = datetime.now() + timedelta(days=item["expiry_days"])
    inserted_item = cargo_collection.insert_one(item)
    return {"message": "Cargo added", "id": str(inserted_item.inserted_id)}

# 📌 **2️⃣ Get All Cargo**
@app.get("/api/get_cargo")
async def get_cargo():
    try:
        items = list(cargo_collection.find({}))
        for item in items:
            item["_id"] = str(item["_id"])  # Convert ObjectId to string
        return items
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch cargo: {str(e)}")

# 📌 **3️⃣ Delete Cargo**
@app.delete("/api/delete_cargo/{item_id}")
async def delete_cargo(item_id: str):
    result = cargo_collection.delete_one({"_id": ObjectId(item_id)})
    if result.deleted_count:
        return {"message": "Cargo deleted successfully"}
    return {"error": "Item not found"}

# 📌 **4️⃣ Update Cargo**
@app.put("/api/update_cargo/{item_id}")
async def update_cargo(item_id: str, updated_data: dict):
    result = cargo_collection.update_one({"_id": ObjectId(item_id)}, {"$set": updated_data})
    if result.modified_count:
        return {"message": "Cargo updated successfully"}
    return {"error": "Item not found or no update applied"}

# 📌 **5️⃣ Smart Storage Placement**
@app.post("/api/placement")
async def placement_recommendation(item: dict):
    suggested_zone = storage_model.predict([[item["size"], item["priority"], item["expiry_days"]]])[0]
    return {"suggested_zone": suggested_zone}

# 📌 **6️⃣ Item Search**
@app.get("/api/search/{item_name}")
async def search_item(item_name: str):
    items = list(cargo_collection.find({"name": {"$regex": item_name, "$options": "i"}}))
    if not items:
        raise HTTPException(status_code=404, detail="Item not found")
    for item in items:
        item["_id"] = str(item["_id"])
    return items

# 📌 **7️⃣ Retrieve Item**
@app.post("/api/retrieve")
async def retrieve_item(item_id: str):
    item = cargo_collection.find_one({"_id": ObjectId(item_id)})
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    cargo_collection.delete_one({"_id": ObjectId(item_id)})
    log_collection.insert_one({"item_id": item_id, "action": "retrieved", "timestamp": datetime.utcnow()})
    return {"message": "Item retrieved successfully"}

# 📌 **8️⃣ Optimize Storage**
@app.post("/api/optimize_storage")
async def optimize_storage():
    items = list(cargo_collection.find().sort("priority", 1))
    if not items:
        raise HTTPException(status_code=404, detail="No items available for optimization")
    return {"rearrange_suggestions": [item["name"] for item in items[:3]]}

# 📌 **9️⃣ Identify Waste Items**
@app.get("/api/waste/identify")
async def identify_waste():
    expired_items = list(cargo_collection.find({"expiry_date": {"$lt": datetime.now()}}))
    for item in expired_items:
        item["_id"] = str(item["_id"])
    return expired_items

# 📌 **🔟 Return Plan for Waste**
@app.post("/api/waste/return-plan")
async def return_plan():
    expired_items = list(cargo_collection.find({"expiry_date": {"$lt": datetime.now()}}))
    return {"plan": f"Return {len(expired_items)} expired items to disposal zone"}

# 📌 **1️⃣1️⃣ Time Simulation**
@app.post("/api/simulate/day")
async def simulate_day():
    expiring_items = list(cargo_collection.find({"expiry_date": {"$lt": datetime.now() + timedelta(days=1)}}))
    for item in expiring_items:
        item["_id"] = str(item["_id"])
    return {"expiring_items": expiring_items}

# 📌 **1️⃣2️⃣ Import Items via CSV**
@app.post("/api/import/items")
async def import_items(file_path: str):
    df = pd.read_csv(file_path)
    records = df.to_dict(orient="records")
    cargo_collection.insert_many(records)
    return {"message": "Items imported successfully"}

# 📌 **1️⃣3️⃣ Export Warehouse Arrangement**
@app.get("/api/export/arrangement")
async def export_arrangement():
    """Exports warehouse data as a downloadable CSV file."""
    items = list(cargo_collection.find({}, {"_id": 0}))
    df = pd.DataFrame(items)
    file_path = "warehouse_arrangement.csv"
    df.to_csv(file_path, index=False)
    
    return FileResponse(file_path, filename="warehouse_arrangement.csv", media_type="text/csv")

# 📌 **1️⃣4️⃣ Logs API**
@app.get("/api/logs")
async def get_logs():
    logs = list(log_collection.find())
    for log in logs:
        log["_id"] = str(log["_id"])
    return logs

# ✅ **Start FastAPI**
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
