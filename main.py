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
    items = list(cargo_collection.find({}))
    for item in items:
        item["_id"] = str(item["_id"])  
    return items

# 📌 **3️⃣ Smart Placement (Assign Cargo to Containers)**
@app.post("/api/placement")
async def placement_recommendation(item: dict):
    containers = list(storage_containers.find())  
    assigned_container = None  

    for container in containers:
        if (
            item["width"] <= container["width"]
            and item["depth"] <= container["depth"]
            and item["height"] <= container["height"]
        ):
            assigned_container = container["containerId"]
            break

    if assigned_container:
        return {"suggested_zone": assigned_container}
    else:
        return {"error": "No suitable storage container found"}

# 📌 **4️⃣ Retrieve Item**
@app.post("/api/retrieve")
async def retrieve_item(item_id: str):
    item = cargo_collection.find_one({"_id": ObjectId(item_id)})
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    cargo_collection.delete_one({"_id": ObjectId(item_id)})
    log_collection.insert_one({"item_id": item_id, "action": "retrieved", "timestamp": datetime.utcnow()})
    return {"message": "Item retrieved successfully"}

# 📌 **5️⃣ Waste Management & Disposal Plan**
@app.post("/api/waste/return-plan")
async def return_plan():
    expired_items = list(cargo_collection.find({"expiry_date": {"$lt": datetime.now()}}))

    return_manifest = {
        "disposalContainerId": "disposalA",
        "undockingDate": "2025-06-01",
        "returnItems": [{"itemId": item["_id"], "name": item["name"], "reason": "Expired"} for item in expired_items],
    }

    return {"returnPlan": return_manifest}

# 📌 **6️⃣ Time Simulation (Usage Tracking)**
@app.post("/api/simulate/day")
async def simulate_day():
    expiring_items = list(cargo_collection.find({"expiry_date": {"$lt": datetime.now() + timedelta(days=1)}}))
    for item in expiring_items:
        item["_id"] = str(item["_id"])

    return {"dateSimulated": str(datetime.now() + timedelta(days=1)), "expiring_items": expiring_items}

# 📌 **7️⃣ Import Items via CSV (Validation Added)**
@app.post("/api/import/items")
async def import_items(file: UploadFile = File(...)):
    try:
        df = pd.read_csv(file.file)
        if not all(col in df.columns for col in ["name", "size", "priority", "expiry_days"]):
            raise HTTPException(status_code=400, detail="Invalid CSV format")

        records = df.to_dict(orient="records")
        cargo_collection.insert_many(records)
        return {"message": "Items imported successfully"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to import CSV: {str(e)}")

# 📌 **8️⃣ Export Warehouse Arrangement**
@app.get("/api/export/arrangement")
async def export_arrangement():
    """Exports warehouse data as a downloadable CSV file."""
    items = list(cargo_collection.find({}, {"_id": 0}))
    df = pd.DataFrame(items)
    file_path = "warehouse_arrangement.csv"
    df.to_csv(file_path, index=False)

    return FileResponse(file_path, filename="warehouse_arrangement.csv", media_type="text/csv")

# 📌 **9️⃣ View Logs**
@app.get("/api/logs")
async def get_logs():
    logs = list(log_collection.find())
    for log in logs:
        log["_id"] = str(log["_id"])
    return logs

# ✅ **Start FastAPI**
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
