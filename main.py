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

# 📌 **1️⃣ Add Cargo (AI Learning Included)**
@app.post("/api/add_cargo")
async def add_cargo(item: dict):
    item["expiry_date"] = datetime.now() + timedelta(days=item["expiry_days"])
    item["retrieval_count"] = 0  # Track how many times this cargo is retrieved
    inserted_item = cargo_collection.insert_one(item)
    
    # ✅ AI Learning: Update Model
    update_ai_model()

    return {"message": "Cargo added", "id": str(inserted_item.inserted_id)}

# 📌 **2️⃣ Get All Cargo**
@app.get("/api/get_cargo")
async def get_cargo():
    items = list(cargo_collection.find({}))
    for item in items:
        item["_id"] = str(item["_id"])  
    return items

# 📌 **3️⃣ AI Smart Placement**
@app.post("/api/placement")
async def placement_recommendation(item: dict):
    suggested_zone = storage_model.predict([[item["size"], item["priority"], item["expiry_days"]]])[0]
    return {"suggested_zone": suggested_zone, "confidence": 100.0}

# 📌 **4️⃣ Retrieve Item (AI Learning Update)**
@app.post("/api/retrieve")
async def retrieve_item(item_id: str):
    item = cargo_collection.find_one({"_id": ObjectId(item_id)})
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    
    # ✅ Increase Retrieval Count (AI Feedback)
    new_count = item.get("retrieval_count", 0) + 1
    cargo_collection.update_one({"_id": ObjectId(item_id)}, {"$set": {"retrieval_count": new_count}})

    # ✅ AI Learning: Update Model
    update_ai_model()

    return {"message": f"{item['name']} retrieved successfully (Total: {new_count} times)"}

# 📌 **5️⃣ Delete Cargo (NEW)**
@app.delete("/api/delete_cargo/{item_id}")
async def delete_cargo(item_id: str):
    result = cargo_collection.delete_one({"_id": ObjectId(item_id)})
    if result.deleted_count:
        return {"message": "Cargo deleted successfully"}
    return {"error": "Item not found"}

# 📌 **6️⃣ Waste Management**
@app.post("/api/waste/return-plan")
async def return_plan():
    expired_items = list(cargo_collection.find({"expiry_date": {"$lt": datetime.now()}}))
    return {"plan": [{"itemId": item["_id"], "name": item["name"], "reason": "Expired"} for item in expired_items]}

# 📌 **7️⃣ Time Simulation**
@app.post("/api/simulate/day")
async def simulate_day():
    expiring_items = list(cargo_collection.find({"expiry_date": {"$lt": datetime.now() + timedelta(days=1)}}))
    return {"dateSimulated": str(datetime.now() + timedelta(days=1)), "expiring_items": expiring_items}

# 📌 **8️⃣ Import & Export Warehouse Data**
@app.post("/api/import/items")
async def import_items(file: UploadFile = File(...)):
    df = pd.read_csv(file.file)
    cargo_collection.insert_many(df.to_dict(orient="records"))
    return {"message": "Items imported successfully"}

@app.get("/api/export/arrangement")
async def export_arrangement():
    df = pd.DataFrame(list(cargo_collection.find({}, {"_id": 0})))
    df.to_csv("warehouse_arrangement.csv", index=False)
    return FileResponse("warehouse_arrangement.csv", filename="warehouse_arrangement.csv")

# 📌 **9️⃣ AI Learning Function**
def update_ai_model():
    """Retrains AI model using updated cargo data."""
    items = list(cargo_collection.find({}, {"size": 1, "priority": 1, "expiry_days": 1, "retrieval_count": 1}))
    
    if len(items) > 3:  # Ensure we have enough data points
        X = np.array([[item["size"], item["priority"], item["expiry_days"]] for item in items])
        y = np.array(["Zone A" if item["retrieval_count"] < 2 else "Zone B" for item in items])
        
        global storage_model
        storage_model = DecisionTreeClassifier().fit(X, y)
        print("✅ AI Model Updated with New Data")

# ✅ Start FastAPI
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
