import os
import pymongo
from fastapi import FastAPI
import nest_asyncio
from bson import ObjectId
import uvicorn
import networkx as nx
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from datetime import datetime, timedelta
from pymongo import MongoClient
import certifi

# ✅ Get MongoDB credentials securely from Render Environment Variables
MONGO_USERNAME = os.getenv("MONGO_USERNAME")
MONGO_PASSWORD = os.getenv("MONGO_PASSWORD")
MONGO_DB = os.getenv("MONGO_DB")

# ✅ Establish MongoDB Connection
uri = f"mongodb+srv://{MONGO_USERNAME}:{MONGO_PASSWORD}@cluster0.38cb2.mongodb.net/{MONGO_DB}?retryWrites=true&w=majority"
client = MongoClient(uri, tlsCAFile=certifi.where())
db = client[MONGO_DB]
cargo_collection = db["cargo"]

# ✅ Enable FastAPI inside Colab
nest_asyncio.apply()
app = FastAPI()

# 📌 AI Model for Smart Storage Placement
X_train = np.array([[5, 90, 30], [10, 50, 60], [3, 20, 15]])  # (Size, Priority, Expiry)
y_train = np.array(["Zone A", "Zone B", "Zone C"])  # Best Storage Zones
storage_model = DecisionTreeClassifier().fit(X_train, y_train)

# 📌 AI Model for Shortest Retrieval Path
G = nx.Graph()
G.add_edges_from([("Zone A", "Zone B"), ("Zone B", "Zone C"), ("Zone C", "Zone A")])

# ✅ API Endpoint: Add New Cargo
@app.post("/add_cargo/")
async def add_cargo(item: dict):
    item["expiry_date"] = datetime.now() + timedelta(days=item["expiry_days"])
    inserted_item = cargo_collection.insert_one(item)
    return {"message": "Cargo added", "id": str(inserted_item.inserted_id)}

# ✅ API Endpoint: Get All Cargo Items
@app.get("/get_cargo/")
async def get_cargo():
    items = []
    for item in cargo_collection.find():
        item["_id"] = str(item["_id"])
        items.append(item)
    return items

# ✅ API Endpoint: Delete Cargo Item
@app.delete("/delete_cargo/{item_id}")
async def delete_cargo(item_id: str):
    result = cargo_collection.delete_one({"_id": ObjectId(item_id)})
    if result.deleted_count:
        return {"message": "Cargo deleted"}
    return {"error": "Item not found"}

# ✅ AI Endpoint: Suggest Smart Storage Placement
@app.post("/smart_placement/")
async def smart_placement(item: dict):
    predicted_zone = storage_model.predict([[item["size"], item["priority"], item["expiry_days"]]])[0]
    return {"suggested_storage_zone": predicted_zone}

# ✅ AI Endpoint: Optimize Retrieval Path
@app.get("/shortest_path/{start_zone}/{end_zone}")
async def shortest_path(start_zone: str, end_zone: str):
    path = nx.shortest_path(G, source=start_zone, target=end_zone)
    return {"shortest_retrieval_path": path}

# ✅ API Endpoint: Identify Waste Items (Expired Cargo)
@app.get("/identify_waste/")
async def identify_waste():
    expired_items = []
    for item in cargo_collection.find({"expiry_date": {"$lt": datetime.now()}}):
        item["_id"] = str(item["_id"])
        expired_items.append(item)
    return expired_items

# ✅ API Endpoint: Predict Cargo Usage in Future Days
@app.get("/simulate_usage/{days}")
async def simulate_usage(days: int):
    future_date = datetime.now() + timedelta(days=days)
    expiring_items = []
    for item in cargo_collection.find({"expiry_date": {"$lt": future_date}}):
        item["_id"] = str(item["_id"])
        expiring_items.append(item)
    return {"expiring_items": expiring_items, "date_simulated": str(future_date)}

# ✅ Start FastAPI Server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
