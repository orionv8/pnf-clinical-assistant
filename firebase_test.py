import firebase_admin
from firebase_admin import firestore

firebase_admin.initialize_app()
_FIRESTORE_DB = firestore.client()

print("Firebase initialized successfully!")
