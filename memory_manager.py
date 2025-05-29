#!/usr/bin/env python3
"""
Agent Memory Manager - GitHub Native
------------------------------------
This script processes pending embedding requests, generates sentence embeddings,
and stores them in a central JSON-based vector store within the agent-memory repository.
It's designed to be run as a GitHub Action.

Expected environment variables:
- GITHUB_TOKEN: A GitHub Personal Access Token with repo scope.
"""

import os
import json
import time
import requests
import base64
from datetime import datetime, timezone

# --- Configuration Constants ---
GITHUB_API_URL = "https://api.github.com"
OWNER = "zipaJopa"  # Your GitHub username or organization
AGENT_MEMORY_REPO_NAME = "agent-memory"
AGENT_MEMORY_REPO_FULL = f"{OWNER}/{AGENT_MEMORY_REPO_NAME}"

PENDING_EMBEDDINGS_PATH = "pending_embeddings"
PROCESSED_EMBEDDINGS_PATH = "processed_pending_embeddings" # Archive path
VECTOR_STORE_FILE_PATH = "vector_store.json"
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2' # From sentence-transformers

# --- GitHub Interaction Helper Class ---
class GitHubInteraction:
    def __init__(self, token):
        self.token = token
        self.headers = {
            "Authorization": f"token {self.token}",
            "Accept": "application/vnd.github.v3+json",
            "X-GitHub-Api-Version": "2022-11-28"
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)

    def _request(self, method, endpoint, data=None, params=None, max_retries=3, base_url=GITHUB_API_URL):
        url = f"{base_url}{endpoint}"
        for attempt in range(max_retries):
            try:
                response = self.session.request(method, url, params=params, json=data)
                
                if 'X-RateLimit-Remaining' in response.headers and int(response.headers['X-RateLimit-Remaining']) < 10:
                    reset_time = int(response.headers.get('X-RateLimit-Reset', time.time() + 60))
                    sleep_duration = max(0, reset_time - time.time()) + 5
                    print(f"Rate limit low. Sleeping for {sleep_duration:.2f} seconds.")
                    time.sleep(sleep_duration)

                response.raise_for_status()
                return response.json() if response.content else {}
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 403 and "rate limit exceeded" in e.response.text.lower():
                    reset_time = int(e.response.headers.get('X-RateLimit-Reset', time.time() + 60 * (attempt + 1)))
                    sleep_duration = max(0, reset_time - time.time()) + 5
                    print(f"Rate limit exceeded. Retrying in {sleep_duration:.2f}s (attempt {attempt+1}/{max_retries}).")
                    time.sleep(sleep_duration)
                    continue
                elif e.response.status_code == 404 and method == "GET":
                    # print(f"Resource not found (404) at {url}") # Common, so can be noisy
                    return None # Indicate not found
                elif e.response.status_code == 422: # Unprocessable Entity, e.g. commit to non-existent branch
                     print(f"Unprocessable Entity (422) at {url}. Message: {e.response.text}")
                     # This might indicate trying to update a file that changed, or other commit issues.
                     # For get_file_content_and_sha, this is handled by returning None for sha if file not found.
                else:
                    print(f"GitHub API request failed ({method} {url}): {e.response.status_code} - {e.response.text}")
                
                if attempt == max_retries - 1:
                    raise
            except requests.exceptions.RequestException as e:
                print(f"GitHub API request failed ({method} {url}): {e}")
                if attempt == max_retries - 1:
                    raise
            time.sleep(2 ** attempt)
        return {}


    def list_files(self, repo_full_name, path):
        print(f"Listing files in {repo_full_name}/{path}...")
        endpoint = f"/repos/{repo_full_name}/contents/{path}"
        response_data = self._request("GET", endpoint)
        if response_data and isinstance(response_data, list):
            return [item for item in response_data if item.get("type") == "file"]
        return []

    def get_file_content_and_sha(self, repo_full_name, file_path):
        # print(f"Getting content and SHA for {repo_full_name}/{file_path}...")
        endpoint = f"/repos/{repo_full_name}/contents/{file_path}"
        file_data = self._request("GET", endpoint)
        if file_data and "content" in file_data and "sha" in file_data:
            content = base64.b64decode(file_data["content"]).decode('utf-8')
            return content, file_data["sha"]
        return None, None

    def create_or_update_file(self, repo_full_name, file_path, content_str, commit_message, current_sha=None, branch="main"):
        print(f"Creating/Updating file '{file_path}' in {repo_full_name}...")
        encoded_content = base64.b64encode(content_str.encode('utf-8')).decode('utf-8')
        
        payload = {
            "message": commit_message,
            "content": encoded_content,
            "branch": branch
        }
        if current_sha:
            payload["sha"] = current_sha
        
        endpoint = f"/repos/{repo_full_name}/contents/{file_path}"
        response = self._request("PUT", endpoint, data=payload)
        if response and "content" in response and "sha" in response["content"]:
            print(f"Successfully committed '{file_path}' (New SHA: {response['content']['sha']})")
            return response["content"]["sha"]
        else:
            print(f"Failed to commit '{file_path}'. Response: {response}")
            return None

    def delete_file(self, repo_full_name, file_path, sha, commit_message, branch="main"):
        print(f"Deleting file '{file_path}' from {repo_full_name}...")
        payload = {
            "message": commit_message,
            "sha": sha,
            "branch": branch
        }
        endpoint = f"/repos/{repo_full_name}/contents/{file_path}"
        # A 200 OK with empty content means success for delete.
        # A 204 No Content also means success for some delete operations.
        # For file deletion, GitHub API returns 200 OK with commit info.
        response = self._request("DELETE", endpoint, data=payload) 
        if response is not None: # Check if response is not None (which _request returns on 404 for GET)
             # For DELETE, a successful response might be an empty dict or contain commit info
            print(f"Successfully deleted '{file_path}'.")
            return True
        print(f"Failed to delete '{file_path}'.")
        return False

# --- Memory Management Logic ---
class MemoryManager:
    def __init__(self, github_token):
        self.gh = GitHubInteraction(github_token)
        try:
            from sentence_transformers import SentenceTransformer
            self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
            print(f"SentenceTransformer model '{EMBEDDING_MODEL_NAME}' loaded.")
        except ImportError:
            print("Error: sentence-transformers library not found. Please install it: pip install sentence-transformers")
            self.embedding_model = None
        except Exception as e:
            print(f"Error loading SentenceTransformer model '{EMBEDDING_MODEL_NAME}': {e}")
            self.embedding_model = None

    def _generate_embedding(self, text):
        if not self.embedding_model:
            print("Embedding model not available. Cannot generate embeddings.")
            return None
        try:
            embedding = self.embedding_model.encode(text, convert_to_tensor=False) # Get numpy array
            return embedding.tolist() # Convert to list for JSON serialization
        except Exception as e:
            print(f"Error generating embedding for text '{text[:50]}...': {e}")
            return None

    def _load_vector_store(self):
        print(f"Loading vector store from {VECTOR_STORE_FILE_PATH}...")
        content_str, sha = self.gh.get_file_content_and_sha(AGENT_MEMORY_REPO_FULL, VECTOR_STORE_FILE_PATH)
        if content_str:
            try:
                data = json.loads(content_str)
                # Basic validation
                if "metadata" in data and "embeddings" in data and isinstance(data["embeddings"], dict):
                    print(f"Vector store loaded successfully. Contains {len(data['embeddings'])} items.")
                    return data, sha
                else:
                    print(f"Warning: {VECTOR_STORE_FILE_PATH} has invalid structure. Initializing new store.")
            except json.JSONDecodeError:
                print(f"Error: Could not parse JSON from {VECTOR_STORE_FILE_PATH}. Initializing new store.")
        
        # If file not found, or invalid, initialize a new structure
        print(f"{VECTOR_STORE_FILE_PATH} not found or invalid. Initializing new vector store.")
        return {
            "metadata": {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "last_updated": datetime.now(timezone.utc).isoformat(),
                "model_name": EMBEDDING_MODEL_NAME,
                "total_items": 0
            },
            "embeddings": {}
        }, None # No SHA for a new file or if existing is invalid

    def _save_vector_store(self, vector_store_data, current_sha):
        print("Saving updated vector store...")
        vector_store_data["metadata"]["last_updated"] = datetime.now(timezone.utc).isoformat()
        vector_store_data["metadata"]["total_items"] = len(vector_store_data["embeddings"])
        
        content_str = json.dumps(vector_store_data, indent=2)
        commit_message = f"feat: Update vector store - {vector_store_data['metadata']['total_items']} items"
        
        new_sha = self.gh.create_or_update_file(
            AGENT_MEMORY_REPO_FULL, 
            VECTOR_STORE_FILE_PATH, 
            content_str, 
            commit_message,
            current_sha=current_sha
        )
        return new_sha is not None

    def _archive_processed_file(self, file_info):
        pending_file_path = file_info["path"]
        pending_file_sha = file_info["sha"]
        file_name = file_info["name"]

        print(f"Archiving processed file: {pending_file_path}")
        
        # 1. Get content of the file to be archived
        content_str, _ = self.gh.get_file_content_and_sha(AGENT_MEMORY_REPO_FULL, pending_file_path)
        if content_str is None:
            print(f"Error: Could not get content of {pending_file_path} for archiving. Skipping archival.")
            return False

        # 2. Create the file in the archive path
        archive_file_path = f"{PROCESSED_EMBEDDINGS_PATH}/{file_name}"
        archive_commit_message = f"chore: Archive processed embedding request {file_name}"
        
        # Check if file already exists in archive (e.g. from a previous partial run)
        # For simplicity, we'll try to create/update. If it fails, it's an issue.
        # A more robust way would be to get SHA if exists.
        _, archive_sha = self.gh.get_file_content_and_sha(AGENT_MEMORY_REPO_FULL, archive_file_path)

        created_in_archive_sha = self.gh.create_or_update_file(
            AGENT_MEMORY_REPO_FULL,
            archive_file_path,
            content_str,
            archive_commit_message,
            current_sha=archive_sha
        )

        if not created_in_archive_sha:
            print(f"Error: Failed to create {archive_file_path} in archive. Skipping deletion of original.")
            return False

        # 3. Delete the original file from pending_embeddings
        delete_commit_message = f"chore: Remove processed embedding request {file_name} from pending"
        deleted_successfully = self.gh.delete_file(
            AGENT_MEMORY_REPO_FULL,
            pending_file_path,
            pending_file_sha, # SHA of the file in the pending_embeddings path
            delete_commit_message
        )
        
        if deleted_successfully:
            print(f"Successfully archived {file_name} to {archive_file_path}")
            return True
        else:
            print(f"Error: Failed to delete original file {pending_file_path} after archiving.")
            # This is a problematic state: file copied to archive but not deleted from pending.
            # Manual intervention might be needed, or next run might re-process (but ID check should prevent duplication in store).
            return False

    def process_pending_embeddings(self):
        if not self.embedding_model:
            print("Cannot process embeddings: embedding model not loaded.")
            return

        print(f"\n--- Starting Embedding Processing Cycle: {datetime.now(timezone.utc).isoformat()} ---")
        pending_files = self.gh.list_files(AGENT_MEMORY_REPO_FULL, PENDING_EMBEDDINGS_PATH)
        
        if not pending_files:
            print("No pending embedding requests found.")
            print("--- Embedding Processing Cycle Finished ---")
            return

        print(f"Found {len(pending_files)} pending request(s).")
        
        vector_store_data, vector_store_sha = self._load_vector_store()
        store_was_modified = False

        for file_info in pending_files:
            file_path = file_info["path"]
            file_name = file_info["name"]
            print(f"\nProcessing pending file: {file_path}...")
            
            content_str, _ = self.gh.get_file_content_and_sha(AGENT_MEMORY_REPO_FULL, file_path)
            if not content_str:
                print(f"Error: Could not retrieve content for {file_path}. Skipping.")
                continue
            
            try:
                pending_data = json.loads(content_str)
            except json.JSONDecodeError:
                print(f"Error: Could not parse JSON from {file_path}. Skipping.")
                # Consider moving to a "failed_parsing" directory
                continue

            items_to_embed = pending_data.get("items", [])
            task_id = pending_data.get("task_id", "unknown_task")
            task_type = pending_data.get("task_type", "unknown_type")
            source_ref = pending_data.get("source_ref", "unknown_source")
            
            if not isinstance(items_to_embed, list):
                print(f"Warning: 'items' field in {file_path} is not a list. Skipping.")
                continue

            processed_item_count_for_file = 0
            for item in items_to_embed:
                item_id = item.get("id")
                text_to_embed = item.get("text")

                if not item_id or not text_to_embed:
                    print(f"Warning: Skipping item in {file_name} due to missing 'id' or 'text'. Item: {item}")
                    continue

                if item_id in vector_store_data["embeddings"]:
                    print(f"Info: Item ID '{item_id}' already exists in vector store. Skipping embedding to avoid duplication.")
                    processed_item_count_for_file +=1 # Count as processed for archival purposes
                    continue 
                    # Or, implement update logic if needed:
                    # print(f"Info: Item ID '{item_id}' already exists. Updating.")

                embedding_vector = self._generate_embedding(text_to_embed)
                if embedding_vector:
                    embedding_record = {
                        "vector": embedding_vector,
                        "text_preview": text_to_embed[:100] + ("..." if len(text_to_embed) > 100 else ""),
                        "source_ref": source_ref,
                        "task_id": task_id,
                        "task_type": task_type,
                        "original_metadata": item.get("metadata", {}),
                        "embedded_at": datetime.now(timezone.utc).isoformat(),
                        "embedding_model": EMBEDDING_MODEL_NAME
                    }
                    vector_store_data["embeddings"][item_id] = embedding_record
                    store_was_modified = True
                    processed_item_count_for_file +=1
                    print(f"  Embedded item ID: {item_id}")
                else:
                    print(f"  Failed to embed item ID: {item_id}. It will remain in the pending file.")
            
            # If all items in the file were processed (or skipped due to already existing), archive the file
            if processed_item_count_for_file == len(items_to_embed):
                if not self._archive_processed_file(file_info):
                    print(f"Warning: Failed to archive {file_name}. It might be reprocessed if script fails before vector store commit.")
                    # If archival fails, we might choose not to commit the vector store changes for this batch
                    # to ensure consistency, or log it as a critical error. For now, we proceed.
            else:
                print(f"Warning: Not all items in {file_name} were processed. File will remain in pending.")


        if store_was_modified:
            if self._save_vector_store(vector_store_data, vector_store_sha):
                print("Successfully updated and saved vector store.")
            else:
                print("CRITICAL ERROR: Failed to save updated vector store. Changes might be lost, and files may be reprocessed if archival already happened.")
        else:
            print("No new embeddings were added to the vector store in this cycle.")
        
        print(f"--- Embedding Processing Cycle Finished: {datetime.now(timezone.utc).isoformat()} ---")

# --- Retrieval Logic (Example - can be expanded or moved to a separate service/API) ---
# This is a very basic search, not efficient for large stores.
# A real system would use FAISS or a dedicated vector DB.
def search_memory_basic(query_text, top_n=5):
    manager = MemoryManager(os.getenv("GITHUB_TOKEN")) # Needs token for loading store
    if not manager.embedding_model:
        print("Cannot search: embedding model not loaded.")
        return []

    vector_store_data, _ = manager._load_vector_store()
    if not vector_store_data or not vector_store_data.get("embeddings"):
        print("Vector store is empty or not loaded.")
        return []

    query_vector = manager._generate_embedding(query_text)
    if query_vector is None:
        return []
    
    import numpy as np
    
    results = []
    for item_id, record in vector_store_data["embeddings"].items():
        item_vector = np.array(record["vector"])
        query_vector_np = np.array(query_vector)
        # Cosine similarity
        similarity = np.dot(item_vector, query_vector_np) / (np.linalg.norm(item_vector) * np.linalg.norm(query_vector_np))
        results.append({"id": item_id, "similarity": similarity, "preview": record["text_preview"], "source": record["source_ref"]})
    
    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results[:top_n]

if __name__ == "__main__":
    github_token = os.getenv("GITHUB_TOKEN")
    if not github_token:
        print("Error: GITHUB_TOKEN environment variable not set for MemoryManager.")
    else:
        manager = MemoryManager(github_token)
        manager.process_pending_embeddings()

        # Example of how search could be used (optional, for testing)
        # if manager.embedding_model: # Check if model loaded
        #     print("\n--- Example Search (if run directly) ---")
        #     sample_query = "AI agent automation trends"
        #     search_results = search_memory_basic(sample_query)
        #     if search_results:
        #         print(f"Top results for '{sample_query}':")
        #         for res in search_results:
        #             print(f"  ID: {res['id']}, Sim: {res['similarity']:.4f}, Preview: {res['preview']}")
        #     else:
        #         print(f"No search results for '{sample_query}' or store is empty.")

```
**Note on Dependencies for `memory_manager.py`:**
This script will require the following Python libraries:
- `requests`
- `sentence-transformers`
- `numpy` (usually installed as a dependency of `sentence-transformers`)

You would typically have a `requirements.txt` in the `agent-memory` repository like:
```
requests
sentence-transformers
numpy
```
And the GitHub Action workflow would install these.
```
