resource "google_artifact_registry_repository" "metric_learning_repo" {
  location      = var.region
  repository_id = "metric-learning"
  description   = "Docker images for frontend (Streamlit), api-backend (FastAPI), and trainer-service (FastAPI/PyTorch)"
  format        = "DOCKER"
}
