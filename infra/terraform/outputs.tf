output "artifact_registry_url" {
  description = "Google Artifact Registry repository URL"
  value       = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.metric_learning_repo.repository_id}"
}

output "gce_instance_name" {
  description = "Deployed GCE instance name"
  value       = google_compute_instance.app_vm.name
}

output "workload_identity_provider" {
  description = "Workload Identity Provider resource name for GitHub Actions"
  value       = google_iam_workload_identity_pool_provider.github_provider.name
}

output "github_cd_service_account" {
  description = "Service Account email for GitHub Actions CD"
  value       = google_service_account.github_cd_sa.email
}
