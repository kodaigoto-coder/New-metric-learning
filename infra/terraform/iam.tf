# Service Account attached to the GCE VM
resource "google_service_account" "gce_runtime_sa" {
  account_id   = "metric-learning-gce-sa"
  display_name = "Metric Learning GCE Runtime Service Account"
}

resource "google_project_iam_member" "gce_ar_reader" {
  project = var.project_id
  role    = "roles/artifactregistry.reader"
  member  = "serviceAccount:${google_service_account.gce_runtime_sa.email}"
}

resource "google_project_iam_member" "gce_log_writer" {
  project = var.project_id
  role    = "roles/logging.logWriter"
  member  = "serviceAccount:${google_service_account.gce_runtime_sa.email}"
}

# Service Account for GitHub Actions CI/CD deployment
resource "google_service_account" "github_cd_sa" {
  account_id   = "metric-learning-gh-cd"
  display_name = "GitHub Actions CD Service Account"
}

resource "google_project_iam_member" "gh_ar_writer" {
  project = var.project_id
  role    = "roles/artifactregistry.writer"
  member  = "serviceAccount:${google_service_account.github_cd_sa.email}"
}

resource "google_project_iam_member" "gh_compute_admin" {
  project = var.project_id
  role    = "roles/compute.instanceAdmin.v1"
  member  = "serviceAccount:${google_service_account.github_cd_sa.email}"
}

resource "google_project_iam_member" "gh_iap_tunnel" {
  project = var.project_id
  role    = "roles/iap.tunnelResourceAccessor"
  member  = "serviceAccount:${google_service_account.github_cd_sa.email}"
}

resource "google_service_account_iam_member" "gh_act_as_gce_sa" {
  service_account_id = google_service_account.gce_runtime_sa.name
  role               = "roles/iam.serviceAccountUser"
  member             = "serviceAccount:${google_service_account.github_cd_sa.email}"
}

# Workload Identity Pool & OIDC Provider for keyless GitHub Actions auth
resource "google_iam_workload_identity_pool" "github_pool" {
  workload_identity_pool_id = "metric-learning-gh-pool"
  display_name              = "GitHub Actions Pool"
}

resource "google_iam_workload_identity_pool_provider" "github_provider" {
  workload_identity_pool_id          = google_iam_workload_identity_pool.github_pool.workload_identity_pool_id
  workload_identity_pool_provider_id = "github-oidc"
  display_name                       = "GitHub OIDC Provider"
  attribute_mapping = {
    "google.subject"       = "assertion.sub"
    "attribute.actor"      = "assertion.actor"
    "attribute.repository" = "assertion.repository"
  }
  attribute_condition = "assertion.repository == '${var.github_repository}'"
  oidc {
    issuer_uri = "https://token.actions.githubusercontent.com"
  }
}

resource "google_service_account_iam_member" "workload_identity_binding" {
  service_account_id = google_service_account.github_cd_sa.name
  role               = "roles/iam.workloadIdentityUser"
  member             = "principalSet://iam.googleapis.com/${google_iam_workload_identity_pool.github_pool.name}/attribute.repository/${var.github_repository}"
}
