variable "project_id" {
  description = "Google Cloud Project ID"
  type        = string
}

variable "region" {
  description = "GCP region for Artifact Registry and Compute resources"
  type        = string
  default     = "us-central1"
}

variable "zone" {
  description = "GCP zone for the GCE VM and Persistent Disk"
  type        = string
  default     = "us-central1-a"
}

variable "instance_name" {
  description = "Name of the GCE VM instance hosting the 3-container metric-learning stack"
  type        = string
  default     = "metric-learning-vm"
}

variable "machine_type" {
  description = "GCE machine type (ensure sufficient RAM for trainer-service mem_limit)"
  type        = string
  default     = "e2-standard-4"
}

variable "data_disk_size_gb" {
  description = "Size of the attached Persistent Disk for /app/data (.pkl, .pt, raw images)"
  type        = number
  default     = 100
}

variable "github_repository" {
  description = "GitHub repository in 'owner/repo' format for Workload Identity Federation"
  type        = string
  default     = "example-org/New-metric-learning"
}
