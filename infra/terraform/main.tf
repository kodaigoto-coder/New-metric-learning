terraform {
  required_version = ">= 1.5.0"

  backend "gcs" {}

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.20"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
  zone    = var.zone
}

resource "google_compute_network" "vpc" {
  name                    = "metric-learning-vpc"
  auto_create_subnetworks = true
}

# Allow Identity-Aware Proxy (IAP) SSH & TCP Forwarding (35.235.240.0/20)
resource "google_compute_firewall" "allow_iap_ssh_and_ui" {
  name    = "metric-learning-allow-iap"
  network = google_compute_network.vpc.name

  allow {
    protocol = "tcp"
    ports    = ["22", "18501"]
  }

  source_ranges = ["35.235.240.0/20"]
  target_tags   = ["metric-learning-host"]
}

resource "google_compute_disk" "appdata_disk" {
  name = "${var.instance_name}-appdata"
  type = "pd-ssd"
  zone = var.zone
  size = var.data_disk_size_gb
}

resource "google_compute_instance" "app_vm" {
  name         = var.instance_name
  machine_type = var.machine_type
  zone         = var.zone
  tags         = ["metric-learning-host"]

  boot_disk {
    initialize_params {
      image = "ubuntu-os-cloud/ubuntu-2204-lts"
      size  = 50
      type  = "pd-balanced"
    }
  }

  attached_disk {
    source      = google_compute_disk.appdata_disk.id
    device_name = "appdata"
  }

  network_interface {
    network = google_compute_network.vpc.name
    access_config {}
  }

  service_account {
    email  = google_service_account.gce_runtime_sa.email
    scopes = ["cloud-platform"]
  }

  metadata_startup_script = <<-EOT
    #!/usr/bin/env bash
    set -euxo pipefail

    # Format and mount attached persistent disk to /mnt/disks/appdata
    DISK_DEV="/dev/disk/by-id/google-appdata"
    MOUNT_DIR="/mnt/disks/appdata"
    mkdir -p "$${MOUNT_DIR}"
    if ! blkid "$${DISK_DEV}"; then
      mkfs.ext4 -m 0 -F -E lazy_itable_init=0,lazy_journal_init=0,discard "$${DISK_DEV}"
    fi
    if ! grep -qs "$${MOUNT_DIR}" /proc/mounts; then
      mount -o discard,defaults "$${DISK_DEV}" "$${MOUNT_DIR}"
    fi
    mkdir -p "$${MOUNT_DIR}/raw" "$${MOUNT_DIR}/manifests" "$${MOUNT_DIR}/artifacts" "/opt/metric-learning"

    # Install Docker Engine and Docker Compose plugin if not installed
    if ! command -v docker >/dev/null 2>&1; then
      apt-get update
      apt-get install -y ca-certificates curl gnupg
      install -m 0755 -d /etc/apt/keyrings
      curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
      chmod a+r /etc/apt/keyrings/docker.gpg
      echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" > /etc/apt/sources.list.d/docker.list
      apt-get update
      apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
    fi
  EOT
}
