resource "kubernetes_stateful_set" "chromadb" {
  metadata {
    name      = "chromadb"
    namespace = kubernetes_namespace.contract_analyzer.metadata[0].name
  }
  spec {
    service_name = "chromadb"
    replicas     = 1
    selector {
      match_labels = {
        app = "chromadb"
      }
    }
    template {
      metadata {
        labels = {
          app = "chromadb"
        }
      }
      spec {
        container {
          name  = "chromadb"
          image = "chromadb/chroma:latest"
          port {
            container_port = 8000
          }
          volume_mount {
            name       = "chroma-data"
            mount_path = "/chroma/chroma"
          }
          env {
            name  = "IS_PERSISTENT"
            value = "TRUE"
          }
          env {
            name  = "ANONYMIZED_TELEMETRY"
            value = "FALSE"
          }
          resources {
            requests = {
              memory = "256Mi"
              cpu    = "250m"
            }
            limits = {
              memory = var.chroma_memory_limit
              cpu    = "500m"
            }
          }
          readiness_probe {
            tcp_socket {
              port = 8000
            }
            initial_delay_seconds = 10
            period_seconds        = 10
          }
          liveness_probe {
            tcp_socket {
              port = 8000
            }
            initial_delay_seconds = 15
            period_seconds        = 15
          }
        }
        volume {
          name = "chroma-data"
          persistent_volume_claim {
            claim_name = kubernetes_persistent_volume_claim.chroma_data.metadata[0].name
          }
        }
      }
    }
  }
}

resource "kubernetes_service" "chromadb" {
  metadata {
    name      = "chromadb"
    namespace = kubernetes_namespace.contract_analyzer.metadata[0].name
  }
  spec {
    selector = {
      app = "chromadb"
    }
    port {
      port        = 8000
      target_port = 8000
    }
  }
}