resource "kubernetes_stateful_set" "ollama" {
  metadata {
    name      = "ollama"
    namespace = kubernetes_namespace.contract_analyzer.metadata[0].name
  }
  spec {
    service_name = "ollama"
    replicas     = 1
    selector {
      match_labels = {
        app = "ollama"
      }
    }
    template {
      metadata {
        labels = {
          app = "ollama"
        }
      }
      spec {
        container {
          name  = "ollama"
          image = "ollama/ollama:latest"
          port {
            container_port = 11434
          }
          volume_mount {
            name       = "ollama-models"
            mount_path = "/root/.ollama"
          }
          resources {
            requests = {
              memory = var.ollama_memory_request
              cpu    = "500m"
            }
            limits = {
              memory = var.ollama_memory_limit
              cpu    = "2000m"
            }
          }
          readiness_probe {
            tcp_socket {
              port = 11434
            }
            initial_delay_seconds = 20
            period_seconds        = 10
          }
          liveness_probe {
            tcp_socket {
              port = 11434
            }
            initial_delay_seconds = 30
            period_seconds        = 15
          }
        }
        volume {
          name = "ollama-models"
          persistent_volume_claim {
            claim_name = kubernetes_persistent_volume_claim.ollama_models.metadata[0].name
          }
        }
      }
    }
  }
}

resource "kubernetes_service" "ollama" {
  metadata {
    name      = "ollama"
    namespace = kubernetes_namespace.contract_analyzer.metadata[0].name
  }
  spec {
    selector = {
      app = "ollama"
    }
    port {
      port        = 11434
      target_port = 11434
    }
  }
}