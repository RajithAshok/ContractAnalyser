resource "kubernetes_deployment" "analyzer" {
  metadata {
    name      = "contract-analyzer"
    namespace = kubernetes_namespace.contract_analyzer.metadata[0].name
  }
  spec {
    replicas = 1
    selector {
      match_labels = {
        app = "contract-analyzer"
      }
    }
    template {
      metadata {
        labels = {
          app = "contract-analyzer"
        }
      }
      spec {
        container {
          name              = "contract-analyzer"
          image             = "contract-analyzer:latest"
          image_pull_policy = "Never"
          stdin             = true
          tty               = true
          env_from {
            config_map_ref {
              name = kubernetes_config_map.config.metadata[0].name
            }
          }
          resources {
            requests = {
              memory = "512Mi"
              cpu    = "250m"
            }
            limits = {
              memory = var.analyzer_memory_limit
              cpu    = "1000m"
            }
          }
          volume_mount {
            name       = "contracts"
            mount_path = "/app/contracts"
          }
        }
        volume {
          name = "contracts"
          host_path {
            path = "/contracts"
            type = "DirectoryOrCreate"
          }
        }
      }
    }
  }
}