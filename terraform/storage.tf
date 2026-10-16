resource "kubernetes_persistent_volume_claim" "ollama_models" {
  metadata {
    name      = "ollama-models-pvc"
    namespace = kubernetes_namespace.contract_analyzer.metadata[0].name
  }
  spec {
    access_modes = ["ReadWriteOnce"]
    resources {
      requests = {
        storage = var.ollama_storage
      }
    }
  }
}

resource "kubernetes_persistent_volume_claim" "chroma_data" {
  metadata {
    name      = "chroma-data-pvc"
    namespace = kubernetes_namespace.contract_analyzer.metadata[0].name
  }
  spec {
    access_modes = ["ReadWriteOnce"]
    resources {
      requests = {
        storage = var.chroma_storage
      }
    }
  }
}