resource "kubernetes_config_map" "config" {
  metadata {
    name      = "contract-analyzer-config"
    namespace = kubernetes_namespace.contract_analyzer.metadata[0].name
  }

  data = {
    OLLAMA_HOST  = "http://ollama:11434"
    CHROMA_HOST  = "http://chromadb:8000"
    MODEL_NAME   = var.ollama_model
    NUM_CTX      = "4096"
    NUM_THREAD   = "4"
  }
}